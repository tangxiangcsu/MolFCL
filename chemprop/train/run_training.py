from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data,load_data
from chemprop.models import build_model, build_pretrain_model,add_FUNC_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint,Early_stop
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.data.scaffold import scaffold_to_smiles
from collections import defaultdict
import pickle

def run_training(args: Namespace, pretrain: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================

    # Get data
    info('Loading data')
    # args.vocab = Vocab(args)
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')
    
    
    # Split data
    debug(f'Load data from {args.exp_id} for Scaffold-{args.runs}')
    if 0<=args.runs<3:
        train_data, val_data, test_data = load_data(data,args,logger)
    else:
        print('='*100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model from {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model = build_model(args, encoder_name=args.encoder_name)
            model_state_dict = model.encoder.state_dict() if args.encoder else model.state_dict()
            pretrained_state_dict = {}
            for param_name in checkpoint.keys():
                if param_name not in model_state_dict:
                    print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
                elif model_state_dict[param_name].shape != checkpoint[param_name].shape:
                    print(f'Pretrained parameter "{param_name}" '
                    f'of shape {checkpoint[param_name].shape} does not match corresponding '
                    f'model parameter of shape {model_state_dict[param_name].shape}.')
                else:
                    pretrained_state_dict[param_name] = checkpoint[param_name]
            model_state_dict.update(pretrained_state_dict)
            if args.encoder:
                model.encoder.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, encoder_name=args.encoder_name)
        
        if args.step == 'func_prompt':
            add_FUNC_prompt(model, args)
        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Early_stop
        early_stop = False

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        if args.early_stop:
            stopper = Early_stop(patience=args.patience,minimize_score=args.minimize_score)
        for epoch in range(args.epochs):
            avg_loss = train(
                model=model,
                pretrain=pretrain,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            
            val_scores = evaluate(
                model=model,
                pretrain=pretrain,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            test_preds = predict(
                model=model,
                pretrain=pretrain,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args) 
        
            if args.early_stop and epoch>= args.last_early_stop:
                early_stop = stopper.step(avg_val_score)
                info(f'Epoch{epoch+1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
                    best_epoch = {best_epoch+1},patience = {stopper.counter}')
            else:
                info(f'Epoch{epoch+1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
                    best_epoch = {best_epoch+1}')
            if args.early_stop and early_stop:
                break
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'),current_args=args, cuda=args.cuda, logger=logger)
        
        test_preds = predict(
            model=model,
            pretrain=pretrain,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return avg_ensemble_test_score

def pre_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.set_device(args.gpu)

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)


    args.data_size = len(data)
    
    debug(f'Total size = {len(data)}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model1 = load_checkpoint(args.checkpoint_paths[0], args=args, logger=logger,encoder_name='CMPNN')
            model2 = load_checkpoint(args.checkpoint_paths[1], args=args, logger=logger,encoder_name='PharmHGT')
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='PharmHGT')
        

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model2 = model2.cuda()

        #logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{args.save_dir}/model'
        
        # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        # args.device = device
        criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        optimizer = Adam([{"params": model1.parameters()},{"params": model2.parameters()}], lr=3e-5)
        scheduler = ExponentialLR(optimizer, 0.99, -1)
        step_per_schedule = 500
        global_step = 0
        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()
        train_size = int(0.98 * len(smiles))
        test_size = len(smiles)-train_size
        train_smiles,test_smiles = torch.utils.data.random_split(smiles,[train_size,test_size])
        train_loader = DataLoader(train_smiles,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=12,
                            drop_last=True)
        test_loader = DataLoader(test_smiles,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=12,
                            drop_last=True)
        # Run training
        for epoch in range(args.start_epochs,args.end_epochs):
            debug(f'Epoch {epoch}')

            debug = logger.debug if logger is not None else print
            total_loss = 0
            step = 'pretrain'
            with tqdm(total=len(train_loader)) as t:
                for batch in train_loader:
                    model1.train()
                    model2.train()
                    # Run model
                    emb1 = model1(step, False, batch, None) # Original graph
                    emb2 = model2(step, True, batch, None) # Augmented graph
                    loss = criterion(emb1, emb2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    t.set_description('Epoch[{}/{}]'.format(epoch+1, args.end_epochs))
                    t.set_postfix(train_loss=loss.item())
                    t.update()
                    total_loss+=loss.item() 
                    if global_step % step_per_schedule == 0:
                        scheduler.step()
            # save model   
            model1.eval()
            model2.eval()
            total_test_loss = 0
            for test_batch in test_loader:
                with torch.no_grad():
                    emb1 = model1(step, False, test_batch, None) # Original graph
                    emb2 = model2(step, True, test_batch, None) # Augmented graph
                    loss = criterion(emb1, emb2)
                    total_test_loss+=loss.item()
            logger.info(f'{global_step} test loss {total_test_loss/len(test_loader):.4f}')
            snapshot(model1, epoch, dump_folder,'original')
            snapshot(model2, epoch, dump_folder,'PharmHGT_augment')
            logger.info(f'[{epoch}/{args.epochs}] train loss {total_loss/len(train_loader):.4f},test loss {total_test_loss/len(test_loader):.4f}')