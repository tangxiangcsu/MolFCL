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
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data,Vocab
from chemprop.models import build_model, build_pretrain_model,add_BRICS_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint,Early_stop
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import wandb
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
    debug(f'Splitting data with seed {args.seed}')
    # if args.separate_test_path:
    #     test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    # if args.separate_val_path:
    #     val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    # if args.separate_val_path and args.separate_test_path:
    #     train_data = data
    # elif args.separate_val_path:
    #     train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    # elif args.separate_test_path:
    #     train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    if args.separate_train_path and args.separate_val_path and args.separate_test_path:
        print('Split by seperate data')
        train_data = get_data(path=args.separate_train_path, args=args, features_path=None, logger=logger)
        val_data = get_data(path=args.separate_val_path, args=args, features_path=None, logger=logger)
        test_data = get_data(path=args.separate_test_path, args=args, features_path=None, logger=logger)
    else:
        print('='*100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            # with open(os.path.join(args.save_dir, name + '_seed{}'.format(args.seed)+ '_smiles.csv'), 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['smiles'])
            #     for smiles in dataset.smiles():
            #         writer.writerow([smiles])
            with open(os.path.join(args.save_dir, name + '_seed{}'.format(args.seed)+ '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
        #     split_indices = []
        #     for smiles in dataset.smiles():
        #         split_indices.append(indices_by_smiles[smiles])
        #         split_indices = sorted(split_indices)
        #     all_split_indices.append(split_indices)
        # with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
        #     pickle.dump(all_split_indices, f)


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
        if args.logger:
            wandb.init(project='MolCLBRICS',group=args.task_name,name=f'run_{args.seed}',reinit=True)
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
        
        if args.step == 'BRICS_prompt':
            add_BRICS_prompt(model, args)
        # if args.freeze:
        #     for name,parameter in model.encoder.named_parameters():
        #         if 'prompt_generator' not in name and 'funcional_group_embedding' not in name:
        #             parameter.requires_grad = False
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
            
            if args.logger:
                wandb.log(
                {
                'train_loss':round(avg_loss,4),
                f'valid_{args.metric}':round(avg_val_score,4),
                f'test_{args.metric}':round(avg_test_score,4),
                }
                )
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

    return ensemble_scores

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
    
    if args.logger:
        wandb.init(project='Mol_BRICSCL',group='pretrain',name=f'run',reinit=True)

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
                    emb1 = model1(step, False, batch, None) # 原始图
                    emb2 = model2(step, True, batch, None) # 增强图
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
                    emb1 = model1(step, False, test_batch, None) # 原始图
                    emb2 = model2(step, True, test_batch, None) # 增强图
                    loss = criterion(emb1, emb2)
                    total_test_loss+=loss.item()
            logger.info(f'{global_step} test loss {total_test_loss/len(test_loader):.4f}')
            snapshot(model1, epoch, dump_folder,'original')
            snapshot(model2, epoch, dump_folder,'PharmHGT_augment')
            if args.logger:
                wandb.log({'train_loss':total_loss/len(train_loader),
                           'test_loss':total_test_loss/len(test_loader)})
            logger.info(f'[{epoch}/{args.epochs}] train loss {total_loss/len(train_loader):.4f}')

    if args.logger:
        wandb.finish()

def run_hyper_opt(args:Namespace,pretrain: bool = False,logger:Logger = None):
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
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')
    
    
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
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
        if args.logger:
            wandb.init(project='MolCLBRICS',group=args.task_name,name=f'run_{args.seed}',reinit=True)
        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model from {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model = build_model(args, encoder_name=args.encoder_name)
            model_state_dict = model.encoder.state_dict()
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
            model.encoder.load_state_dict(model_state_dict)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, encoder_name=args.encoder_name)
        
        if args.step == 'BRICS_prompt':
            add_BRICS_prompt(model, args)

        
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
        #scheduler = build_lr_scheduler(optimizer, args)

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
                scheduler=None,
                args=args,
                n_iter=n_iter,
                logger=logger,
            )
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
            '''test_preds = predict(
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
            avg_test_score = np.nanmean(test_scores)'''
            
            if args.logger:
                wandb.log(
                {
                'train_loss':round(avg_loss,4),
                f'valid_{args.metric}':round(avg_val_score,4),
                }
                )
            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args) 
            if args.early_stop:
                early_stop = stopper.step(avg_val_score)
                info(f'Epoch{epoch+1}/{args.epochs},valid_{args.metric} = {avg_val_score:.6f},\
                    best_epoch = {best_epoch+1},patience = {stopper.counter}')
            else:
                info(f'Epoch{epoch+1}/{args.epochs},valid_{args.metric} = {avg_val_score:.6f},\
                    best_epoch = {best_epoch+1}')
            if args.early_stop and early_stop:
                break
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        
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
        info(f'Model {model_idx} test {args.metric} = {avg_val_score:.6f}')
    return avg_val_score,model

def run_tSNE(args:Namespace,pretrain: bool = False, logger: Logger = None):

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================


    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=None)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug = print
    info = print
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        print('='*100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    
    if args._all:
        test_data = data
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
    
    print(f'Total size = {len(data):,} | '
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

    if args.scaffold_tSNE:
        scaffold_to_indices = scaffold_to_smiles(test_data.mols(), use_indices=True)
        indices_to_scaffold = defaultdict(str)
        all_scaffold_num = len(scaffold_to_indices.keys())
        scaffold_to_ids = {s:i for i,s in enumerate(scaffold_to_indices.keys())}
        for key,value in scaffold_to_indices.items():
            for i in value:
                indices_to_scaffold[i]=key
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        model = load_checkpoint(args.checkpoint_path,current_args=args, cuda=args.cuda, logger=logger)
        model = model.encoder
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()
        model.eval()
        num_iters, iter_step = len(test_data), args.batch_size
        batch_size = args.batch_size
        molecular_f = []
        for i in range(0, num_iters, iter_step):
            # Prepare batch
            mol_batch = MoleculeDataset(test_data[i:i + batch_size])
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()
            # Run model
            batch = smiles_batch
            step = 'finetune'
            with torch.no_grad():
                molecular_feat = model(step, pretrain, batch, features_batch)
            molecular_feat = molecular_feat.data.cpu().numpy()
            molecular_f.extend(molecular_feat)
        if args.scaffold_tSNE:
            num_tasks = all_scaffold_num
        else:
            num_tasks = args.num_tasks
        valid_targets = [[] for _ in range(num_tasks)]
        valid_f = [[] for _ in range(num_tasks)]
        if args.scaffold_tSNE:
            for i in range(num_tasks):
                for j in range(len(molecular_f)):
                    if scaffold_to_ids[indices_to_scaffold[j]]==i:
                        valid_f[i].append(molecular_f[j])
                        valid_targets[i].append(scaffold_to_ids[indices_to_scaffold[j]])
            return valid_f,valid_targets,scaffold_to_indices.keys()
        else:
            for i in range(num_tasks):
                for j in range(len(molecular_f)):
                    if test_targets[j][i] is not None:  # Skip those without targets
                        valid_f[i].append(molecular_f[j])
                        valid_targets[i].append(test_targets[j][i])
            return valid_f,valid_targets
        
def save_fg_feature(args:Namespace, logger: Logger = None):
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
    
    if args.logger:
        wandb.init(project='Mol_BRICSCL',group='pretrain',name=f'run',reinit=True)

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
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')


    args.data_size = len(data)
    
    debug(f'Total size = {len(data)}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
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

        debug(model)
        debug(f'Number of M2 parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()
        
        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()
        train_loader = DataLoader(smiles,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=12,
                            drop_last=False)
        # pca = PCA(n_components=182)
        # Run training
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            debug = logger.debug if logger is not None else print
            total_loss = 0
            step = 'pretrain'
            for batch in tqdm(train_loader):
                model.eval()
                model.encoder(step, False, batch, None) # 增强图
            
            func_save = model.encoder.encoder.func_save
            func_num = model.encoder.encoder.func_num
            s2func_f ={}
            for k1,v1 in func_save.items():
                if func_num[k1]>1:
                    s2func_f[k1] = (v1/(func_num[k1]-1)).data.cpu()
                else:
                    s2func_f[k1] = torch.randn([1,300])
            pickle.dump(s2func_f, open(f'./embedding/func2embedding.pkl','wb'))

def run_visualization(args:Namespace,pretrain: bool = False, logger: Logger = None):
    from chemprop.features.featurization import match_group
    from rdkit import Chem
    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================


    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=None)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug = print
    info = print
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        print('='*100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    test_data = data
    if args._all:
        test_data = data
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
    
    print(f'Total size = {len(data):,} | '
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

    if args.scaffold_tSNE:
        scaffold_to_indices = scaffold_to_smiles(test_data.mols(), use_indices=True)
        indices_to_scaffold = defaultdict(str)
        all_scaffold_num = len(scaffold_to_indices.keys())
        scaffold_to_ids = {s:i for i,s in enumerate(scaffold_to_indices.keys())}
        for key,value in scaffold_to_indices.items():
            for i in value:
                indices_to_scaffold[i]=key
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        model = load_checkpoint(args.checkpoint_path,current_args=args, cuda=args.cuda, logger=logger)
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()
        model.eval()
        num_iters, iter_step = len(test_data), args.batch_size
        batch_size = args.batch_size
        molecular2predict = {}
        molecular2target = {}
        molecular2att = {}
        molecular2frage_atom = {}
        molecular2funcname = {}
        for i in range(0, num_iters, iter_step):
            # Prepare batch
            if i + args.batch_size > len(test_data):
                break
            mol_batch = MoleculeDataset(test_data[i:i + batch_size])
            smiles_batch, features_batch,targets = mol_batch.smiles(), mol_batch.features(),mol_batch.targets()
            # Run model
            batch = smiles_batch
            step = 'finetune'
            with torch.no_grad():
                m_predict = model(step, pretrain, batch, features_batch)
            self_att = model.encoder.encoder.self_att[:,0,:].data.cpu().numpy()
            m_predict = m_predict.data.cpu().numpy()
            for j in range(batch_size):
                smiles = smiles_batch[j]
                mol = Chem.MolFromSmiles(smiles)
                f_feat,frags_idx_lst,smart2name_lst = match_group(mol)
                if len(frags_idx_lst)>0:
                    molecular2att[smiles] = self_att[j]
                    molecular2predict[smiles] = m_predict[j]
                    molecular2frage_atom[smiles] = frags_idx_lst
                    molecular2target[smiles] = targets[j]
                    molecular2funcname[smiles] = smart2name_lst
        pickle.dump(molecular2att, open(f'./visualization_file/{args.exp_id}_molecular2att.pkl','wb'))
        pickle.dump(molecular2predict, open(f'./visualization_file/{args.exp_id}molecular2predict.pkl','wb'))
        pickle.dump(molecular2frage_atom, open(f'./visualization_file/{args.exp_id}molecular2frage_atom.pkl','wb'))
        pickle.dump(molecular2target, open(f'./visualization_file/{args.exp_id}molecular2target.pkl','wb'))
        pickle.dump(molecular2funcname, open(f'./visualization_file/{args.exp_id}molecular2funcname.pkl','wb'))