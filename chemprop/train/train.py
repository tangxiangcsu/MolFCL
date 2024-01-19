from argparse import Namespace
import logging
from typing import Callable, List, Union

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange
import wandb
from torch.utils.data import DataLoader
from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
import pdb

def train(model: nn.Module,
          pretrain: bool,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: bool = False) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """

    model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        step = 'finetune'
        # Run model
        model.zero_grad()
        preds = model(step, pretrain, batch, features_batch)
        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            if args.l2_norm>0:
                l2_norm = torch.norm(model.encoder.encoder.funcional_group_embedding.embedding,p=2,dim=1)
                loss = loss_func(preds, targets) * class_weights * mask  + args.l2_norm*torch.mean(l2_norm)
            else:
                loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        
        loss_sum += loss.item()
        iter_count += 1

        loss.backward(retain_graph=True)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)
    return loss_sum / iter_count
