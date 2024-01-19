import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np

from chemprop.train.run_training import pre_training
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp

def pretrain(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    pre_training(args, logger)

if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = './data/zinc15_250K.csv'
    args.gpu = 3
    args.start_epochs = 0
    args.end_epochs = 50
    args.batch_size = 1024
    args.exp_id = 'pretrain'
    args.logger = True
    args.ffn_drop_out = 0.1
    args.dropout = 0.1
    args.pretrain = True
    args.encoder_drop_out = 0.1
    args.encoder_head = 4
    args.gru_head = 6
    args.add_step=''
    args.step =''
    args.save_BRICS_f = False
    args.atom_messages = True
    modify_train_args(args)
    args.checkpoint_paths = None
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    pretrain(args, logger)
