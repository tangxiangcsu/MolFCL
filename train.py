import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-time independent runs"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.num_runs = len(init_seed)
    task_names = get_task_names(args.data_path)
    # Run training on different random seeds for each run
    all_scores = []
    for run_num in range(args.num_runs):
        args.seed = init_seed[run_num]
        info(f'Run {args.seed}')
        # args.separate_train_path=f"./KPGT/{args.exp_id}/scaffold-{run_num}-train.csv"
        # args.separate_val_path=f"./KPGT/{args.exp_id}/scaffold-{run_num}-val.csv"
        # args.separate_test_path=f"./KPGT/{args.exp_id}/scaffold-{run_num}-test.csv"
        args.save_dir = os.path.join(save_dir, f'run_{args.seed}')
        makedirs(args.save_dir)
        model_scores = run_training(args, args.pretrain, logger)# 设置官能团提示增强
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    info(f'{args.num_runs}-time runs')

    # Report scores for each run
    for run_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed[run_num]} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed[run_num]} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    avg_scores = sorted(avg_scores,reverse=not args.minimize_score)[:3]
    print(avg_scores)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                    f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = "./data/esol.csv"
    args.metric = "rmse"
    args.dataset_type = "regression" # classification  regression
    args.split_type = "scaffold_balanced"# scaffold_balanced
    args.exp_name = "finetune"
    args.exp_id = "esol" 
    args.logger = False
    args.checkpoint_path = "./ckpt/original_MoleculeModel_0702_0946_40th_epoch.pkl"# 6.4的预训练模型时构建的子结构反应图来实现的
    args.encoder = False
    args.gpu = 3
    args.epochs = 100
    args.pretrain = False
    args.atom_messages =  True
    args.increase_parm = 1 # 提示生成器学习率调整参数
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs =2
    args.hidden_size = 300
    args.ffn_hidden_size = 300
    args.add_reactive = False
    args.add_step = 'concat_mol_frag_attention'# ['concat_mol_frag_attention','add_attention','cat_attention','add','FNC_encoder']
    '''
    molecular_motif_cross_att: 分子和motif交叉注意力机制
    add_frag_attention: 设置一个可学习的向量，将初始化的分子片段信息通过自注意力机制学习到该向量中，再将其累加到所有的原子向量中
    add_attention: we run a high view of Pharm and add a learning vector, add the vector to all atom
    cat_attention: we concat graph,pharm,atom feature to attention the three f ,only get the graph to MLP layer
    FNC_encoder: 
    molecular_motif_cross_frag_atttention
    concat_frag_attention: 设置一个可学习的向量，将初始化的分子片段信息通过自注意力机制学习到该向量中，再将其concat到所有的原子向量中
    multi_frag_attention: 将片段应用乘积注意力机制
    add_BRICS_attention: 我们设想PharmHGT的模型学的足够好，那么当我们给定新的分子的时候，能够得到它的BRICS嵌入表示
    '''
    args.early_stop = False
    args.patience = 30
    args.last_early_stop = 0
    args.step = 'BRICS_prompt'# BRICS_prompt
    args.batch_size = 256
    args.self_att_hidden = 32 # default 32
    '''
    bbbp: [8,7,4]
    tox21: [8,2,6]
    toxcast: [5,2,4]
    sider: [9,14,17]
    clintox: [15,19,3]
    bace: [19,11,14]
    esol: [8,4,7]
    freesolv: [7,3,8]
    lipo: [4,3,1]
    '''
    args.freeze = False# 当冻结编码器模型时，我们只能更新预测层的模型参数和提示生产器的模型（optional）
    args.seed = [8]
    args.save_smiles_splits = False# 是否保存划分后的数据集
    args.ffn_num_layers = 2
    args.mask_self = False
    args.mapping = False
    args.encoder_name = "CMPNN" # ['CMPNN', 'MPNN','PharmHGT','HyperGraph','CMPNDGL']
    args.ffn_drop_out = 0.1
    args.encoder_drop_out = 0.1
    args.prompt_drop_out = 0.1
    args.dropout = 0.1
    args.l2_norm = 0
    args.encoder_head = 4
    args.gru_head = 6
    args.depth = 3
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    args.num_attention = 2
    mean_auc_score, std_auc_score = run_stat(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')
