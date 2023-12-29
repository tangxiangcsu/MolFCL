from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph,get_atom_fdim, get_bond_fdim, mol2graph,get_pharm_fdim,get_react_fdim
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from torch_scatter import scatter_add
import pdb
import copy
from rdkit import Chem
with open('./chemprop/data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
    smart2name = dict(zip(smart, name))
    func2index = {smart2name[sm]:i for i,sm in enumerate(smart)}
    idex2func = {value:key for key,value in func2index.items()}
class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.encoder_drop_out
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        self.func_save = {smart2name[sm]:torch.zeros([1,300]).cuda() for sm in smart}
        self.func_num = {smart2name[sm]:1 for sm in smart}
    def forward(self, step, mol_graph, features_batch=None) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num,\
            f_brics, f_reacts, p2r, r2p, r2revb, p_scope, pharm_num,\
            mapping, mapping_scope,func2atom,func2atom_scope= mol_graph.get_components()
        assert len(a_scope)==len(p_scope)
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_brics, f_reacts, p2r, r2p, r2revb, mapping, func2atom= (
                f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(),
                f_brics.cuda(), f_reacts.cuda(), p2r.cuda(), r2p.cuda(), r2revb.cuda(), 
                mapping.cuda(), func2atom.cuda())
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        padding_zero = torch.zeros((1,self.hidden_size)).cuda()
        for i,(m_start,m_size) in enumerate(func2atom_scope):
            cur_m = func2atom.narrow(0,m_start,m_size) #  
            m_start,m_size = mapping_scope[i]
            a_start,a_size = a_scope[i]
            cur_a = input_atom.narrow(0,a_start,a_size)
            cur_a = torch.cat([padding_zero,cur_a],dim=0) #
            cur_a = cur_a[cur_m]
            cur_a = cur_a.sum(dim=1)
            fg_name_index = mapping.narrow(0,m_start,m_size)
            for i,fg_name_index in enumerate(fg_name_index):
                self.func_save[idex2func[int(fg_name_index)]]+=cur_a[i].unsqueeze(0)
                self.func_num[idex2func[int(fg_name_index)]]+=1

class CMPN_ONLY_ATOM(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN_ONLY_ATOM, self).__init__()
        args.atom_fdim = atom_fdim or get_atom_fdim()
        args.bond_fdim = bond_fdim or get_bond_fdim() + \
                            (not args.atom_messages) * args.atom_fdim # * 2
        args.pharm_fdim = get_pharm_fdim()
        args.react_fdim = get_react_fdim() + (not args.atom_messages) * args.pharm_fdim # * 2
        self.graph_input = graph_input
        self.args = args
        self.atom_fdim = args.atom_fdim
        self.bond_fdim = args.bond_fdim
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args, pretrain)
        self.encoder.forward(step, batch, features_batch)

