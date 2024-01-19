from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from chemprop.features import BatchMolGraph,get_atom_fdim, get_bond_fdim, mol2graph,get_pharm_fdim,get_react_fdim
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from torch_scatter import scatter_add
import pdb
import copy
from rdkit import Chem
# from chemprop.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
# from multiprocessing import Pool

class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
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
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
            
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)
        self.only_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.gru = BatchGRU(self.hidden_size)
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        # add & concat functional group features
        if self.args.add_step == 'func_frag_attention':
            self.W_i_atom_new = nn.Linear(self.atom_fdim*2, self.hidden_size, bias=self.bias)
        elif self.args.add_step =='concat_mol_frag_attention':
            self.funcional_group_embedding = FunctionalGroupEmbedding(82,self.hidden_size)
            self.W_molecular = nn.Linear(self.hidden_size*2,self.hidden_size)
    def forward(self, step, mol_graph, features_batch=None) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num,\
            f_brics, f_reacts, p2r, r2p, r2revb, p_scope, pharm_num,\
            mapping, mapping_scope,func2atom,func2atom_scope,descriptors= mol_graph.get_components()
        assert len(a_scope)==len(p_scope)
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_brics, f_reacts, p2r, r2p, r2revb, mapping, func2atom = (
                f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(),
                f_brics.cuda(), f_reacts.cuda(), p2r.cuda(), r2p.cuda(), r2revb.cuda(), 
                mapping.cuda(), func2atom.cuda())
        if self.args.step == 'func_prompt':# 使用高层次BRICS片段进行反应
            # make sure the prompt exists
            assert self.W_i_atom.prompt_generator
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            if self.args.add_step =='func_frag_attention':
                # 官能团前面有占位的填充点
                f_group = self.funcional_group_embedding(mapping)# get random functional group embedding
                pharm_hiddens,self.self_att = self.W_i_atom.prompt_generator(input_atom, a_scope, f_group, mapping_scope,func2atom,func2atom_scope)
                fg_out = torch.zeros(1, self.hidden_size).cuda()
                new_atom_lst = [fg_out]
                for i,(a_start,a_size) in enumerate(a_scope):
                    cur_a =  input_atom.narrow(0, a_start, a_size)+ pharm_hiddens[i]
                    new_atom_lst.append(cur_a)
                input_atom = torch.cat(new_atom_lst, 0)
            elif self.args.add_step =='concat_frag_attention':
                pharm_hiddens,self.self_att = self.W_i_atom.prompt_generator(None, input_atom, a_scope, f_group, group_scope, mapping, mapping_scope)
                fg_zeros = torch.zeros(1, self.hidden_size).cuda()
                fg_out = torch.zeros(1, self.hidden_size * 2).cuda()
                new_atom_lst = [fg_out]
                for i,(a_start,a_size) in enumerate(a_scope):
                    m_start,m_size = mapping_scope[i]
                    _,g_size = group_scope[i]
                    cur_g = pharm_hiddens[i].narrow(0,1,g_size-1)
                    cur_m = mapping.narrow(0,m_start,m_size)
                    cur_a = input_atom.narrow(0, a_start, a_size)
                    cur_a = torch.cat([fg_zeros,cur_a],dim=0)
                    for k in range(cur_m.shape[0]):
                        m = cur_a[cur_m[k]]
                    cur_a = torch.cat([fg_zeros,cur_a],dim=0)
                    cur_a =  torch.cat([cur_a,pharm_hiddens[i].expand(cur_a.shape[0],-1)],dim=1)
                    new_atom_lst.append(cur_a)
                new_atom_lst = torch.cat(new_atom_lst, 0)
                input_atom = self.W_i_atom_new(new_atom_lst)
            elif self.args.add_step == 'all_add_frag_attention':
                # 设置为frag-attention才能得到BRICS的片段特征，实际上没有用到attention机制
                fg_out = torch.zeros(1,self.hidden_size).cuda()
                assert len(group_scope)==len(a_scope)
                new_atom_lst = [fg_out]
                for i,(g_start,g_size) in enumerate(group_scope):
                    a_start,a_size = a_scope[i][0],a_scope[i][1]
                    cur_g = f_group.narrow(0,g_start,g_size).sum(dim=0,keepdim=True)
                    cur_a = input_atom.narrow(0,a_start,a_size)+cur_g
                    new_atom_lst.append(cur_a)
                input_atom = torch.cat(new_atom_lst, 0)
        else:
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        if self.args.add_step == '':
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)
        if self.args.step == 'func_prompt' and self.args.add_step == 'concat_mol_frag_attention':
            f_group = self.funcional_group_embedding(mapping)# get random functional group embedding
            pharm_hiddens,self.self_att = self.W_i_atom.prompt_generator(atom_hiddens, a_scope, f_group, mapping_scope,func2atom,func2atom_scope)
            mol_vecs = self.act_func(self.W_molecular(torch.cat([mol_vecs,pharm_hiddens],dim=-1))) # 300
            mol_vecs = self.dropout_layer(mol_vecs)
        return mol_vecs  # B x H

class FunctionalGroupEmbedding(nn.Module):
    def __init__(self, num_groups, features_dim):
        super(FunctionalGroupEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.Tensor(num_groups, features_dim),requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding)
    
    def forward(self, group_indices):
        # 通过索引获取对应的特征向量
        return F.embedding(group_indices, self.embedding)

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message

class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
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
        output = self.encoder.forward(step, batch, features_batch)
        return output

