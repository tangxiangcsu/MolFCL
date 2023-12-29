import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from functools import partial
import copy
from argparse import Namespace
from typing import List, Union
import math
from chemprop.nn_utils import get_activation_function
from chemprop.features import create_dgl_batch 
from chemprop.features import BatchMolGraph,get_atom_fdim, get_bond_fdim, mol2graph,get_pharm_fdim,get_react_fdim
import numpy as np

class CMPNDGLEncoder(nn.Module):
    def __init__(self,args):
        super(CMPNDGLEncoder,self).__init__()

        self.depth = args.depth
        self.args = args
        self.hidden_size = args.hidden_size
        self.bias = args.bias


        # Dropout
        self.dropout_layer = nn.Dropout(p=args.encoder_drop_out)
        # Activation
        self.act_func = get_activation_function(args.activation)
        
        # Input
        input_dim = args.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = args.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + args.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(self.hidden_size*2,self.hidden_size)

        self.gru = BatchGRU(self.args)

        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        
        # dont use
        self.cls = nn.Parameter(torch.randn(1,133), requires_grad=True)
        self.W_i_atom_new = nn.Linear(args.atom_fdim*2, self.hidden_size, bias=self.bias)

    def atom_message_pass(self,node,field):
        agg_messge = node.mailbox['m'] #传递的消息
        return {field: node.data['f'] + (agg_messge.sum(dim=1) * agg_messge.max(dim=1)[0])}
    
    def reverse_edge(self,tensor):
        n = tensor.size(0)
        assert n%2 ==0
        delta = torch.ones(n).type(torch.long)
        delta[torch.arange(1,n,2)] = -1
        return tensor[delta+torch.tensor(range(n))]

    def update_node(self,node,field):
        agg_messge = node.mailbox['mail'] #传递的消息
        return {field: agg_messge.sum(dim=1) * agg_messge.max(dim=1)[0]}

    def update_bond_message(self,edge,layer,init_bond_feat):
        return {'e':self.dropout_layer(self.act_func(init_bond_feat+layer(edge.src['f'] - edge.data['rev_h'])))}

    def forward(self,step, bg, features_batch):
        f_atoms = bg.nodes['a'].data['f']
        f_bonds = bg.edges[('a','b','a')].data['x']
        if self.args.step=='BRICS_prompt':
            assert self.W_i_atom.prompt_generator
            bg.nodes['a'].data['f'] = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            bg = self.W_i_atom.prompt_generator(bg)
        else:
            bg.nodes['a'].data['f'] = self.W_i_atom(f_atoms)
        input_atom = bg.nodes['a'].data['f']
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        bg.nodes['a'].data['f'] = input_atom
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        bg.edges[('a','b','a')].data['x'] = self.act_func(input_bond)
        for depth in range(self.depth - 1):
            bg.update_all(fn.copy_e('x','m'),partial(self.atom_message_pass,field='f'),etype=('a','b','a'))
            bg.edges[('a','b','a')].data['rev_h']=self.reverse_edge(bg.edges[('a','b','a')].data['x'])
            bg.apply_edges(partial(self.update_bond_message,layer=self._modules[f'W_h_{depth}'],init_bond_feat = message_bond),etype=('a','b','a'))
        bg.update_all(fn.copy_e('x','mail'),partial(self.update_node,field='messge'),etype=('a','b','a'))
        atom_message = self.lr(torch.cat([bg.nodes['a'].data['messge'],bg.nodes['a'].data['f'],message_atom],dim=1))
        graph_emb = self.gru(bg,atom_message)
        graph_emb = self.dropout_layer(self.act_func(self.W_o(graph_emb)))
        return graph_emb

class BatchGRU(nn.Module):
    def __init__(self, args:Namespace):
        super(BatchGRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))
        self.args = args
        self.direction = 2
    def split_batch(self,bg,message,ntype,device):
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = message.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst
    
    def forward(self,bg,atom_message):
        h = self.split_batch(bg,atom_message,'a',self.args.device)
        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('a')
        start_index = torch.cat([torch.tensor([0],device=self.args.device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hidden_size).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)
        return graph_embed

class CMPNDGL(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 pharm_fdim: int = None,
                 react_fdim: int = None,
                 graph_input: bool = False):
        super(CMPNDGL, self).__init__()
        args.atom_fdim = atom_fdim or get_atom_fdim()
        args.bond_fdim = bond_fdim or get_bond_fdim() +\
                    (not args.atom_messages) * args.atom_fdim
        args.pharm_fdim = pharm_fdim or get_pharm_fdim()
        args.react_fdim = react_fdim or get_react_fdim() +\
                    (not args.atom_messages) * args.pharm_fdim
        self.graph_input = graph_input
        self.args = args
        self.encoder = CMPNDGLEncoder(self.args)

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = create_dgl_batch(batch, self.args, pretrain).to(self.args.device)
        output = self.encoder.forward(step, batch, features_batch)
        return output