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
from collections import defaultdict
# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)  # 不影响
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True,args:Namespace=None):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        self.args = args
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6,hid_dim)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
        if self.args.add_step =='FNC_encoder':
            self.fnc_encoder = FNC_encoder(self.hid_dim)
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)

        return hidden_lst
        
    def forward(self,bg,suffix='h',smiles=[]):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        smiles2BRICS_f = defaultdict()
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)
        if self.args.add_step == 'add_BRICS_attention' and self.args.step == 'BRICS_prompt': # 测试提取片段特征，用于下游任务微调
            assert len(smiles)==p_pharmj.shape[0]
            hidden = bg.nodes['p'].data[f'f_{suffix}']
            node_size = bg.batch_num_nodes('p')
            start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
            for i  in range(bg.batch_size):
                start, size = start_index[i],node_size[i]
                assert size != 0, size
                cur_hidden = hidden.narrow(0, start, size)
                smiles2BRICS_f[smiles[i]] = cur_hidden
        
        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj,mask) + a_pharmj

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)
        return graph_embed,smiles2BRICS_f

class FNC_encoder(nn.Module):
    def __init__(self,hid_dim):
        super(FNC_encoder,self).__init__()
        self.att = MultiHeadedAttention(6,hid_dim*2)
        self.W_p = nn.Linear(hid_dim,hid_dim*2)
        self.W_a = nn.Linear(hid_dim,hid_dim*2)
        self.dropout = nn.Dropout(p=0.1)
        self.act = get_activation_function('ReLU')
    def split_batch(self,bg,graph_embed,field,device):
        
        pharm__hidden = self.act(self.W_p(bg.nodes['p'].data[field]))
        atom_hidden = self.act(self.W_a(bg.nodes['a'].data[field])) # 300
        pharm_node_size = bg.batch_num_nodes('p')
        atom_node_size = bg.batch_num_nodes('a')

        pharm_start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(pharm_node_size,0)[:-1]])
        atom_start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(atom_node_size,0)[:-1]])
        max_num_node = max(atom_node_size+pharm_node_size)+1
        hidden_lst = []
        for i in range(bg.batch_size):
            graph_e = graph_embed.narrow(0,i,1)
            pharm_e = pharm__hidden.narrow(0,pharm_start_index[i],pharm_node_size[i])
            atom_e = atom_hidden.narrow(0,atom_start_index[i],atom_node_size[i])
            cur_hidden = torch.cat([graph_e,pharm_e,atom_e],dim=0)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst,0)
        return hidden_lst
    def forward(self,bg,graph_embed,suffix='h'):
        self.suffix = suffix
        all_cat = self.split_batch(bg,graph_embed,f'f_{suffix}',bg.device)
        mask = (all_cat!=0).type(torch.float32).matmul((all_cat.transpose(-1,-2)!=0).type(torch.float32))==0
        graph_emd = self.att(all_cat,all_cat,all_cat,mask)
        return graph_emd[:,0,:]

class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU()):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP,self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act
        self.homo_etypes = [('a','b','a')]
        self.hetero_etypes = []
        self.node_types = ['a','p']
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
    def update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}
    
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}

    def forward(self,bg):
        suffix = self.suffix
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        update_funcs = {e:(fn.copy_e('h','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        update_funcs.update({e:(fn.copy_src(f'f_junc_{suffix}','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')

        # last update of junc feature
        bg.multi_update_all({e:(fn.copy_src(f'f_junc_{suffix}','mail'),
                                 partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
                                 cross_reducer='sum')

class PharmEncoder(nn.Module):
    def __init__(self,args):
        super(PharmEncoder,self).__init__()
        hid_dim = args.hidden_size
        self.act = get_activation_function(args.activation)
        self.depth = args.depth
        # init
        # atom view
        self.w_atom = nn.Linear(args.atom_fdim,hid_dim)
        self.w_bond = nn.Linear(args.bond_fdim,hid_dim)
        # pharm view
        self.w_pharm = nn.Linear(args.pharm_fdim,hid_dim)
        self.w_reac = nn.Linear(args.react_fdim,hid_dim)
        # junction view
        self.w_junc = nn.Linear(args.atom_fdim + args.pharm_fdim,hid_dim)

        ## define the view during massage passing
        self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='a',suffix='h',act=self.act)
        self.mp_aug = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='ap',suffix='aug',act=self.act)
        # define ablation embedding
        # self.embedding = nn.Embedding(34,34)
        ## readout
        self.readout = Node_GRU(hid_dim,args=args)
        self.readout_attn = Node_GRU(hid_dim,args=args)

        self.W_o = nn.Linear(hid_dim*4,hid_dim)
        self.dropout = nn.Dropout(args.dropout)

    def init_feature(self,bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        # bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(torch.sum(self.embedding(bg.edges[('p','r','p')].data['x']) * bg.edges[('p','r','p')].data['x'].unsqueeze(-1),dim=1)))
        bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))
        
    def forward(self,step, bg, features_batch, smiles):
        """
        Args:
            bg: a batch of graphs
        """
        self.init_feature(bg)
        self.mp(bg)
        self.mp_aug(bg)
        embed_f,smiles2BRICS_f_1 = self.readout(bg,'h',smiles)
        embed_aug,smiles2BRICS_f = self.readout_attn(bg,'aug',smiles)
        embed = torch.cat([embed_f,embed_aug],1)
        embed = self.act(self.W_o(embed))
        embed = self.dropout(embed)
        return embed,smiles2BRICS_f
    
class PharmHGT(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 pharm_fdim: int = None,
                 react_fdim: int = None,
                 graph_input: bool = False):
        super(PharmHGT, self).__init__()
        
        args.atom_fdim = atom_fdim or get_atom_fdim()
        args.bond_fdim = bond_fdim or get_bond_fdim()+ (not args.atom_messages) * args.atom_fdim
        args.pharm_fdim = pharm_fdim or get_pharm_fdim()
        args.react_fdim = react_fdim or get_react_fdim()+ (not args.atom_messages) * args.pharm_fdim
        self.graph_input = graph_input
        self.args = args
        self.encoder = PharmEncoder(self.args)

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch,smiles = create_dgl_batch(batch, self.args, pretrain)
            batch = batch.to(self.args.device)
        output,smiles2BRICS_f = self.encoder.forward(step, batch, features_batch,smiles)
        if self.args.step=='BRICS_prompt' and self.args.add_step =='add_BRICS_attention':
            return output,smiles2BRICS_f # 这个只用在Save_BRICS_Feature，其余的我们不通过这一层
        else:
            return output