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


def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    # 在构图时是双向边
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)#节点特征
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

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
        self.linears = clones(nn.Linear(d_model, d_model), h)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:# batch_size,atom_num,pharm_num
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

class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU(),message_head=4,drop_out=0.1):
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
        self.homo_etypes = []
        self.hetero_etypes = []
        self.node_types = []
        if 'a' in view:
            self.homo_etypes.append(('a','b','a'))
            self.node_types.append('a')
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
            self.node_types.append('p')
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(message_head,hid_dim,dropout=drop_out) # 为每个视图定义多头注意力机制

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
        # 传递给节点的消息，原始节点特征，depth-1层消息传递后节点的特征
    
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}

    def forward(self,bg):

        suffix = self.suffix
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)# 对相应节点类型的节点特征进行重命名f_{suffix},额外的进行存储
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)# 对应边的特征重新命名为'h'

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        # 内置消息函数，使用边缘特征计算消息
        update_funcs = {e:(fn.copy_e('h','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        update_funcs.update({e:(fn.copy_u(f'f_junc_{suffix}','m'),
                                partial(self.msg_func, attn=self.attn[''.join(e)],
                                        field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')# 更新节点的特征
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])# 将边反向，并交换特征
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)# 边特征'm'
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)# 更新边特征
        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')
        # last update of junc feature
        bg.multi_update_all({e:(fn.copy_u(f'f_junc_{suffix}','mail'),
                                 partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
                                 cross_reducer='sum')

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True,gru_head=6,drop_out=0.1):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_jp = MultiHeadedAttention(gru_head,hid_dim,dropout=drop_out)
        self.att_apj = MultiHeadedAttention(gru_head,hid_dim,dropout=drop_out)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True,bidirectional=bidirectional)
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)# 获得每个分子中ntype类型的节点数
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])# 每个类型节点的开始节点下标
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)# 截取该i个目标分子的节点p,a特征向量
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst
    

    def split_batch_junc(self, bg, suffix, device):
        pharm_j = bg.nodes['p'].data[suffix]
        atom_j = bg.nodes['a'].data[suffix]
        p_node_size = bg.batch_num_nodes('p')# 获得每个分子中ntype类型的节点数
        a_node_size = bg.batch_num_nodes('a')# 获得每个分子中ntype类型的节点数
        p_start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(p_node_size,0)[:-1]])
        a_start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(a_node_size,0)[:-1]])
        max_num_node = max(p_node_size+a_node_size)
        hidden_lst = []
        for i in range(bg.batch_size):
            p_start, p_size = p_start_index[i],p_node_size[i]
            a_start, a_size = a_start_index[i],a_node_size[i]
            p_hidden = pharm_j.narrow(0, p_start, p_size)
            a_hidden = atom_j.narrow(0, a_start, a_size)
            cur_hidden = torch.cat((p_hidden,a_hidden),dim=0)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst
    def forward(self,bg,suffix='h'):
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)# [batch_size,max_p_len,hid_dim]
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)# [batch_size,max_a_len,hid_dim]
        junc_pharmj = self.split_batch_junc(bg, f'f_junc_{suffix}',device)

        mask = (junc_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        z_jp = self.att_jp(junc_pharmj,p_pharmj,p_pharmj,mask)
        mask = (a_pharmj!=0).type(torch.float32).matmul((z_jp.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_apj(a_pharmj, z_jp, z_jp, mask) + a_pharmj # 

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)# 初始隐状态
        h, hidden = self.gru(h,hidden)# batch , max_a_len , hid_dim*self.direction
        # hidden [self.direction,batch_size,hid_dim]
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('a')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)# 将原子级别的特征求平均作为整个分子的特征

        return graph_embed
        # batch_size,hid_dim 

class PharmEncoder(nn.Module):
    def __init__(self,args) -> None:
        super(PharmEncoder,self).__init__()
        self.hidden_size = args.hidden_size
        self.activation = get_activation_function(args.activation)
        self.dropout = args.dropout
        self.depth = args.depth
        message_head = 4
        gru_head = 6

        # atom view
        self.w_atom = nn.Linear(args.atom_fdim,self.hidden_size)
        self.w_bond = nn.Linear(args.bond_fdim,self.hidden_size)
        # pharm view
        self.w_pharm = nn.Linear(args.pharm_fdim,self.hidden_size)
        self.w_reac = nn.Linear(args.react_fdim,self.hidden_size)
        # junction view
        self.w_junc = nn.Linear(args.atom_fdim + args.pharm_fdim,self.hidden_size)

        #  define the view during massage passing
        self.mp_aug = MVMP(msg_func=add_attn,hid_dim=self.hidden_size,depth=self.depth,view='apj',suffix='aug',act=self.activation,message_head=message_head,drop_out=self.dropout)#  raw paper head = 4
        self.mp = MVMP(msg_func=add_attn,hid_dim=self.hidden_size,depth=self.depth,view='a',suffix='h',act=self.activation,message_head=message_head,drop_out=self.dropout)
        ## readout
        self.readout = Node_GRU(self.hidden_size,gru_head=gru_head,drop_out=self.dropout)# raw paper head = 6
        self.readout_attn = Node_GRU(self.hidden_size,gru_head=gru_head,drop_out=self.dropout)
        self.W_o = nn.Linear(self.hidden_size*4,self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def init_feature(self,bg):
        bg.nodes['a'].data['f'] = self.activation(self.w_atom(bg.nodes['a'].data['f']))# atom level node Hidden feature
        bg.edges[('a','b','a')].data['x'] = self.activation(self.w_bond(bg.edges[('a','b','a')].data['x']))# atom level edge Hidden feature
        
        bg.nodes['p'].data['f'] = self.activation(self.w_pharm(bg.nodes['p'].data['f']))# pharm level node Hidden feature
        bg.edges[('p','r','p')].data['x'] = self.activation(self.w_reac(bg.edges[('p','r','p')].data['x']))# pharm level edge Hidden feature
        
        bg.nodes['a'].data['f_junc'] = self.activation(self.w_junc(bg.nodes['a'].data['f_junc'])) # junction level for atom node Hidden feature [atom_dim+pharm_dim]
        bg.nodes['p'].data['f_junc'] = self.activation(self.w_junc(bg.nodes['p'].data['f_junc']))# junction level for pharm node Hidden feature

    def forward(self,step, bg, features_batch):
        self.init_feature(bg)
        self.mp(bg)
        self.mp_aug(bg)
        embed_h = self.readout(bg,'h')
        embed_aug = self.readout_attn(bg,'aug')
        embed = torch.cat([embed_h,embed_aug],1)
        embed = self.activation(self.W_o(embed))
        embed = self.dropout(embed)
        return  embed

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
        args.bond_fdim = bond_fdim or get_bond_fdim()
        args.pharm_fdim = pharm_fdim or get_pharm_fdim()
        args.react_fdim = react_fdim or get_react_fdim()
        self.graph_input = graph_input
        self.args = args
        self.encoder = PharmEncoder(self.args)

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = create_dgl_batch(batch, self.args, pretrain).to(self.args.device)
        output = self.encoder.forward(step, batch, features_batch)
        return output

