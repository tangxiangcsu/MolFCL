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
from chemprop.features import BatchMolGraph,get_atom_fdim, get_bond_fdim, mol2graph,get_pharm_fdim,get_react_fdim,create_dglHyper_batch
import numpy as np

test = 'atom_fg_hyper'

def add_attn(node,field,attn,message_filed):
    feat = node.data[field].unsqueeze(1)
    return {field: (attn(feat,node.mailbox[message_filed],node.mailbox[message_filed])+feat).squeeze(1)}

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

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), h)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
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

class HyperGraphEncoder(nn.Module):
    def __init__(self,atom_fdim,bond_fdim,pharm_fdim,react_fdim,args:Namespace):
        super(HyperGraphEncoder,self).__init__()
         # Dropout
        self.dropout_layer = nn.Dropout(p=args.encoder_drop_out)
        self.hidden_size = args.hidden_size
        self.pharm_fdim = pharm_fdim
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.react_fdim = react_fdim
        self.depth = args.depth
        self.bias = args.bias
        '''
        三级操作
        （1）原子级别，使用CMPNN算法，更新原子级别特征和键特征
        （2）碎片级别：通过，碎片的特征取决于所属原子的强度
        （3）超图级别：超图含有反应信息，且超边包含多个片段，使用注意力机制更新超边信息

        反向更新：
        （1）一个碎片节点可能属于多个超边，超边对碎片节点的作用强度不一致，使用注意力机制，更新碎片节点
        
        ''' 
        # atom view
        self.w_atom = nn.Linear(self.atom_fdim,self.hidden_size)
        self.w_bond = nn.Linear(self.bond_fdim,self.hidden_size)
        for depth in range(self.depth-1):
            self._modules[f'W_h_atom_{depth}'] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        
        # fg view
        self.w_pharm = nn.Linear(self.pharm_fdim,self.hidden_size)
        self.fg_att = MultiHeadedAttention(args.encoder_head,self.hidden_size,args.encoder_drop_out)
        self.fg_lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        # hyperGraph
        self.hyper_reac = nn.Linear(self.react_fdim,self.hidden_size)
        self.w_hypergraph = nn.Linear(self.pharm_fdim,self.hidden_size)
        self.hyper_att = MultiHeadedAttention(args.encoder_head,self.hidden_size,args.encoder_drop_out)
        self.hyper2fg_att = MultiHeadedAttention(args.encoder_head,self.hidden_size,args.encoder_drop_out)
        self.hyper_reac_hyper = MultiHeadedAttention(args.encoder_head,self.hidden_size,args.encoder_drop_out)
        self.hyper_lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        for depth in range(self.depth-1):
            self._modules[f'W_h_hyper_{depth}'] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        # Activation
        self.act = get_activation_function(args.activation)

        # read out
        self.a2p_gru = Batch_GRU(args)
        self.p2hyper_gru = Batch_GRU(args)
        self.atom_fg_hyper_gru = Hyper_GRU(args)
        self.test = test
        self.W_o = nn.Linear(self.hidden_size*2,self.hidden_size)
    def init_feature(self,bg):
        bg.nodes['atom'].data['n'] = self.act(self.w_atom(bg.nodes['atom'].data['n']))# atom level node Hidden feature
        bg.edges[('atom','bond','atom')].data['e'] = self.act(self.w_bond(bg.edges[('atom','bond','atom')].data['e']))# atom level edge Hidden feature
        
        bg.nodes['pharm'].data['n'] = self.act(self.w_pharm(bg.nodes['pharm'].data['n']))# pharm level node Hidden feature
        
        bg.nodes['hyper_edge'].data['n'] = self.act(self.w_hypergraph(bg.nodes['hyper_edge'].data['n'])) # junction level for atom node Hidden feature [atom_dim+pharm_dim]
        bg.edges[('hyper_edge','react','hyper_edge')].data['e'] = self.act(self.hyper_reac(bg.edges[('hyper_edge','react','hyper_edge')].data['e']))# junction level for pharm node Hidden feature
    def reverse_edge(self,tensor):
        n = tensor.size(0)
        assert n%2 ==0
        delta = torch.ones(n).type(torch.long)
        delta[torch.arange(1,n,2)] = -1
        return tensor[delta+torch.tensor(range(n))]
    def index_select_ND(self,source:torch.Tensor,index:torch.Tensor):
        index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
        suffix_dim = source.size()[1:]  # (hidden_size,)
        final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
        target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
        target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
        
        target[index==0] = 0
        return target
    def atom_message_pass(self,node,field):
        agg_messge = node.mailbox['m'] #传递的消息
        return {field: node.data['n'] + (agg_messge.sum(dim=1) * agg_messge.max(dim=1)[0])}
    
    def update_bond_message(self,edge,layer,init_bond_feat):
        return {'e':self.dropout_layer(self.act(init_bond_feat+layer(edge.src['n'] - edge.data['rev_h'])))}
    
    def update_node(self,node,field):
        agg_messge = node.mailbox['mail'] #传递的消息
        return {field: agg_messge.sum(dim=1) * agg_messge.max(dim=1)[0]}
    def update_other(self,node,field):
        agg_messge = node.mailbox['mail'] #传递的消息
        return {field: agg_messge.sum(dim=1) * agg_messge.max(dim=1)[0]}
    def forward(self,step, batch, features_batch):
        self.init_feature(batch)
        hyper_edge_feature = batch.nodes['hyper_edge'].data['n']
        hyper_react_feature = batch.edges[('hyper_edge','react','hyper_edge')].data['e']
        pharm_feature = batch.nodes['pharm'].data['n'] # 
        bond_feat = batch.edges[('atom','bond','atom')].data['e']
        atom_feat = batch.nodes['atom'].data['n']

        message_atom = atom_feat.clone()
        message_bond = bond_feat.clone()
        message_react = hyper_react_feature.clone()
        message_fg = pharm_feature.clone()
        message_hyper = hyper_edge_feature.clone()


        hyper_graph_func = {}
        hyper_graph_func.update({('pharm','in','hyper_edge'):(fn.copy_u('n','hyper_m'),partial(add_attn,attn = self.hyper_att,field='n',message_filed = 'hyper_m'))})
        hyper_graph_func.update({('hyper_edge','react','hyper_edge'):(fn.copy_e('e','hyper_reac_m'),partial(add_attn,attn = self.hyper_reac_hyper,field='n',message_filed = 'hyper_reac_m'))})
        
        fg_func={('atom','junc','pharm'):(fn.copy_u('n','fg_m'),partial(add_attn,attn = self.fg_att,field='n',message_filed = 'fg_m'))}
        # fg_func.update({('hyper_edge','con','pharm'):(fn.copy_u('n','hyper2fg_m'),partial(add_attn,attn = self.hyper2fg_att,field='n',message_filed = 'hyper2fg_m'))})
        # Message passing
        for depth in range(self.depth - 1):
            if 'atom' in self.test:
                batch.update_all(fn.copy_e('e','m'),partial(self.atom_message_pass,field='n'),etype=('atom','bond','atom'))
                batch.edges[('atom','bond','atom')].data['rev_h']=self.reverse_edge(batch.edges[('atom','bond','atom')].data['e'])
                batch.apply_edges(partial(self.update_bond_message,layer=self._modules[f'W_h_atom_{depth}'],init_bond_feat = message_bond),etype=('atom','bond','atom'))
            
            # 更新片段特征
            if 'fg' in self.test:
                #batch.update_all(fn.copy_u('n','fg_m'),partial(add_attn,attn = self.fg_att,field='n',message_filed = 'fg_m'),etype=('atom','junc','pharm'))
                batch.multi_update_all(fg_func,cross_reducer = 'sum')

            if 'hyper' in self.test:# 二级片段，只能发生一次反应
                # 更新超边特征
                batch.multi_update_all(hyper_graph_func,cross_reducer = 'sum')
                # batch.update_all(fn.copy_u('n','hyper_m'),partial(add_attn,attn = self.hyper_att,field='n',message_filed = 'hyper_m'),etype=('pharm','in','hyper_edge'))
                # batch.update_all(fn.copy_e('e','hyper_reac_m'),partial(add_attn,attn = self.hyper_reac_hyper,field='n',message_filed = 'hyper_reac_m'),etype=('hyper_edge','react','hyper_edge'))
                # 反向更新，一个超边包含多个片段，一个片段属于多个超边
                # batch.update_all(fn.copy_u('n','hyper2fg_m'),partial(add_attn,attn = self.hyper2fg_att,field='n',message_filed = 'hyper2fg_m'),etype=('hyper_edge','con','pharm'))
                batch.edges[('hyper_edge','react','hyper_edge')].data['rev_h']=self.reverse_edge(batch.edges[('hyper_edge','react','hyper_edge')].data['e'])
                batch.apply_edges(partial(self.update_bond_message,layer=self._modules[f'W_h_hyper_{depth}'],init_bond_feat = message_react),etype=('hyper_edge','react','hyper_edge'))
        if 'atom' in self.test:
            # last update of node feature
            batch.update_all(fn.copy_e('e','mail'),partial(self.update_node,field='messge'),etype=('atom','bond','atom'))
            atom_message = self.lr(torch.cat([batch.nodes['atom'].data['messge'],batch.nodes['atom'].data['n'],message_atom],dim=1))
        
        # last update of fg feature
        if 'fg' in self.test:
            fg_update_func = {('atom','junc','pharm'):(fn.copy_u('n','mail'),partial(self.update_other,field='messge'))}
            #fg_update_func.update({('hyper_edge','con','pharm'):(fn.copy_u('n','mail'),partial(self.update_other,field='messge'))})
            batch.multi_update_all(fg_update_func,cross_reducer='sum')
            #batch.update_all(fn.copy_u('n','mail'),partial(self.update_other,field='messge'),etype=('atom','junc','pharm'))
            fg_message = self.fg_lr(torch.cat([batch.nodes['pharm'].data['messge'],batch.nodes['pharm'].data['n'],message_fg],dim=1))

        if 'hyper' in self.test and self.test!='atom_fg_nohyper':
            # hyper graph message
            hyper_update_func = {('hyper_edge','react','hyper_edge'):(fn.copy_e('e','mail'),partial(self.update_node,field='messge'))}
            hyper_update_func.update({('pharm','in','hyper_edge'):(fn.copy_u('n','mail'),partial(self.update_node,field='messge'))})
            batch.multi_update_all(hyper_update_func,cross_reducer = 'sum')
            hyper_graph_message = self.hyper_lr(torch.cat([batch.nodes['hyper_edge'].data['messge'],batch.nodes['hyper_edge'].data['n'],message_hyper],dim=1))
        
        # read out

        '''graph_emd1 = self.a2p_gru(batch,atom_message,fg_message,'atom','pharm')
        graph_emd2 = self.a2hyper_gru(batch,atom_message,hyper_graph_message,'atom','hyper_edge')
        graph_emb = self.dropout_layer(self.act(self.W_o(torch.cat([graph_emd1,graph_emd2],dim=1))))'''
        
        if self.test == 'atom':
            graph_emb = self.a2p_gru(batch,atom_message,None,'atom','pharm')
            graph_emb = self.dropout_layer(self.act(self.W_o(graph_emb)))
        elif self.test == 'atom_fg':
            graph_emb = self.a2p_gru(batch,atom_message,fg_message,'atom','pharm')
            graph_emb = self.dropout_layer(self.act(self.W_o(graph_emb)))
        elif self.test =='atom_fg_nohyper':# 这个是想测试通过超边来更新片段级别特征，但是不使用超边特征
            graph_emb = self.a2p_gru(batch,atom_message,fg_message,'atom','pharm')
            graph_emb = self.dropout_layer(self.act(self.W_o(graph_emb)))
        elif self.test == 'atom_hyper':
            graph_emb = self.a2p_gru(batch,atom_message,fg_message,'atom','pharm')
            graph_emb = self.dropout_layer(self.act(self.W_o(graph_emb)))
        elif self.test == 'atom_fg_hyper':# (1)pharm级别和hyper edge级别聚合（2）pharm级别和hyper edge级别聚合
            graph_emb = self.atom_fg_hyper_gru(batch,atom_message,fg_message,hyper_graph_message)
            graph_emb = self.dropout_layer(self.act(self.W_o(graph_emb)))
        return graph_emb

class Batch_GRU(nn.Module):
    def __init__(self,args:Namespace) -> None:
        super(Batch_GRU,self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.direction = 2
        self.att_mix = MultiHeadedAttention(args.gru_head,self.hidden_size)
        self.test=test
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
    def forward(self,batch,main_message,sub_message,main_field,sub_fielld):
        if self.test =='atom':
            h  = self.split_batch(batch,main_message,main_field,self.args.device)
        elif self.test == 'atom_fg' or self.test == 'atom_fg_nohyper':
            main_f = self.split_batch(batch,main_message,main_field,self.args.device)
            sub_f = self.split_batch(batch,sub_message,sub_fielld,self.args.device)
            mask = (main_f!=0).type(torch.float32).matmul((sub_f.transpose(-1,-2)!=0).type(torch.float32))==0
            h = self.att_mix(main_f, sub_f, sub_f,mask) + main_f
        elif self.test == 'atom_hyper':
            pass
        elif self.test == 'atom_fg_hyper':
            main_f = self.split_batch(batch,main_message,main_field,self.args.device)
            sub_f = self.split_batch(batch,sub_message,sub_fielld,self.args.device)
            mask = (main_f!=0).type(torch.float32).matmul((sub_f.transpose(-1,-2)!=0).type(torch.float32))==0
            h = self.att_mix(main_f, sub_f, sub_f,mask) + main_f
            
        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = batch.batch_num_nodes(main_field)
        start_index = torch.cat([torch.tensor([0],device=self.args.device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(batch.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hidden_size).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed

class Hyper_GRU(nn.Module):
    def __init__(self,args:Namespace) -> None:
        super(Hyper_GRU,self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.direction = 2
        self.p2hyper_att = MultiHeadedAttention(args.gru_head,self.hidden_size)
        self.a2p_att = MultiHeadedAttention(args.gru_head,self.hidden_size)
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

    def forward(self,batch,atom_message,pharm_message,hyper_edge_message):
        atom_f = self.split_batch(batch,atom_message,'atom',self.args.device)
        pharm_f = self.split_batch(batch,pharm_message,'pharm',self.args.device)
        hyper_edge_f = self.split_batch(batch,hyper_edge_message,'hyper_edge',self.args.device)
        mask1 = (pharm_f!=0).type(torch.float32).matmul((hyper_edge_f.transpose(-1,-2)!=0).type(torch.float32))==0
        mask2 = (atom_f!=0).type(torch.float32).matmul((pharm_f.transpose(-1,-2)!=0).type(torch.float32))==0
        pharm_f = self.p2hyper_att(pharm_f,hyper_edge_f,hyper_edge_f,mask1)+pharm_f
        h = self.a2p_att(atom_f,pharm_f,pharm_f,mask2)+atom_f
        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = batch.batch_num_nodes('atom')
        start_index = torch.cat([torch.tensor([0],device=self.args.device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(batch.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hidden_size).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed
class HyperGraph(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 pharm_fdim: int = None,
                 react_fdim: int = None,
                 graph_input: bool = False):
        super(HyperGraph, self).__init__()
        
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim()
        self.pharm_fdim = pharm_fdim or get_pharm_fdim()
        self.react_fdim = react_fdim or get_react_fdim()
        self.graph_input = graph_input
        self.args = args
        self.encoder = HyperGraphEncoder(self.atom_fdim,self.bond_fdim,
                                         self.pharm_fdim,self.react_fdim,self.args)

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = create_dglHyper_batch(batch, self.args, pretrain).to(self.args.device)
        output = self.encoder.forward(step, batch, features_batch)
        return output