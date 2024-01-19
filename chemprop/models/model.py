from argparse import Namespace

from .augmented_encoder import PharmHGT
from .cmpn import CMPN
import torchvision
from chemprop.nn_utils import get_activation_function, initialize_weights,index_select_ND
import pdb
from functools import partial
import logging
from mimetypes import init
from turtle import forward, hideturtle, up
import torch
import torch.nn as nn
from typing import NamedTuple, Union, Callable
import torch.nn.functional as F
import math
import copy
import numpy as np
from dgl import function as fn
from chemprop.features.featurization import get_atom_fdim,get_bond_fdim,get_pharm_fdim

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, pretrain: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

    def create_encoder(self, args: Namespace, encoder_name):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        if encoder_name == 'CMPNN':
            self.encoder = CMPN(args)
        elif encoder_name == 'PharmHGT':
            self.encoder = PharmHGT(args)
    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        if not self.pretrain:
            output = self.ffn(self.encoder(*input))

            # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
            if self.classification and not self.training:
                output = self.sigmoid(output)
            if self.multiclass:
                output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
                if not self.training:
                    output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        else:
            output = self.ffn(self.encoder(*input))

        return output

def build_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', pretrain=args.pretrain)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model

def build_pretrain_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.ffn_hidden_size = args.hidden_size//2
    args.output_size = args.hidden_size

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', pretrain=True)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)
    
    initialize_weights(model)

    return model

def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(self.hidden_size, 32)
        self.w_k = nn.Linear(self.hidden_size, 32)
        self.w_v = nn.Linear(self.hidden_size, 32)
        self.args = args
        self.dense = nn.Linear(32, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self,fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)
        
        return hidden_states,attn

class Prompt_generator(nn.Module):
    def __init__(self, args:Namespace):
        super(Prompt_generator, self).__init__()
        self.brics_fdim = args.pharm_fdim
        self.react_fdim = args.react_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Activation
        self.act_func = get_activation_function(args.activation)
        # add frage attention
        self.fg = nn.Parameter(torch.randn(1,self.hidden_size*3), requires_grad=True)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        self.atten_layers = nn.ModuleList([AttentionLayer(args) for _ in range(args.num_attention)])
        self.linear = nn.Linear(self.hidden_size,self.hidden_size)
        self.lr = nn.Linear(self.hidden_size*3,self.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size)
    def forward(self, f_atom, atom_scope, f_group, group_scope, mapping, mapping_scope):
        max_frag_size = max([g_size for _,g_size in group_scope])+1 # 加上一行填充位置
        f_frag_list = []
        padding_zero = torch.zeros((1,self.hidden_size)).cuda()
        for i,(g_start, g_size) in enumerate(group_scope):
            a_start,a_size = atom_scope[i]
            m_start,m_size = mapping_scope[i]
            cur_a = f_atom.narrow(0,a_start,a_size)
            cur_g = f_group.narrow(0,g_start,g_size)
            cur_m = mapping.narrow(0,m_start,m_size) #  
            cur_a = torch.cat([padding_zero,cur_a],dim=0) #
            cur_a = cur_a[cur_m]
            cur_g = torch.cat([cur_a.sum(dim=1),cur_a.max(dim=1)[0],cur_g],dim=1)
            cur_brics = torch.cat([self.fg,cur_g],dim=0)
            cur_frage = torch.nn.ZeroPad2d((0,0,0,max_frag_size-cur_brics.shape[0]))(cur_brics)
            f_frag_list.append(cur_frage.unsqueeze(0))
        f_frag_list = torch.cat(f_frag_list, 0)
        f_frag_list = self.act_func(self.lr(f_frag_list))
        hidden_states,self_att = self.atten_layers[0](f_frag_list,f_frag_list)
        for k,att in enumerate(self.atten_layers[1:]):
            hidden_states,self_att = att(hidden_states,f_frag_list)
        f_out = self.linear(hidden_states)
        f_out = self.norm(f_out)* self.alpha
        return f_out[:,0,:],self_att

class PromptGeneratorOutput(nn.Module):
    def __init__(self,args,self_output):
        super(PromptGeneratorOutput, self).__init__()
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)
    def forward(self,hidden_states: torch.Tensor):
        hidden_states = self.self_out(hidden_states)
        return hidden_states

def prompt_generator_output(args):
    return lambda self_output : PromptGeneratorOutput(args,self_output)

def add_FUNC_prompt(model:nn.Module,args:Namespace = None):
    model.encoder.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.encoder.W_i_atom)
    return model