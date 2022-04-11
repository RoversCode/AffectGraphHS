# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/19 14:07
import torch
from torch import nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    图卷积层
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight) # hidden(16，batch中最长的句子长度，2*hidden state)  (16,(这一个batch最长句子的长度),600)
        #改变一下adj的形状，以便运算
        adj=adj[:,:len(hidden[0]),:len(hidden[0])]
        denom = torch.sum(adj, dim=2, keepdim=True) + 1 #每个点的度计算
        output = torch.matmul(adj, hidden) / denom  #计算完后归一化
        if self.bias is not None: #是否加上偏置
            return output + self.bias
        else:
            return output