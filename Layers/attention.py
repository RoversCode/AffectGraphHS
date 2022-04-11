# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/24 16:17
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention) 几个头
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head  #//整除
        if out_dim is None:
            out_dim = embed_dim   #默认输出维度等于输入进来的维度
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)  #k
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)  #q
        self.proj = nn.Linear(n_head * hidden_dim, out_dim) #v
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):  #k lstm q  gcn
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  #batch大小
        k_len = k.shape[1]  #sentence长度
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)  #得到k ht
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim) #将head的维度删掉
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)  #得到q  gcn
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1) #dim 2 和1 调换 (600,58)
            score = torch.bmm(qx, kt) #bmm可以执行批量矩阵相乘, 得到一个(batch,58,58)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=-1)  # (16,58,58)  在列->的方向，进行了softmax。也就是58个词，每一个词关注其他词的分数都有了
        output = torch.bmm(score, kx) # (16,58,600) ，将这个分数和kx相乘得到attention矩阵
        # c=torch.split(output, mb_size, dim=0)
        # output = torch.cat(c, dim=-1) # (16,58,600)
        output = self.proj(output)  #(16,58,600) #为什么最后还要进行一次全连接？这个全连接似乎没什么必要
        output = self.dropout(output)
        return output, score #返回score是为了后面方便做可视化
