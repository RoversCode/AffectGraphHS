# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/14 21:12
from torch import nn
import torch
from Layers import DynamicLSTM,GraphConvolution,Attention  #LSTM
import torch.nn.functional as F
#基于多任务的情感知识共享 Sentiment knowledge sharing based on multi-task SKSMT

class AFFGCN(nn.Module):
    '''
    Depracted
     模型类AFFGCN，情感图加上LSTM以及自定义的Attention
    '''
    def __init__(self,embedding_matrix,opt):
        super(AFFGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.embed.requires_grad_(True)
        self.text_lstm=DynamicLSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim) #600
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.atten=Attention(2*opt.hidden_dim) #注意力机制
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self,inputs):
        '''
        :param inputs:  输入的数据
        :return:
        '''
        #adj依赖图，text——indices句子索引，sentic_adj情感图
        text_indices, dependent_adj , sentic_adj= inputs

        text_len = torch.sum(text_indices != 0, dim=-1) # 在最后一个维度(列axis)，计算不等于的个数。也就是在计算每个句子的实际长度

        text = self.embed(text_indices)  #得到text_indices对应的词向量 16*128*300
       # text = self.text_embed_dropout(text)  #Droupt    这个策略先不用
        text_out, (_, _) = self.text_lstm(text, text_len) # lstm，这里意思是舍弃掉了LSTM产生的每个句子的memory c和hidden state  h的信息，只要输出信息
        #text_out (batch,?,600)

        x = F.relu(self.gc1(text_out, dependent_adj)) #text_out(batch,这个batch最长的句子长度,2*lstm的hidden state)
        # x = F.relu(self.gc2(x, sentic_adj))
        # x = F.relu(self.gc3(x, dependent_adj))
        # x = F.relu(self.gc4(x, sentic_adj))
        # x = F.relu(self.gc5(x, dependent_adj))
        # x = F.relu(self.gc6(x, sentic_adj)) #(batch,?,600)
        x = F.relu(self.gc2(x, dependent_adj))
        x = F.relu(self.gc3(x, dependent_adj))
        x = F.relu(self.gc4(x, dependent_adj))

        # alpha_mat,_=self.atten(text_out,x) #(batch,?,600)
        # alpha=alpha_mat.sum(1, keepdim=True) #(batch,1,600)
        # alpha=alpha.squeeze(1) #(batch,1,600)
        # output=self.fc(alpha)



        alpha_mat = torch.matmul(x, text_out.transpose(1, 2)) #(这个batch最长的句子的长度，这个batch最长的句子的长度)
        #这个语句就相当于每一个句子压缩成一个形状为(1,?)的向量
        alpha=alpha_mat.sum(1, keepdim=True) #（batch,1,batch最长的句子的长度),alpha_mat是(batch,?,?)，三个维度，所以是在行方向执行相加。也就是说一个句子的每个单词的词向量都相加起来了
        alpha = F.softmax(alpha, dim=2) #这个语句在列方向上进行归一化。也就是说这样可以类似得到一个attention的得分。
        x = torch.matmul(alpha, text_out).squeeze(1) #最后用这个得分和text_out相乘。  (16,600)

        output = self.fc(x) #(16,2)

        return output


