# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/4/3 14:58
from torch import nn
import torch
from Layers import DynamicLSTM,GraphConvolution,Attention  #LSTM
import torch.nn.functional as F
from transformers import RobertaModel #SKSMT模型所需
class SKSMT(nn.Module):

    def __init__(self,opt,vocabulary_size):
        super(SKSMT, self).__init__()
        self.opt=opt
        self.roberta=RobertaModel.from_pretrained("./Pre-trained/Roberta")  #加载roberta模型
        self.roberta.resize_token_embeddings(vocabulary_size) #因为roberta的词表是扩充过的，所以embedding的形状也要进行扩充
        self.text_lstm=DynamicLSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True,bidirectional=True) #加载LSTM模型
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim) #768*2 加载GCN
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.atten=Attention(2*opt.hidden_dim) #注意力机制
        self.mlp=nn.Linear(2*opt.hidden_dim,256)  #降维处理
        self.fc = nn.Linear(256, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.1)
        pass
    def forward(self,inputs):
        '''
        :param inputs:  输入的数据
        :return:
        '''
        text_indices,attention_mask,depenency_adj,sentiment_adj=inputs #获取输入

        text_len = torch.sum(text_indices != 1, dim=-1) # 在最后一个维度(列axis)，计算不等于的个数。也就是在计算每个句子的实际长度外加两个special token <s>
        text_embedding=self.roberta(input_ids=text_indices,attention_mask=attention_mask)  #(batch,sen_length,embedding_dim) (?,?,768)
        text_embedding=self.text_embed_dropout(text_embedding.last_hidden_state)  #Dropout (16,128,768)
        #(batch,max(batch_sentence),2*768)
        text_out,(_,_)=self.text_lstm(text_embedding,text_len)

        x = F.relu(self.gc1(text_out, depenency_adj))
        x = F.relu(self.gc1(x, depenency_adj))  #情感图暂时不要了
        x = F.relu(self.gc1(x, depenency_adj))
        x = F.relu(self.gc1(x, depenency_adj))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2)) #(这个batch最长的句子的长度，这个batch最长的句子的长度)
        #这个语句就相当于每一个句子压缩成一个形状为(1,?)的向量
        alpha=alpha_mat.sum(1, keepdim=True) #（batch,1,batch最长的句子的长度),alpha_mat是(batch,?,?)，三个维度，所以是在行方向执行相加。也就是说一个句子的每个单词的词向量都相加起来了
        alpha = F.softmax(alpha, dim=2) #这个语句在列方向上进行归一化。也就是说这样可以类似得到一个attention的得分。
        x = torch.matmul(alpha, text_out).squeeze(1) #最后用这个得分和text_out相乘。  (16,768*2)


        x=self.mlp(x) # (16,256)
        #这里应该降维


        output = self.fc(x) #(16,2)

        return output