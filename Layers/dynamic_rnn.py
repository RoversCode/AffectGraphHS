# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
'''
LSTM层

'''
class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list 这个具体是作用是什么？
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)  #x_len是一个张量，形状(16,)，x_sort得到-x_len从小到大的索引(相当于得到x_len从大到小的索引)。
        x_unsort_idx = torch.argsort(x_sort_idx).long() #x_unsort_idx的值是x_sort_idx从小到大排序，数的索引
        x_len = x_len[x_sort_idx]  # 这个时候x_len得到自己原本从大到小的排序
        x = x[x_sort_idx.long()] #在batch维度上进行了调换，具体调换是由x_sort_idx决定的，这个操作会得到句子从长到短的Tensor(16,128,300)
        """pack，因为我们为了句子长度一致方便计算，所以会将句子进行填充。所以每个句子会有很多的无意义字符(对应0)
        这个pack_padded_sequence的操作就是，将句子继续压缩，取消掉了我们pad的部分，然后将所有句子的(senten_length,emdding_dim)按dim=0方向
        拼接在一块。所以这个x_emb_p里面的数据时(batch个句子的长度相加，embdding_dim)
        """
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.to('cpu'), batch_first=self.batch_first)

        if self.rnn_type == 'LSTM':
            if h0 is None: #ht包含了每一个句子最后的hidden state,ct就是每个句子最后的containing the final cell state for each element in the sequence.
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)  #x_emb_p=(batch个句子的长度相加，embdding_dim)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else: 
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[ #将句子的顺序调整为原来的顺序
            x_unsort_idx]  
        ht = torch.transpose(ht, 0, 1) #然后再调回来

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""#上面对句子进行了压缩，现在要解压缩回来。因为max_len的信息已经丢失，所以这里max_len更正为这一个batch最长的句子的长度。这个函数返回一个
            #tuple，(数据，每一个句子的长度信息)
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first) #解压缩
            out = out[0]  #取得数据
            out = out[x_unsort_idx] #返回原来的句子顺序
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[ #返回原来顺序
                    x_unsort_idx]
                ct = torch.transpose(ct, 0, 1) #在调回来

            return out, (ht, ct)
