# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/18 15:54
import math
import random
import torch

class SKSMTBatchIteraotr(object):
    """
    SKSMT模型的Batch类
    """
    def __init__(self, data, batch_size, sort_key='context_indices', shuffle=True, sort=False):
        '''
        :param data:
        :param batch_size:
        :param sort_key:
        :param shuffle:
        :param sort:
        '''
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)  #装着batch的数据
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        '''
        :param data: 数据
        :param batch_size: batch大小
        :return:
        '''
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort: #????
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data

        batches=[]
        for i in range(num_batch):
            batches.append(self.get_batch_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return  batches

    def get_batch_data(self,batch_data):
        batch_context = []
        batch_context_indices = []
        batch_attention_mask=[]
        batch_dependency_graph = []
        batch_sentic_graph = []
        batch_label = []
        for item in batch_data:
            context,  context_indices,  dependency_graphs, label , sentiment_graphs = \
                item['context'], \
                item['context_indices'],\
                item['dependency_graph'],\
                item['label'],\
                item['sentiment_graph']

            batch_context.append(context)
            batch_context_indices.append(context_indices['input_ids'])
            batch_attention_mask.append(context_indices['attention_mask'])
            batch_dependency_graph.append(dependency_graphs)
            batch_sentic_graph.append(sentiment_graphs)
            batch_label.append(label)

        return {
            'context': batch_context, \
            'context_indices': torch.tensor(batch_context_indices), \
            'attention_mask': torch.tensor(batch_attention_mask),\
            'dependency_graph': torch.tensor(batch_dependency_graph), \
            'sentiment_graph': torch.tensor(batch_sentic_graph), \
            'label': torch.tensor(batch_label),
        }
    def __iter__(self): #??
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
