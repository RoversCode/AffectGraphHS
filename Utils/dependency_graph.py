# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/14 21:17
import numpy as np
import spacy
import pickle
from tqdm import tqdm
import logging
nlp = spacy.load('en_core_web_sm')


class DependencyGraph(object):
    '''
    一个平平无奇的构造依赖图类c
    '''
    def __init__(self,data,graph_name):
        self.idx2graph={} #依赖图的装入
        self.data=data  #数据
        self.graph_name=graph_name
    def process(self):
        with open("./Output/"+self.graph_name+'.graph.new','wb') as fout:
            for i in tqdm(range(0,len(self.data),1)):
                #文本都是已经处理过的，这里不在处理
                text=self.data[i]
                adj_matrix = self.__dependency_adj_matrix(text)  # 构建依赖图矩阵
                self.idx2graph[i] = adj_matrix
            pickle.dump(self.idx2graph, fout)  # 存储依赖图矩阵
        logging.info('done !!!' + self.graph_name)
        fout.close()
    def __dependency_adj_matrix(self,text): #私有方法
        # https://spacy.io/docs/usage/processing-text
        document = nlp(text)
        seq_len = len(document)
        matrix = np.zeros((seq_len, seq_len)).astype('float32') #先构建一个空壳矩阵

        for token in document:
            matrix[token.i][token.i] = 1
            for child in token.children:  # 这个token,children是有依赖的词的索引
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
        return matrix

