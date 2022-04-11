# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/3/16 15:55
import logging

import numpy as np
import spacy
import pickle
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

class SentimentGraph(object):
    '''
    一个平平无奇的构造情感图类，关键在于sentiwordnet是否有我们数据集的词，致命的是，这个sentiwordnet 基本没有我们想要的词
    '''
    def __init__(self,data,graph_name):
        self.data=data
        self.graph_name=graph_name
        self.idx2graph={}

    def process(self):
        #加载情感得分
        sentimentNet=load_sentic_word()
        with open("./Output/"+self.graph_name+'.graph.new','wb') as fout:
            for i in tqdm(range(0,len(self.data))):
                #文本都是已经处理过的，这里不在处理
                text=self.data[i]
                adj_matrix = self.__sentiment_adj_matrix(text,sentimentNet)  # 构建依赖图矩阵
                self.idx2graph[i] = adj_matrix
            pickle.dump(self.idx2graph, fout)  # 存储依赖图矩阵
        print('done !!!', self.graph_name)
        logging.info('done !!!'+self.graph_name)
        fout.close()

        #构建情感图矩阵
    def __sentiment_adj_matrix(self,text,sentimentNet):
        word_list = nlp(text)
        seq_len = len(word_list)
        matrix = np.zeros((seq_len, seq_len)).astype('float32') #这个点后面还需要处理一下。过大截断，过小补充

        for i in range(seq_len):
            for j in range(i,seq_len):
                word_i = str(word_list[i])
                word_j = str(word_list[j])
                #这一句语句出问题了
                if word_i not in sentimentNet or word_j not in sentimentNet or word_i == word_j:  #如果我们的词在senticNet中找不带，跳出这一轮循环
                    continue
                sentic = abs(float(sentimentNet[word_i] - sentimentNet[word_j]))  #两个词的情感得分相减取绝对值
                matrix[i][j] = sentic
                matrix[j][i] = sentic

        return matrix


'''
    load senticNet
'''
def load_sentic_word():
    path = './sentimentNet/sentiwordnet.txt'
    sentimentNet={}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        word=nlp(word)
        for w in word:
            w=str(w.lemma_)
        sentimentNet[w] = float(sentic)
    fp.close()
    return sentimentNet