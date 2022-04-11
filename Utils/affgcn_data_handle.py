# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2022/4/3 15:14
import logging
import os
import pickle
import spacy
import numpy as np
import pandas as pd
from Utils.dependency_graph import DependencyGraph
from Utils.sentiment_graph import SentimentGraph
from tqdm import tqdm
#spacy
sp_nlp=spacy.load('en_core_web_sm')  #读取spacy的流程包，第一次使用spacy先下载这个流程包


class AFFGCNDatesetReader:
    def __init__(self,dataset='Semeval', embed_dim=300,max_seq_len = -1):
        logging.info("preparing {0} dataset ...".format(dataset))
        print("preparing {0} dataset ...".format(dataset))
        fname = {  # 数据所在
            'Semeval': {
                'train_data_label': './Data/Semeval/olid-training-v1.0.tsv',
                'test_data': './Data/Semeval/testset-levela.tsv',
                'test_label': './Data/Semeval/labels-levela.csv'
            },
            'SOLID':{
                'train_data': './Data/SOLID/task_a_distant.tsv'
            },
            'Davision':{
                'train_data': './Data/SOLID/task_a_distant.tsv'
            }
        }
        #令牌化器
        if dataset not in fname:  #如果所要数据集没有的话，发生异常
            raise ValueError("数据库里面没有你要的数据集，必须为Semeval/SOLID/Davision")

        dataset_path = fname[dataset]
        if  dataset.__eq__('Semeval'):
            self.semeval=AFFGCNSemevalReader(dataset=dataset, dataset_path=dataset_path, embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif dataset.__eq__('SOLID'):
            pass
        else:
            pass


class AFFGCNSemevalReader:
    def __init__(self,dataset,dataset_path,embed_dim=300,max_seq_len = -1):
        '''
        :param dataset:
        :param dataset_path: 所需的数据路径
        :param embed_dim: 词向量嵌入维度
        :param max_seq_len: 句子最大长度
        '''
        #返回训练集数据和测试集数据
        train_data,test_data=AFFGCNSemevalReader.__read_and_clean_text__([dataset_path['train_data_label'], dataset_path['test_data']])
        train_label,test_label=AFFGCNSemevalReader.__read_label([dataset_path['train_data_label'], dataset_path['test_label']])
        #得到数据之后，先建立一下两种图
        if os.path.exists('./Output/'+dataset+'_train_dependency.graph.new'):
            print("loading {0} dependency grpah...".format(dataset))
            fin = open('./Output/'+dataset+ '_train_dependency.graph.new', 'rb')  # 读取依赖图
            train_dependency_graph = pickle.load(fin)
            fin.close()
            fin = open('./Output/'+dataset+ '_test_dependency.graph.new', 'rb')  # 读取依赖图
            test_dependency_graph = pickle.load(fin)
            fin.close()
        else: #构建图,训练集没有说明测试集也没有
            train_dependency_graph=DependencyGraph(train_data,dataset+'_train_dependency')
            train_dependency_graph.process()
            train_dependency_graph=train_dependency_graph.idx2graph
            test_dependency_graph=DependencyGraph(test_data,dataset+'_test_dependency')
            test_dependency_graph.process()
            test_dependency_graph=test_dependency_graph.idx2graph

        if os.path.exists('./Output/'+dataset+'_train_sentiment.graph.new'):
            print("loading {0} sentiment grpah...".format(dataset))
            fin = open('./Output/'+dataset+ '_train_sentiment.graph.new', 'rb')  # 读取情感图
            train_sentiment_graph = pickle.load(fin)
            fin.close()
            fin = open('./Output/'+dataset+ '_test_sentiment.graph.new', 'rb')  # 读取情感图
            test_sentiment_graph = pickle.load(fin)
            fin.close()
        else: #构建图,训练集没有说明测试集也没有
            train_sentiment_graph=SentimentGraph(train_data,dataset+'_train_sentiment')
            train_sentiment_graph.process()
            train_sentiment_graph=train_sentiment_graph.idx2graph
            test_sentiment_graph=SentimentGraph(test_data,dataset+'_test_sentiment')
            test_sentiment_graph.process()
            test_sentiment_graph=test_sentiment_graph.idx2graph

 #############################两个图构建完毕###################################################


        if os.path.exists('./Output/'+dataset+'_word2idx.pkl'):  #如果词序列已经存在了
            print("loading {0} tokenizer...".format(dataset))
            with open('./Output/'+dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:#不存在，就分词，将词映射成序列
            tokenizer = Tokenizer() #Tokenizer是自己写的
            tokenizer.fit_on_text(train_data)
            with open('./Output/'+dataset+'_word2idx.pkl', 'wb') as f:  #存储文件 word2idx
                 pickle.dump(tokenizer.word2idx, f)


        #构造词向量矩阵
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)


        #包装好数据
        self.train_data=Dataset(AFFGCNSemevalReader.__read_data__(train_data, train_label, train_dependency_graph
                                                                  , train_sentiment_graph, tokenizer, max_seq_len))
        self.test_data=Dataset(AFFGCNSemevalReader.__read_data__(test_data, test_label, test_dependency_graph
                                                                 , test_sentiment_graph, tokenizer, max_seq_len))



    @staticmethod
    def __read_and_clean_text__(fnames):
        '''
            对文本进行清理，然后返回文本
           @:fnames  是一个列表包含训练集和测试集数据的路径
           @:return 返回训练集文本和测试集文本
        '''
        train_data=pd.read_csv(fnames[0],sep='\t')
        test_data=pd.read_csv(fnames[1],sep='\t')
        #小写化
        train_data = train_data.tweet.str.lower() # 小写化
        test_data = test_data.tweet.str.lower()  # 测试集也小写化
        #表情文本化
        train_data=EmojiToText(train_data)
        test_data=EmojiToText(test_data)
        #文本清洗
        train_data=DataCleaned(train_data)
        test_data=DataCleaned((test_data))

        return train_data,test_data


    @staticmethod
    def __read_label(fnames):
        '''
        :param fnames: 应该是一个列表，里面是训练集和测试集label数据的path
        :return: 返回label数据
        '''

        train_data = pd.read_csv(fnames[0], sep='\t')
        test_data=pd.read_csv(fnames[1])

        # 将标注转变为0和1  1为OFF ，0为NOT
        y_train = [1 if i == "OFF" else 0 for i in train_data["subtask_a"]]
        y_test = [1 if i == "OFF" else 0 for i in test_data['label']]
        return y_train,y_test



    @staticmethod
    def __read_data__(data,labels,dependency_graphs,sentiment_graphs,tokenizer,max_seq_len=-1):
        '''
             构造了一个数据：all_data，里面含有原本的句子，离散句子，两个图，还有句子的标签
        '''
        all_data=[]
        for i in tqdm(range(len(data))):
            context=data[i]
            context_indices=tokenizer.text_to_sequence(context)
            dependency_graph = dependency_graphs[i]  # 句子对应的依赖图
            sentiment_graph=sentiment_graphs[i] #句子对应的情感图
            label=labels[i]


            # 判断句子长度和图的行数不一致或者句子长度和语义图的行数，说明句子和两个图没对应上，输出这些信息，触发异常。
            if len(context_indices) != dependency_graph.shape[0] or len(context_indices) != sentiment_graph.shape[0]:
                print(context)
                print(len(context_indices))
                print(dependency_graph.shape, sentiment_graph.shape)
                raise ValueError("图和文本对不上啦-JJ")


            if max_seq_len>0:
                #补充或者截断图
                if len(dependency_graph)>max_seq_len:#截断多出来的部分
                    dependency_graph=np.split(dependency_graph,[0,max_seq_len])[1]
                    sentiment_graph=np.split(sentiment_graph,[0,max_seq_len])[1]
                else:#反之，补充
                    dependency_graph = np.pad(dependency_graph,(0,max_seq_len-dependency_graph.shape[0]),'constant')  #
                    sentiment_graph = np.pad(sentiment_graph,(0,max_seq_len-sentiment_graph.shape[0]),'constant')



                if len(context_indices) < max_seq_len:  # 如果句子的长度小于max_seq_len，
                    context_indices = context_indices + [0] * (max_seq_len - len(context_indices))  # 补充
                else: #截断
                    context_indices = context_indices[:max_seq_len]  # 截断

            else:
                print("没有指定max_seq_len")
                logging.info("没有指定max_seq_len")
                raise ValueError("没有指定max_seq_len")

            data_pack={
                'context': context,  # 句子
                'context_indices': context_indices,  # 句子索引
                'dependency_graph': dependency_graph,  # 依赖图
                'label': label,
                'sentiment_graph': sentiment_graph,  # 情感图
            }

            all_data.append(data_pack)
        return  all_data



class SOLIDReader:
    def __init__(self,num):
        '''
        这个数据集使用来辅助学习的，所以需要控制读取数据的数量。
        :param num:
        '''
        pass

'''
分词类
'''
class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx # <pad>词典序为0
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<UNK>'] = self.idx  #  <UNK> ，词典序为1
            self.idx2word[self.idx] = '<UNK>'
            self.idx += 1
        else: #如果word2idx已经存在，直接读取就行
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self,text):
        '''
        这个函数给 word2idx  {词:index} ， idx2word  {index:词}赋值
        建立词典
        '''
        vocabulary=[]
        for x in text:
            x = x.lower().strip() #将这一行的数据小写化，并且去掉前后空格
            tweet_tokens = sp_nlp(x)  # #spacy
            vocabulary = vocabulary + [str(y) for y in tweet_tokens]  #制造词典
        for word in vocabulary:
            if word not in self.word2idx: #word2idx  {词:index}
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word  #idx2word  {index:词}
                self.idx += 1

    def text_to_sequence(self, text):
        '''
        将词转换为词典对应的index
        '''
        words = sp_nlp(text)
        words = [str(x) for x in words]

        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def EmojiToText(texts):
    '''
    Description:将texts含有的表情文本化
    Params:
        texts:数据文本
    '''
    import emoji
    filtered_tweets=[]
    for tweet in texts:
        filtered_sentences=emoji.demojize(tweet)
        # print(filtered_sentences)
        filtered_tweets.append(filtered_sentences)

    return  filtered_tweets



def  DataCleaned(texts):
    '''
    文本清洗
    '''
    #from nltk import word_tokenize #分词  deprected
    #nltk.download('punkt')  第一次运行需要下载
    # For train data
    filtered_tweets = []
    for tweet in texts:
        tweet_tokens = sp_nlp(tweet)  # 分词
        filtered_sentence=''
        # 删除一些句子没有信息量的东西
        for w in tweet_tokens:
            w=str(w.lemma_) #词根还原
           # w = str(w) #不进行词性还原了
            if (w != 'url' and w != '@user' and w !='user' and w != '@' and w != ','  and w != "’" and w != '.' and w != '#' and w !=':'
                    and w != ';' and w !='&' and w !="''" and w != '-' and w !="@us"):
                filtered_sentence=filtered_sentence+' '+w
        filtered_sentence= filtered_sentence.strip() #删除前后空格
        filtered_tweets.append(filtered_sentence)


    return filtered_tweets


def build_embedding_matrix(word2idx, embed_dim, type):
    '''
    建立预训练的embedding_matrix，预训练的模型是glove。
    Deprecated：已经改用了RoBERTa
    '''
    #embedding_matrix_file 文件名
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)

    if os.path.exists("./Output/"+embedding_matrix_file_name):  #如果存在直接加载就好了
        print('loading embedding_matrix:', embedding_matrix_file_name)
        logging.info('loading embedding_matrix:'+embedding_matrix_file_name)
        embedding_matrix = pickle.load(open("./Output/"+embedding_matrix_file_name, 'rb'))
    else:  #不存在就建立
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros 构建了一个零向量矩阵
        # 给unk随机初始化(赋予意义)
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim),(1, embed_dim))
        if embed_dim == 100:
            fname = "./Pre-trained/glove/glove.6B.100d.txt"  #embding 100
        else:
            fname = './Pre-trained/glove/glove.42B.300d.txt' #embedding 300

        #得到词向量(词典和glove预训练共同拥有的词)
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        logging.info('building embedding_matrix:'+embedding_matrix_file_name)

        #把从glove预训练中得到的词向量赋给embedding_matrix，在glove没找到的(还是原样)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open("./Output/"+embedding_matrix_file_name, 'wb')) #存储预训练静态词向量矩阵
        logging.info('done !!!' + embedding_matrix_file_name)
    return embedding_matrix


def load_word_vec(path, word2idx=None):
    '''
     根据word2idx里面存储的词，找到glove预训练对应的词向量。最后返回一个含有词及其对应的向量的字典。
     Note:需要注意的是，我们的词在glove预训练中不一定有
    '''
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec