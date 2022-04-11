# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2021/11/18 11:45
# from Layers import Attention
#
# import  torch
#
# if __name__ == '__main__':
#     a=torch.randn((16,58,600))
#     b=torch.randn((16,58,600))
#     model=Attention(600)
#
#     out,score=model(a,b)
fname = {  # 数据所在
    'Semeval': {
        'train_data_label': './Data/Semeval/olid-training-v1.0.tsv',
        'test_data': './Data/Semeval/testset-levela.tsv',
        'test_label': './Data/Semeval/labels-levela.csv'
    },
    'SOLID': {
        'train_data': './Data/SOLID/task_a_distant.tsv'
    },
    'Davision': {
        'train_data': './Data/SOLID/task_a_distant.tsv'
    }
}

if "Semevssssal" not in fname:
    raise ValueError("数据库里面没有你要的数据集，必须为Semeval SOLID Davision")