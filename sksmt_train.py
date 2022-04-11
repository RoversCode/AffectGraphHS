# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2021/11/17 11:00
import argparse
import os
import logging
import torch
from torch import nn
from Models import SKSMT,AFFGCN #模型
from Utils import SKSMTDatesetReader,SKSMTBatchIteraotr
from torchsummary import summary #输出模型详细信息
import math
from sklearn import metrics
logger = logging.getLogger(__name__)

class Instructor:
    def __init__(self,opt):
        self.opt=opt #配置
        dataset=SKSMTDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len = opt.max_seq_len)
        self.train_data_loader = SKSMTBatchIteraotr(data=dataset.semeval.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = SKSMTBatchIteraotr(data=dataset.semeval.test_data, batch_size=opt.batch_size, shuffle=False)
        self.model = opt.model_class(opt,dataset.semeval.vocab_size).to(opt.device)
       # summary(self.model,(16,128,300))
        self._print_args()
        self.global_f1 = 0.  #全局的f1

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        '''
        输出一些训练参数的信息
        :return:
        '''
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params  #需要训练的参数
            else:
                n_nontrainable_params += n_params   #不参与训练的参数
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def run(self,repeats=3):
        '''
        :param repeats:
        :return:
        '''
        loss_function=nn.CrossEntropyLoss() #损失函数
        _params=filter(lambda  p: p.requires_grad, self.model.parameters()) #将需要训练的参数过滤出来
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)#优化器
        if not os.path.exists('log/'):
            os.mkdir('log/')
        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'a+', encoding='utf-8')
        f_out.write("\n"+"#"*100)
        test_acc=[]
        test_f1=[]
        test_pre = []
        test_recall = []
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):  #重复训练repeats次。
            print('repeat: ', (i+1))
            f_out.write('\nrepeat: '+str(i+1))
            self._reset_params()  #给参数初始化
            max_test_acc, max_test_f1 ,max_test_precision , max_test_recall= self._train(loss_function, optimizer) #训练
            test_acc.append(max_test_acc)
            test_f1.append(max_test_f1)
            test_pre.append(max_test_precision)
            test_recall.append(max_test_recall)
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            f_out.write('\n\tmax_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg/repeats)
        print("max_test_f1_avg:", max_test_f1_avg/repeats)
        f_out.write('\n\tmax_test_acc_avg: {0}, max_test_f1_avg: {1}'.format(max_test_acc_avg/repeats, max_test_f1_avg/repeats))
        for i,j,k,p in zip(test_acc,test_f1,test_pre,test_recall):
            print('max_test_acc: {0}     max_test_f1: {1}   max_test_pre: {2}  max_test_recall: {3}'.format(i, j, k, p))
        print(self.opt.log_info)
        f_out.close()

    def _train(self,loss_funciton,optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        max_test_precision = 0
        max_test_recall = 0
        global_step = 0
        continue_not_increase = 0    #判断早停的依据
        for epoch in range(self.opt.num_epoch): #epoch
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader): #batch级别
                global_step +=1
                self.model.train() #进入训练状态
                optimizer.zero_grad() #清除梯度
                inputs = [sample_batched[col].to(self.opt.device) if col != 'context'  else sample_batched[col] for col in self.opt.inputs_cols]
                targets = sample_batched['label'].to(self.opt.device)  #label

                outputs = self.model(inputs)
                loss=loss_funciton(outputs,targets)
                loss.backward()  #计算梯度
                optimizer.step()  #更新梯度

                if global_step % self.opt.log_step==0:  #log_step个batch训练后，测试一次性能
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc=n_correct/n_total #训练集的准确率
                    test_acc, test_f1, test_precision, test_recall = self._evaluate_acc_f1() #测试

                    if test_f1 > max_test_f1:  #f1是核心指标，如果有提升，就进来
                        max_test_acc=test_acc
                        max_test_recall=test_recall
                        max_test_precision=test_precision
                        increase_flag=True  #性能有提升标志置为真
                        max_test_f1=test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1=test_f1
                            #在这保存模型？
                    #输出训练信息已经达到的最好的性能。
                    print("max test f1: ",max_test_f1,"max test acc: ",max_test_acc,"max test pre: ",max_test_precision,"max test recall: ",max_test_recall)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))

            if increase_flag ==False: #早停机制，如果在一个epoch里面，性能没有提升过。
                continue_not_increase +=1
                if continue_not_increase >= self.opt.estop:  #忍耐度默认为4
                    print('early stop.')
                    break
            else:
                continue_not_increase=0 #如果有提升，也就是increase_flag为真，计数重置
        return max_test_acc, max_test_f1 , max_test_precision , max_test_recall


    def _evaluate_acc_f1(self):
        '''
        模型性能性能评估
        :return:
        '''
        self.model.eval()   #switch model to evaluation model
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) if col != 'context' else t_sample_batched[col] for col
                            in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                              average='macro')
        precision_macro = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                                                  labels=[0, 1], average='macro')
        recall_macro = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                                            average='macro')
        return test_acc, f1, precision_macro, recall_macro

    def _reset_params(self):
        '''
        给训练参数初始化
        :return:
        '''
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)   #参数初始化，初始化方法由自己指定
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def main():
    parser=argparse.ArgumentParser()
    #需要的参数
    parser.add_argument("--dataset",default="Semeval",help="数据集Semeval ")
    parser.add_argument("--sentiment", default="SOLID", help="辅助任务的数据集名字 ")
    parser.add_argument("--output_dir",default="./Output/",type=str,help="输出目录")
    parser.add_argument("--pretrain_dir",default="./Pre-trained/",help="预训练模型所在的目录")
    parser.add_argument('--optimizer', default='adam', type=str,help="优化器")
    parser.add_argument("--max_seq_len",default=128,type=int,help="输入句子长度，默认128。")
    parser.add_argument("--do_train",action='store_true',help="决定是否要训练")
    parser.add_argument("--do_eval",action='store_true',help="决定是否要进行测试")
    parser.add_argument("--learning_rate",default=5e-5,type=float,help="学习率")
    parser.add_argument("--num_train_epochs",default=3.0,type=float,help="训练次数")
    parser.add_argument('--device', default="cuda:0", type=str,help="训练设备")
    parser.add_argument('--model_name', default='sksmt', type=str,help="要训练的模型名称") ###
    parser.add_argument('--embed_dim', default=768, type=int,help="词向量维度")
    parser.add_argument('--batch_size', default=16, type=int,help="batch大小")
    parser.add_argument('--hidden_dim', default=768, type=int, help="LSTM的Hidden维度")
    parser.add_argument('--polarities_dim', default=2, type=int,help="分类极性")
    parser.add_argument('--repeat', default=3, type=int,help="训练几次")
    parser.add_argument('--l2reg', default=0.00001, type=float,help="正则化")
    parser.add_argument('--initializer', default='xavier_uniform_', type=str,help="训练参数初始化")
    parser.add_argument('--num_epoch', default=100, type=int,help="迭代次数")
    parser.add_argument('--log_step', default=5, type=int,help="几个batch更新一次")
    parser.add_argument('--save', default=True, type=bool,help="是否保存最后的全局性能")
    parser.add_argument('--estop', default=4, type=int,help="早停机制,4次性能没有进步就停止训练")
    parser.add_argument('--log_info', default="运行完毕", type=str)

    opt=parser.parse_args();

    '''
    为了方便Debug，配置数值会在这里改
    '''


    #####################################################测试结尾
    optimizers = {#优化器
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    model_classes = { #模型种类
        'affgcn': AFFGCN,
        'sksmt': SKSMT,
    }
    #数据输入形式
    input_colses = {
        'affgcn': ['context_indices',  'dependency_graph', 'sentiment_graph'],
        'sksmt': ['context_indices','attention_mask','dependency_graph','sentiment_graph'],
    }

    #输入参数初始化
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,  #
        'xavier_normal_': torch.nn.init.xavier_normal,  #
        'orthogonal_': torch.nn.init.orthogonal_,  #
    }

    #用日志包，得到想要的信息
    #日记包的基础配置
    logging.basicConfig(filename="ModelLog.txt",filemode="a+",
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S'
                        )



    opt.model_class =model_classes[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 如果GPU可用，用GPU
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer] #优化器选择

    ins=Instructor(opt)
    ins.run(opt.repeat)




if __name__ == '__main__':
    main()