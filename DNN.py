#!/usr/bin/env python

import sys
import os
import pickle as pkl       #数据持久存储模块
import time 
import random 

import numpy as np
import theano.tensor as T
import theano
import pylearn2.train
import pylearn2.models.mlp as p2_md_mlp
import pylearn2.datasets.dense_design_matrix as p2_dt_dd
import pylearn2.training_algorithms.sgd as p2_alg_sgd      #SGD 随机梯度下降算法
#A module containing different learning rules for use with the SGD training algorithm.
import pylearn2.training_algorithms.learning_rule as p2_alg_lr  
import pylearn2.costs.mlp.dropout as p2_ct_mlp_dropout  #dropout防止过拟合
#Termination criteria used to determine when to stop running a training algorithm.
import pylearn2.termination_criteria as p2_termcri  
from numpy import dtype   #create a data type object


def main():
    base_name = sys.argv[1]        # 获取第一个参数   sys.argv[ ]记录（获取）命令行参数  sys(system)  argv(argument variable)参数变量，该变量为list列表
    n_epoch = int(sys.argv[2])     #获取第二个参数
    n_hidden = int(sys.argv[3])    #获取第三个参数作为隐层神经元个数
    include_rate = float(sys.argv[4])

    in_size = 1001          #输入层神经元个数（标记基因个数）
    out_size = 1            #输出层神经元个数
    b_size = 200            #偏差值
    l_rate = 5e-4           #学习速率
    l_rate_min = 1e-5       #学习速率最小值
    decay_factor = 0.9      #衰减因数
    lr_scale = 3.0         
    momentum = 0.5
    init_vals = np.sqrt(6.0/(np.array([in_size, n_hidden])+np.array([n_hidden, out_size])))  #初始值，返回平方根

    print 'loading data...'  #显示载入数据

    X_tr = np.load('geno_X_tr_float64.npy')        # tr(traing)以numpy专用二进制类型保存训练数据集的数据
    Y_tr = np.load('pheno_Y_tr_0-4760_float64.npy')  
    Y_tr_pheno = np.array(Y_tr)
    X_va = np.load('geno_X_va_float64.npy')        #验证集（模型选择，在学习到不同复杂度的模型中，选择对验证集有最小预测误差的模型）
    Y_va = np.load('pheno_Y_va_0-4760_float64.npy')
    Y_va_target = np.array(Y_va)                  
    X_te = np.load('geno_te_float64.npy')        #测试集（对学习方法的评估）
    Y_te = np.load('pheno_Y_te_0-4760_float64.npy')
    Y_te_target = np.array(Y_te)

    

    random.seed(0)   #设置生成随机数用的整数起始值。调用任何其他random模块函数之前调用这个函数
    monitor_idx_tr = random.sample(range(88807), 5000)   #监测训练
    #将训练数据集类型设为32位浮点型，The DenseDesignMatrix class and related code Functionality for representing data that can be described as a dense matrix (rather than a sparse matrix) with each row containing an example and each column corresponding to a different feature.
    data_tr = p2_dt_dd.DenseDesignMatrix(X=X_tr.astype('float32'), y=Y_tr.astype('float32'))
    X_tr_monitor, Y_tr_monitor_target = X_tr[monitor_idx_tr, :], Y_tr_target[monitor_idx_tr, :]
    #一个隐层，用Tanh（）作激活函数; 输出层用线性函数作激活函数
    h1_layer = p2_md_mlp.Tanh(layer_name='h1', dim=n_hidden, irange=init_vals[0], W_lr_scale=1.0, b_lr_scale=1.0) 
    o_layer = p2_md_mlp.Linear(layer_name='y', dim=out_size, irange=0.0001, W_lr_scale=lr_scale, b_lr_scale=1.0)
    #Multilayer Perceptron；nvis(Number of “visible units” input units)  layers(a list of layer objects，最后1层指定MLP的输出空间) 
    model = p2_md_mlp.MLP(nvis=in_size, layers=[h1_layer, o_layer], seed=1)
    dropout_cost = p2_ct_mlp_dropout.Dropout(input_include_probs={'h1':1.0, 'y':include_rate}, 
                                             input_scales={'h1':1.0, 
                                                           'y':np.float32(1.0/include_rate)})
    #随机梯度下降法
    algorithm = p2_alg_sgd.SGD(batch_size=b_size, learning_rate=l_rate, 
                               learning_rule = p2_alg_lr.Momentum(momentum),
                               termination_criterion=p2_termcri.EpochCounter(max_epochs=1000),
                               cost=dropout_cost)
    #训练 根据前面的定义 ：dataset为一个密集型矩阵，model为MLP多层神经网络，algorithm为SGD
    train = pylearn2.train.Train(dataset=data_tr, model=model, algorithm=algorithm)
    train.setup()

    x = T.matrix()             #定义为一个二维数组
    #fprop(state_below) does the forward prop transformation
    y = model.fprop(x)  
    f = theano.function([x], y)  #定义一个function函数，输入为x,输出为y

    MAE_va_old = 10.0      #平均绝对误差
    MAE_va_best = 10.0
    MAE_tr_old = 10.0      #训练误差
    MAE_te_old = 10.0
    MAE_1000G_old = 10.0
    MAE_1000G_best = 10.0
    MAE_GTEx_old = 10.0
    #base_name = sys.argv[1]      # 获取第一个参数   sys.argv[ ]记录（获取）命令行参数
    outlog = open(base_name + '.log', 'w')   
    log_str = '\t'.join(map(str, ['epoch', 'MAE_va', 'MAE_va_change', 'MAE_te', 'MAE_te_change', 
                              'MAE_tr', 'MAE_tr_change', 'learing_rate', 'time(sec)']))     
    print log_str     #输出运行日志
    outlog.write(log_str + '\n')
    #Python的标准输出缓冲（这意味着它收集“写入”标准出来之前，将其写入到终端的数据）。调用sys.stdout.flush()强制其“缓冲
    sys.stdout.flush()   

    for epoch in range(0, n_epoch):
        t_old = time.time()
        train.algorithm.train(train.dataset)

        Y_va_hat = f(X_va.astype('float32')).astype('float64')
        Y_te_hat = f(X_te.astype('float32')).astype('float64')
        Y_tr_hat_monitor = f(X_tr_monitor.astype('float32')).astype('float64')
       
        #计算平均绝对误差
        MAE_va = np.abs(Y_va_target - Y_va_hat).mean()  
        MAE_te = np.abs(Y_te_target - Y_te_hat).mean()
        MAE_tr = np.abs(Y_tr_monitor_target - Y_tr_hat_monitor).mean()
       
        #误差变换率
        MAE_va_change = (MAE_va - MAE_va_old)/MAE_va_old    
        MAE_te_change = (MAE_te - MAE_te_old)/MAE_te_old
        MAE_tr_change = (MAE_tr - MAE_tr_old)/MAE_tr_old
       
        #将old误差值更新为当前误差值
        MAE_va_old = MAE_va       
        MAE_te_old = MAE_te
        MAE_tr_old = MAE_tr
       
        #返回当前的时间戳（1970纪元后经过的浮点秒数）
        t_new = time.time()
        l_rate = train.algorithm.learning_rate.get_value()
        log_str = '\t'.join(map(str, [epoch+1, '%.6f'%MAE_va, '%.6f'%MAE_va_change, '%.6f'%MAE_te, '%.6f'%MAE_te_change,
                                  '%.6f'%MAE_tr, '%.6f'%MAE_tr_change, '%.5f'%l_rate, int(t_new-t_old)]))
        print log_str
        outlog.write(log_str + '\n')
        sys.stdout.flush()

        if MAE_tr_change > 0:           #训练误差变换率大于0时，学习速率乘上一个衰减因子
            l_rate = l_rate*decay_factor
        if l_rate < l_rate_min:         #学习速率小于最小速率时，更新为最小速率
            l_rate = l_rate_min

        train.algorithm.learning_rate.set_value(np.float32(l_rate))

        if MAE_va < MAE_va_best:
            MAE_va_best = MAE_va
            outmodel = open(base_name + '_bestva_model.pkl', 'wb')
            pkl.dump(model, outmodel)
            outmodel.close()    
            np.save(base_name + '_bestva_Y_te_hat.npy', Y_te_hat)
            np.save(base_name + '_bestva_Y_va_hat.npy', Y_va_hat)

        

    print 'MAE_va_best : %.6f' % (MAE_va_best)
    outlog.write('MAE_va_best : %.6f' % (MAE_va_best) + '\n')
    outlog.close()

if __name__ == '__main__':
    main()

