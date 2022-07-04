import shelve
import torch
from matplotlib import  pyplot as plt
# from models.Nets import CNNMnist_Transfer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    file_name = ['/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_0_alpha_5).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_1_alpha_5).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5).json',
                #  '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5_kalman_1).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5_kalman_2).json']
                #  '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5_kalman_5).json']
    # /root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(4)_E(1)_trainnum(10000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_60).json
                #   /root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(4)_E(1)_trainnum(10000)_P(1)_name(cifar10_fedmd_avg_0_alpha_d_5).json
    # file_name = ['/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(500)_N(4)_E(1)_trainnum(5000)_P(1)_name(cifar10_fedmd_avg_0_alpha_d_60).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(500)_N(4)_E(1)_trainnum(5000)_P(1)_name(cifar10_fedmd_avg_1_alpha_d_60).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(500)_N(4)_E(1)_trainnum(5000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_60).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(500)_N(4)_E(1)_trainnum(5000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_60_kalman_1).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar10)_R(500)_N(4)_E(1)_trainnum(5000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_60_kalman_2).json']
             
    file_name = ['/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(500)_N(5)_E(1)_trainnum(5000)_P(1)_name(cifar100_fedmd_avg_0_alpha_60).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(500)_N(5)_E(1)_trainnum(5000)_P(1)_name(cifar100_fedmd_avg_1_alpha_d_60).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(500)_N(5)_E(1)_trainnum(5000)_P(1)_name(cifar100_fedmd_avg_2_alpha_d_60).json',
                 '/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(500)_N(5)_E(1)_trainnum(5000)_P(1)_name(cifar100_fedmd_avg_2_alpha_d_60_kalman_1).json']
    #             #   /root/My_Fed_mutual/save/Result_dataset(cifar100)_R(300)_N(4)_E(1)_trainnum(10000)_P(1)_name(cifar100_fedmd_avg_1_alpha_d_60).json
    # file_name = ['/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(300)_N(20)_E(1)_trainnum(500)_P(1)_name(cifar100_fedmd_avg_0_alpha_d_60).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(300)_N(20)_E(1)_trainnum(500)_P(1)_name(cifar100_fedmd_avg_1_alpha_d_60).json',
    #              '/root/My_Fed_mutual/save/Result_dataset(cifar100)_R(300)_N(20)_E(1)_trainnum(500)_P(1)_name(cifar100_fedmd_avg_2_alpha_d_60).json']
#   a = '/sa'

# /root/My_Fed_mutual/save/Result_dataset(cifar100)_R(300)_N(20)_E(1)_trainnum(500)_P(1)_name(cifar100_fedmd_avg_0_alpha_d_1).json
    name = ['individual','avg','weight_avg','weight_avg_kalman_1','weight_avg_kalman_2','weight_avg_kalman_5']
    for i in range(0,len(file_name)):   
        with open(file_name[i]) as f:
            f  = json.load(f)
            num = len(f['test_acc'])
            # for i in range(num):
            plt.plot(f['test_acc'],label=name[i])
            print(f"idx = {i} size = {len(f['test_acc'])}",end=' ')
                # print(f"client = {i} best_acc")
            epochs = len(f['test_acc'])
            print('last_acc = {} '.format(f['test_acc'][epochs-1]),end = ' ')
            print('best_acc = {} '.format(max(f['test_acc'])))

            # print(f['test_acc'][200])
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.legend()

    plt.show()  
    plt.savefig('test.png')