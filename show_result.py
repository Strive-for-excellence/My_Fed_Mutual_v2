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

    file_name = [r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_0_alpha_5).json',
                 r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_1_alpha_5).json',
                 r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5).json',
                 r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_1_kalman_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_4_alpha_d_1).json'
                 r'Z:\zhangxingyan\My_Fed_mutual/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_4_alpha_d_1_kalman_1).json']
    #  r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(10)_E(1)_trainnum(1000)_P(1)_name(fedmd_avg_2_alpha_5_kalman_10).json']
    # /root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(4)_E(1)_trainnum(10000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_60).json
                #   /root/My_Fed_mutual/save/Result_dataset(cifar10)_R(300)_N(4)_E(1)_trainnum(10000)_P(1)_name(cifar10_fedmd_avg_0_alpha_d_10).json
    file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_0_alpha_d_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_1_alpha_d_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_1_kalman_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_4_alpha_d_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_4_alpha_d_1_kalman_1).json']
    #                                                    Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_0_alpha_d_1).json
    file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_0_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_1_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_5_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_2_alpha_d_0.01_kalman_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_4_alpha_d_0.01_kalman_1).json']

    file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_fedmd_avg_0_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_fedmd_avg_1_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_fedmd_avg_5_alpha_d_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_fedmd_avg_2_alpha_d_0.01_kalman_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(20)_E(1)_trainnum(1000)_P(1)_lr(0.01)_name(cifar10_fedmd_avg_4_alpha_d_0.01_kalman_1).json']

    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_0_alpha_d_100).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_1_alpha_d_100).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_5_alpha_d_100).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_2_alpha_d_100_kalman_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_lr(0.001)_name(cifar10_fedmd_avg_4_alpha_d_100_kalman_1).json']


    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_0_alpha_d_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_1_alpha_d_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual/save/Result_dataset(cifar10)_R(1000)_N(4)_E(1)_trainnum(1000)_P(1)_name(cifar10_fedmd_avg_2_alpha_d_1_kalman_1).json']
    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_0_alpha_d_0.05).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_1_alpha_d_0.05).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_2_alpha_d_0.05).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_2_alpha_d_0.05_kalman_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_4_alpha_d_0.05).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2/save/Result_dataset(cifar100)_R(1000)_N(5)_E(1)_trainnum(1000)_P(1)_name(cifar100_fedmd_avg_4_alpha_d_0.05_kalman_1).json']
    #                                                    Result_dataset(mnist)_R(1000)_N(5)_E(1)_trainnum(1000)_P(2)_name(mnist_fedmd_avg_1_alpha_5).json

    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_0_alpha_5).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_1_alpha_5).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_5_alpha_5).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_2_alpha_5_kalman_1).json']
    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_0_d_alpha_50).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_1_d_alpha_50).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_5_d_alpha_50).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(5)_E(1)_trainnum(100)_P(1)_name(mnist_fedmd_avg_4_d_alpha_50_kalman_1).json']
    file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_0_d_alpha_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_1_d_alpha_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_5_d_alpha_0.01).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_2_d_alpha_0.01_kalman_1).json',
                 r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_4_d_alpha_0.01_kalman_1).json']
    # file_name = [r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_0_alpha_3).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_1_alpha_3).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_5_alpha_3).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_2_alpha_3_kalman_1).json',
    #              r'Z:\zhangxingyan\My_Fed_mutual_v2\save\Result_dataset(mnist)_R(300)_N(20)_E(1)_trainnum(100)_P(1)_lr(0.01)_name(mnist_fedmd_avg_4_alpha_3_kalman_1).json']
    #
    # #
    name = ["individual","fedmd","based on confidence","based on uncertainty ","based on unceratinty T = 20"]

#     name = ['1 individual','2 avg','3 weight_avg','4 weight_avg_kalman','5 weight_avg_MCdropout','6 weight_avg_kalman_MCdropout','7 weight_avg_kalman_MCdropout_10','8','9']
    for  i in range(10,100):
        name.append(str(i))

    for i in range(0,len(file_name)):
        with open(file_name[i]) as f:
            f  = json.load(f)
            num = len(f['test_acc'])
            # for i in range(num):
            plt.plot(f['test_acc'], label=name[i])

            # if i < 4:
            #     plt.plot(f['test_acc'],'r-',label=name[i])
            # else:
            #     plt.plot(f['test_acc'],'g--',label=name[i])

            print(f"idx = {i} size = {len(f['test_acc'])}",end=' ')
                # print(f"client = {i} best_acc")
            epochs = len(f['test_acc'])
            print('last_acc = {} '.format(f['test_acc'][epochs-1]),end = ' ')
            print('best_acc = {} '.format(max(f['test_acc'])))

            # print(f['test_acc'][200])
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
    # plt.title('mnist dataset dirichlet alpha=0.01')
    plt.title('mnist dataset pathological alpha=5')
    plt.show()  
    plt.savefig('test.png')