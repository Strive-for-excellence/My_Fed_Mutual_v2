from numpy import *
from numpy.random import  rand
A = array([[0, 2], [1, 1], [2, 0]]).T
print(A)  # [[0 1 2][2 1 0]]
B = cov(A)  # 默认行为变量计算方式，即X为行，Y也为行
print(B)  # [[ 1. -1.][-1.  1.]]
C = cov(A, rowvar=False)  # 此时列为变量计算方式 即X为列，Y也为列
print(C)  # [[ 2.  0. -2.][ 0.  0.  0.][-2.  0.  2.]]
