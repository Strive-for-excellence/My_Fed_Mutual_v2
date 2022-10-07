import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import  rand
A = np.array([[0, 2], [1, 1], [2, 0]])
# A =  np.sum(A,axis=1)
print(np.sum(A,axis=0))