import numpy as np
import math
import string, os
from scipy import linalg, sparse
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.linalg import block_diag
import numpy.matlib


block = np.array([[1,2,3],[2,3,4],[3,4,5]])
t = 10 
# time = np.array([11,12,13,14,15,16,17,18,19,20]).reshape(-1,1)
# t = time.shape[0]
# for k in range(1, t): 
#     print(time[k])

# print("\n")

# for k in range(t-1,-1,-1):
#     print(time[k])

x_dr = np.array([1.0,2.0,3.0,4.0,5.0])
x_op = np.copy(x_dr) 

for i in range(5):
    x_op[i] = x_op[i] +0.02

print(x_dr)
print(x_op)