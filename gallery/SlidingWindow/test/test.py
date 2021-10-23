import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag


x_op_k1 = np.array([2.88057, 0.04928, -2.91152])


dt = 0.1

A = np.array([[1,2,3], [2,3,4], [3,4,5]])

A = A[1:,:]
A[-1,:] = np.array([0,0,0])
print(A[-1,:]) 
