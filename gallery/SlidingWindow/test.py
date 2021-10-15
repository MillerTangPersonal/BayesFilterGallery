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
for k in range(1,t):
    print(k)