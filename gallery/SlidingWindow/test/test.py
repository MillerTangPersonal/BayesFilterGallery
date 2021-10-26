import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag


x_op_k1 = np.array([2.88057, 0.04928, -2.91152],dtype=float)


dt = 0.1
r = np.array([1,2,3])
C = np.eye(3) 
r = r.reshape(-1,1)

T = np.block([[C,  -C.dot(r)],
                [0, 0, 0, 1]])

G = T[0:3,0:3]
r = T[0:3,3]
print(r)
