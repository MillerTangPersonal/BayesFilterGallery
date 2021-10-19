import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag


x_op_k1 = np.array([2.88057, 0.04928, -2.91152])


dt = 0.1

Qi = np.array([[0.00442, 0.0],
               [0.0, 0.00819]
            ])

x = np.squeeze(x_op_k1)

x = np.array([2.88488, 0.05029, -2.91141])

A = np.array([[dt * math.cos(x[2]), 0],
              [dt * math.sin(x[2]), 0],
              [0                 , dt]])

W_prop = A.dot(Qi).dot(A.T)
print(W_prop)
