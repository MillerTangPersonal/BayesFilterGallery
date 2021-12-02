'''Util functions for rotation operation'''
import numpy as np
import math
from scipy import linalg

'''skew operation'''
def skew(r):
    r = np.squeeze(r)
    r_skew = np.array([[ 0.0,   -r[2],   r[1]],
                       [ r[2],   0.0,   -r[0]],
                       [-r[1],   r[0],    0.0]], dtype=float)
    return r_skew

'''help function'''
def zeta(phi):
    # equ. (101) in eskf
    phi_norm = np.linalg.norm(phi)
    if phi_norm == 0:
        dq = np.array([1, 0, 0, 0])
    else:
        dq_xyz = (phi*(math.sin(0.5*phi_norm)))/phi_norm
        dq = np.array([math.cos(0.5*phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
    return dq