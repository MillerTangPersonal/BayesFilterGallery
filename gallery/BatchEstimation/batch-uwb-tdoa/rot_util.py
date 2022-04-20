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

'''compute right Jacobian with phi'''
def rightJacob(phi):
    norm_phi = linalg.norm(phi)
    if norm_phi == 0:
        return np.eye(3)
    else:
        term1 = (1.0 - math.cos(norm_phi)) / (norm_phi**2)
        term2 = (norm_phi - math.sin(norm_phi)) / (norm_phi**3)
        phi_skew = skew(phi)

        return np.eye(3) - term1 * phi_skew + term2 * phi_skew.dot(phi_skew) 