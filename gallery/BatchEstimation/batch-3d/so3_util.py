'''Util functions for SO3 operation'''
import numpy as np
import math
from scipy import linalg

'''compute rotation matrix from axis-angle vector'''
def axisAngle_to_Rot(theta):
    # compute the norm
    theta = np.squeeze(theta)
    nm = linalg.norm(theta)
    if nm == 0:
        Psi = np.eye(3)
    else:
        rot_vec = (theta/nm).reshape(-1,1);     # column vector
        rot_skew = skew(rot_vec)
        Psi = math.cos(nm)*np.eye(3) + ((1-math.cos(nm))*rot_vec.dot(rot_vec.T)) - (math.sin(nm)*rot_skew)
        
    return Psi

'''skew operation'''
def skew(r):
    r = np.squeeze(r)
    r_skew = np.array([[0.0,   -r[2],   r[1]],
                       [r[2],      0,  -r[0]],
                       [-r[1],  r[0],    0.0]], dtype=float)
    return r_skew

'''compute transformation matrix from C and r'''
def getTrans(C, r):
    r = r.reshape(-1,1)
    T = np.block([[C,    -C.dot(r)],
                  [0,  0,  0,  1.0]])
    return T
