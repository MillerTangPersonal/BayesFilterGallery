'''
some helpling functions for the UKF
'''
import numpy as np
from numpy.linalg import inv
import math
def isin(t_np,t_k):
    # check if t_k is in the numpy array t_np. If t_k is in t_np, return the index and bool = Ture.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        b = True
        return res[0][0], b
    b = False
    return 0, b

def cross(v):    # input: 3x1 vector, output: 3x3 matrix
    vx = np.array([
        [ 0,    -v[2], v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1],  v[0], 0 ] 
    ])
    return vx

def denormalize(scl, norm_data):
    # @param: scl: the saved scaler   norm_data: 
    norm_data = norm_data.reshape(-1,1)
    new = scl.inverse_transform(norm_data)
    return new


def getSigmaP(X, L, kappa, dim):
    # get sigma points
    # return sigma_points, the num of sigma points
    X = X.reshape(-1,1)
    w = np.zeros((dim,1))
    mu = np.concatenate((X, w), axis=0)
    L_num = mu.shape[0]
    Z_SP= mu

    for idx in range(L_num):
        # i=idx: 1,...,L 
        z_i  = mu + math.sqrt(L_num + kappa) * L[:, idx].reshape(-1,1)
        z_iL = mu - math.sqrt(L_num + kappa) * L[:, idx].reshape(-1,1)
        Z_SP = np.hstack((Z_SP, z_i))
        Z_SP = np.hstack((Z_SP, z_iL))
        
    # Z_SP = [z_0; z_1, z_(1+L), ..., z_i, z_(i+L)]
    return Z_SP, Z_SP.shape[1], L_num

def getAlpha(idx,L_num,kappa):
    if idx == 0:
        alpha = kappa/(L_num + kappa)
    else:
        alpha = 1/(2*(L_num + kappa))
    return alpha

def wrapToPi(err_th):
    # wrap the theta error to [-pi, pi]
    # wrap a scalar angle error
    while err_th < -math.pi:
        err_th += 2 * math.pi
    while err_th > math.pi:
        err_th -= 2 * math.pi
    return err_th

def wrapToPi_vector(err_th):
    # wrap a vector angle error
    # wrap to [-pi, pi]
    err_th = (err_th + np.pi) % (2 * np.pi) - np.pi
    return err_th