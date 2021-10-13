''' sliding window filter for a 2D estimation'''
import numpy as np
import math
import string, os
from scipy import linalg
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.linalg import block_diag
import numpy.matlib

import matplotlib.pyplot as plt
import matplotlib.style as style
# import help functions
from util import getAlpha, wrapToPi

np.set_printoptions(precision=5)

# motion_model
# propgate the state and noise
def motion_model(x, T, v, om):
    A = np.array([[T * math.cos(x[2]), 0],
                  [T * math.sin(x[2]), 0],
                  [0                 , T]])
    V = np.array([v,  om]).reshape(-1,1)
    x_new = x + np.squeeze(A.dot(V))
    x_new[2] = wrapToPi(x_new[2])
    return x_new.reshape(-1,1)

def nv_propagation(x, T, v_var, om_var):
    # input sensor noise
    Q_var = np.diag([v_var[0,0], om_var[0,0]]); 
    A = np.array([[T * math.cos(x[2]), 0],
                  [T * math.sin(x[2]), 0],
                  [0                 , T]])

    Q_prop = A.dot(Q_var).dot(A.T)
    return Q_prop

# meas. model
def meas_model(x, l_xy):
    # x = [x_k, y_k, theta_k]
    # l_xy = [x_l, y_l]
    x_m = l_xy[0]-x[0]-d*math.cos(x[2])
    y_m = l_xy[1]-x[1]-d*math.sin(x[2])

    r_m = np.sqrt( x_m**2 + y_m**2 )
    phi_m = np.arctan2(y_m, x_m) - x[2]

    meas = np.array([r_m, phi_m])
    return meas.reshape(-1,1)


def get_ex(x, x_w1, w1, w_len, r_meas, b_meas, l, r_var, b_var):
    # input x is a 3 x w_len numpy array
    R = np.diag([r_var[0,0], b_var[0,0]])

    # init. to save the error in motion and meas
    e_v = np.empty(0)
    e_y = np.empty(0)
    # save the propagated motion noise and meas. noise
    W_v = np.empty((0,0))
    W_y = np.empty((0,0))

    e_0_v = x[:,0].reshape(-1,1) - x_w1.reshape(-1,1)
    W_pro = nv_propagation(x_w1, T, v_var, om_var)

    e_v = np.append(e_v, np.squeeze(e_0_v))
    # save W_pro
    W_v = block_diag(W_v, W_pro)


    # the landmark idx 
    l_idx = np.nonzero(r_meas[w1,:])
    l_idx = np.squeeze(np.asarray(l_idx))

    R_num = np.count_nonzero(r_meas[w1,:],axis=0)
    # iterate over all the meas. at init time
    # stack to e_y
    for i in range(R_num):  
        l_xy = l[l_idx[i], :]             # the r_idx th landmark pos.
        r_land = r_meas[w1, l_idx[i]]     # [timestamp, id of landmark]
        b_land = b_meas[w1, l_idx[i]]
        y_w1 = np.array([r_land, b_land]).reshape(-1,1)

        e_0_y = y_w1 - meas_model(x[:,0], l_xy)    
        # compute angle error, wrap to pi
        e_0_y[1] = wrapToPi(e_0_y[1])
        e_y = np.append(e_y, np.squeeze(e_0_y))
        # --------- save the variance of meas. ------------ #
        W_y = block_diag(W_y, R)

    # loop over the window size
    for idx in range(w_len-1):
        idx = idx + 1
        # ------------ get the error and noise variance in motion --------- #
        # propagate input noise through motion model
        W_pro = nv_propagation(x[:,idx-1], T, v_var, om_var)
        x_pro = motion_model(x[:,idx-1], T, v[idx], om[idx])

        # get the error in motion (wrap the angle error to Pi)
        e_idx_v = x_pro- x[:, idx-1].reshape(-1,1);  e_idx_v[2] = wrapToPi(e_idx_v[2])
        
        # append the motion error
        e_v = np.append(e_v, np.squeeze(e_idx_v))
        # save W_pro
        W_v = block_diag(W_v, W_pro)

        # ----------------- get the error in measurements ----------------- #
        l_idx = np.nonzero(r_meas[w1+idx,:])
        l_idx = np.squeeze(np.asarray(l_idx))

        R_num = np.count_nonzero(r_meas[w1+idx,:],axis=0)
        # iterate over all the meas. at timestamp w1+idx
        # stack to e_y
        for j in range(R_num): 
            l_xy = l[l_idx[j], :]             # the r_idx th landmark pos.
            r_l = r_meas[w1+idx, l_idx[j]]     # [timestamp, id of landmark]
            b_l = b_meas[w1+idx, l_idx[j]]
            y = np.array([r_l, b_l]).reshape(-1,1)
            e_idx_y = y - meas_model(x[:,0], l_xy)
            # compute angle error, wrap to pi
            e_idx_y[1] = wrapToPi(e_idx_y[1])
            e_y = np.append(e_y, np.squeeze(e_idx_y))

            # --------- save the variance of meas. ------------ #
            W_y = block_diag(W_y, R)

    e_v = e_v.reshape(-1,1);  e_y = e_y.reshape(-1,1)

    return e_v, e_y, W_v, W_y


if __name__ == "__main__":
    # load data
    os.chdir("/home/wenda/BayesFilterGallery/gallery")   
    curr = os.getcwd()
    # load .mat data
    t = loadmat(curr+'/dataset2.mat')['t']
    x_true = loadmat(curr+'/dataset2.mat')['x_true']
    y_true = loadmat(curr+'/dataset2.mat')['y_true']
    th_true = loadmat(curr+'/dataset2.mat')['th_true']
    l = loadmat(curr+'/dataset2.mat')['l']
    # measurements
    r_meas = loadmat(curr+'/dataset2.mat')['r'];   r_var = loadmat(curr+'/dataset2.mat')['r_var']
    b_meas = loadmat(curr+'/dataset2.mat')['b'];   b_var = loadmat(curr+'/dataset2.mat')['b_var']
    # inputs
    v = loadmat(curr+'/dataset2.mat')['v'];   v_var = loadmat(curr+'/dataset2.mat')['v_var']
    om = loadmat(curr+'/dataset2.mat')['om']; om_var = loadmat(curr+'/dataset2.mat')['om_var']
    d = loadmat(curr+'/dataset2.mat')['d']

    vicon_gt = np.concatenate([x_true, y_true], axis=1)
    vicon_gt = np.concatenate([vicon_gt, th_true], axis=1)
    K = t.shape[0]
    T = 0.1

    # -------------------- initial Position ---------------------- #
    X0 = np.zeros(3)
    X0[0] = x_true[0]
    X0[1] = y_true[0]
    X0[2] = th_true[0]
    Xpr = np.zeros((3,K));  Xpo = np.zeros((3,K))
    Xpr[:,0] = X0;    Xpo[:,0] = X0
    # Initial covariance
    P0 = np.diag([0.01, 0.01, 0.01])
    Ppr = np.zeros((K, 3, 3));      Ppo = np.zeros((K, 3, 3))
    Ppr[0] = P0;                    Ppo[0] = P0     
    # noise
    # Q = np.diag([v_var[0,0], om_var[0,0]]);   R = np.diag([r_var[0,0], b_var[0,0]])
   
    # Set Max range
    r_max = 10

    for i in range(r_meas.shape[0]):
        for j in range(r_meas.shape[1]):
            if r_meas[i,j] > r_max:
                r_meas[i,j] = 0.0

    # test code
    w1 = 600;               w2 = 610; 
    t_wd = t[w1:w2, :]

    # measurement
    r_wd = r_meas[w1:w2, :]; b_wd = b_meas[w1:w2, :]
    # input
    v_wd  = v[w1:w2, :];     om_wd = om[w1:w2, :]
    # ground truth
    x_wd_true = np.concatenate((x_true[w1:w2,:], y_true[w1:w2,:], th_true[w1:w2,:]), axis=1).T

    # prior from previous
    x_w1 = np.array([x_true[w1], y_true[w1], th_true[w1]])
    # dead reckon for operating point
    w_len = w2 - w1

    x_op = np.zeros((3, w_len))
    x_op[:,0] = np.squeeze(x_w1)            # gt for state at t_w1

    # dead reckoning 
    for idx in range(w_len-1):    # 0 ~ (w_len-1) 
        idx = idx + 1             # 1 ~ w_len 

        x_pro = motion_model(x_op[:,idx-1], T, v_wd[idx], om_wd[idx])
        x_op[:,idx] = np.squeeze(x_pro)



    # cost function
    # construct error vector
    e_v, e_y, W_v, W_y = get_ex(x_op, x_w1, w1, w_len, r_meas, b_meas, l, r_var, b_var)

    # stack error in motion and meas. 
    ex = np.concatenate((e_v,e_y),axis=0)

    # construct covariance matrix
    W = block_diag(W_v, W_y)

    J = 0.5 * ex.T.dot(np.linalg.inv(W)).dot(ex)


    # debug
    plt.plot(x_wd_true[0,:], x_wd_true[1,:], label='gt')
    plt.plot(x_op[0,:], x_op[1,:],label='dr')
    plt.xlim([-2, 15])
    plt.ylim([-4, 4])
    plt.legend()
    plt.show()

