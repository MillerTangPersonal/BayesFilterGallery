''' sliding window filter for a 2D estimation'''
from matplotlib import colors
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
    x = np.squeeze(x)
    A = np.array([[T * math.cos(x[2]), 0],
                  [T * math.sin(x[2]), 0],
                  [0                 , T]])
    V = np.array([v,  om]).reshape(-1,1)
    x_new = x + np.squeeze(A.dot(V))
    x_new[2] = wrapToPi(x_new[2])
    return x_new.reshape(-1,1)

def nv_propagation(x, T, v_var, om_var):
    # provide the invert directly
    x = np.squeeze(x)
    # input sensor noise
    Q_var = np.diag([v_var[0,0], om_var[0,0]]); 
    A = np.array([[T * math.cos(x[2]), 0],
                  [T * math.sin(x[2]), 0],
                  [0                 , T]])

    # this noise propagation is not correct. why?
    # Q_prop = A.dot(Q_var).dot(A.T) + 1e-10*np.eye(3)          

    # test
    Q_prop = np.diag([0.0044, 0.0044, 0.0082])     # ????
    Q_inv = np.linalg.inv(Q_prop)
    return Q_inv

# meas. model
def meas_model(x, l_xy, d):
    # x = [x_k, y_k, theta_k]
    # l_xy = [x_l, y_l]
    x_m = l_xy[0]-x[0]-d*math.cos(x[2])
    y_m = l_xy[1]-x[1]-d*math.sin(x[2])

    r_m = np.sqrt( x_m**2 + y_m**2 )
    phi_m = np.arctan2(y_m, x_m) - x[2]
    
    # safety code: wrap to PI
    phi_m = wrapToPi(phi_m)
    meas = np.array([r_m, phi_m])
    return meas.reshape(-1,1)

def compute_F(x_op, T, v, om):
    F = np.array([[1, 0, -T * math.sin(x_op[2]) * v],
                  [0, 1, -T * math.cos(x_op[2]) * v],
                  [0, 0, 1]],dtype=float)
    return F

def compute_G(x_op, l_xy, d):
    # x_op = [x, y, theta]
    # denominator 1
    D1 = np.sqrt( (l_xy[0] - x_op[0] - d*math.cos(x_op[2]))**2 + (l_xy[1] - x_op[1] - d*math.sin(x_op[2]))**2 )
    # denominator 2
    D2 = ( l_xy[0]-x_op[0]-d*math.cos(x_op[2]) )**2 + ( l_xy[1]-x_op[1]-d*math.sin(x_op[2]) )**2

    g11 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
    g12 = -(l_xy[1] - x_op[1] - d*math.sin(x_op[2]))
    g13 = (l_xy[0] - x_op[0]) * d * math.sin(x_op[2]) - (l_xy[1] - x_op[1]) * d * math.cos(x_op[2])

    g21 = l_xy[1] - x_op[1] - d*math.sin(x_op[2])
    g22 = - (l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
    g23 = d**2 - d*math.cos(x_op[2])*(l_xy[0] - x_op[0]) - d*math.sin(x_op[2])*(l_xy[1] - x_op[1])

    G = np.array([[g11/D1, g12/D1, g13/D1],
                  [g21/D2, g22/D2, g23/D2 -1]])

    return np.squeeze(G)


# W_inv is very large

def construct_A_b(x_op, x_w1, w1, w_len, r_meas, b_meas, l, r_var, b_var, v_wd, om_wd, d):
    # decode x_op: from vector to matrix
    x_op = x_op.reshape(-1,3).T             # consider a better way for this convertion
    # input x is a 3 x w_len numpy array
    R_inv = np.diag([1.0/r_var[0,0], 1.0/b_var[0,0]])

    # init. to save the error in motion and meas
    e_v = np.empty(0);      e_y = np.empty(0)
    # save the propagated motion noise and meas. noise
    W_v_inv = np.empty((0,0));  W_y_inv = np.empty((0,0))

    # construct H matrix
    H_U = np.eye(3*(w_len+1))       # the uppper matrix has a fixed dimension
    H_L = np.empty((0,0))           # the lower matrix doesn't have a fixed dimension. It depends on how many measurements we get during this window size

    e_0_v = x_op[:,0].reshape(-1,1) - x_w1.reshape(-1,1)
    W_0_inv = nv_propagation(x_w1, T, v_var, om_var)

    e_v = np.append(e_v, np.squeeze(e_0_v))
    # save W_pro
    W_v_inv = block_diag(W_v_inv, W_0_inv)

    # ------------------- loop over the window size ------------------------#
    for idx in range(w_len):    # 0,1,2
        idx = idx + 1           # 1,2,3
        # ------------ get the error and noise variance in motion --------- #
        # propagate input noise through motion model
        W_inv = nv_propagation(x_op[:,idx-1], T, v_var, om_var)
        x_pro_idx = motion_model(x_op[:,idx-1], T, v_wd[idx], om_wd[idx])    # x_pro: 1, 2, 3

        # get the error in motion (wrap the angle error to Pi)
        e_idx_v = x_pro_idx- x_op[:, idx].reshape(-1,1);  e_idx_v[2] = wrapToPi(e_idx_v[2])
        # append the motion error
        e_v = np.append(e_v, np.squeeze(e_idx_v))

        # save W_pro
        W_v_inv = block_diag(W_v_inv, W_inv)

        # --------------- construct H_U matrix ------------------------ #
        H_U[(3*idx):(3+3*idx), (3*idx-3):(3*idx)] = -1*compute_F(x_op[:,idx-1], T, v_wd[idx], om_wd[idx])  # get F_{0,1,2}

    
    for k in range(w_len+1):    # 0,1,2,3
        # ----------------- get the error in measurements ----------------- #
        l_k = np.nonzero(r_meas[w1+k,:]);    l_k = np.asarray(l_k).reshape(-1,1)

        M = np.count_nonzero(r_meas[w1+k,:],axis=0)        # at timestamp k, we have M meas. in total

        # check if we have meas. or not
        # print(M)
        if M: 
            G = np.empty((0,3))                                # empty G to contain all the Gs in timestep k
            for m in range(M): 
                l_xy = l[l_k[m,0], :]                            # the m-th landmark pos.
                r_l = r_meas[w1+k, l_k[m,0]]                     # [timestamp, id of landmark]
                b_l = b_meas[w1+k, l_k[m,0]]
                y = np.array([r_l, b_l]).reshape(-1,1)
                e_k_y = y - meas_model(x_op[:,k], l_xy, d)
                # compute angle error, wrap to pi
                e_k_y[1] = wrapToPi(e_k_y[1])
                e_y = np.append(e_y, np.squeeze(e_k_y))

                # --------- save the variance of meas. ------------ #
                W_y_inv = block_diag(W_y_inv, R_inv)
                # compute G
                G_i = compute_G(x_op[:,k], l_xy, d)
                G = np.concatenate((G,G_i), axis=0)

            H_L = block_diag(H_L,G)
            
        else:
            # when no measurements, append dummy G, e_y, and W_y_inv
            G = np.zeros((1,3))    
            H_L = block_diag(H_L,G)
            e_k_y=np.array(0)
            e_y = np.append(e_y, np.squeeze(e_k_y))
            W_y_inv = block_diag(W_y_inv, 0)

    e_v = e_v.reshape(-1,1);  e_y = e_y.reshape(-1,1)


    # stack error in motion and meas. 
    ex = np.concatenate((e_v,e_y),axis=0)    

    # construct covariance matrix
    W_inv = block_diag(W_v_inv, W_y_inv)

    H = np.concatenate((H_U, H_L), axis=0)  # stack H_U, H_L vertically
    A = H.T.dot(W_inv).dot(H)
    b = H.T.dot(W_inv).dot(ex)

    return A, b

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
    T = 0.1                  # 10 Hz

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

    # Sliding window
    x_est = np.zeros((3,1250))

    for sw in range(1):
        print(sw)
        w1 = sw;               w2 = sw+50; 

        w1 = 500;               w2 = 600; 

        t_wd = t[w1:w2+1, :]
        # prior from previous estimate
        x_w1 = np.array([x_true[w1], y_true[w1], th_true[w1]])
        # measurement
        r_wd = r_meas[w1:w2+1, :]; b_wd = b_meas[w1:w2+1, :]
        # input
        v_wd  = v[w1:w2+1, :];     om_wd = om[w1:w2+1, :]
        # ground truth
        x_wd_true = np.concatenate((x_true[w1:w2+1,:], y_true[w1:w2+1,:], th_true[w1:w2+1,:]), axis=1).T

        # dead reckon for operating point
        w_len = w2 - w1
        # column vector x_op = [x_op[0]; x_op[1]; x_op[2];]
        x_op = np.zeros((3*(w_len +1),1))         # 602-600 = 2 => we have x_op[0], x_op[1], x_op[2]
        x_op[0:3] = x_w1.reshape(-1,1)            # gt for state at t_w1

        # dead reckoning 
        for idx in range(w_len):      # 0 ~ (w_len) 
            idx = idx + 1             # 1 ~ w_len+1 
            x_pro = motion_model(x_op[3*idx-3: 3*idx], T, v_wd[idx], om_wd[idx])
            x_op[3*idx: 3*idx+3] = x_pro.reshape(-1,1)

        # x_dr = x_op

        # Gauss-Newton 
        iter = 0;      delta_p = 1;    max_iter = 10; 

        while (iter < max_iter) and (delta_p > 0.0001):
            iter = iter + 1; 
            # print("\nIteration {0}\n".format(iter))
            A, b = construct_A_b(x_op, x_w1, w1, w_len, r_meas, b_meas, l, r_var, b_var, v_wd, om_wd, d)
            # For Ax=b, we can write a RTS smoother to solve it or using the numpy.linalg.solve (based on LU decomposition) to solve for x.
            dx = np.linalg.solve(A, b)

            # update the operating point
            x_op = x_op + dx

            # check for convergence
            err_x = dx.reshape(-1,3).T        # consider a better way
            error = 0
            for i in range(err_x.shape[1]):
                error = error + np.sqrt(err_x[0,i]**2 + err_x[1,i]**2)

            delta_p = error/(w_len+1)
        
        # print(delta_p)

        # save the est. at timestep sw
        x_est_sw = x_op.reshape(-1,3).T

        # x_est[:, sw] = x_est_sw[:,0]

        x_est = x_est_sw

    # x_dr = x_dr.reshape(-1,3).T

    plt.plot(x_est[0,:], x_est[1,:],'blue')
    plt.plot(x_true[500:799,0], y_true[500:799,0],'r')
    # plt.plot(x_dr[0,:], x_dr[1,:],'green')
    plt.xlim([-2, 10])
    plt.ylim([-3, 4])
    plt.show()






