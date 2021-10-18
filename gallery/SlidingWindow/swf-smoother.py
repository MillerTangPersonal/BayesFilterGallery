''' sliding window filter for a 2D estimation using RTS smoother'''
import numpy as np
import string, os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# import RTS-smoother
from RTS_smoother import RTS_Smoother_2D

np.set_printoptions(precision=5)


if __name__ == "__main__":
    # load data
    os.chdir("/home/wenda/BayesFilterGallery/gallery")   
    curr = os.getcwd()
    # load .mat data
    t = loadmat(curr+'/dataset2.mat')['t']
    # landmark
    l = loadmat(curr+'/dataset2.mat')['l']
    # inputs
    v = loadmat(curr+'/dataset2.mat')['v'];   v_var = loadmat(curr+'/dataset2.mat')['v_var']
    om = loadmat(curr+'/dataset2.mat')['om']; om_var = loadmat(curr+'/dataset2.mat')['om_var']
    d = loadmat(curr+'/dataset2.mat')['d']
    # measurements
    r_meas = loadmat(curr+'/dataset2.mat')['r'];   r_var = loadmat(curr+'/dataset2.mat')['r_var']
    b_meas = loadmat(curr+'/dataset2.mat')['b'];   b_var = loadmat(curr+'/dataset2.mat')['b_var']
    # ground truth
    x_true = loadmat(curr+'/dataset2.mat')['x_true']
    y_true = loadmat(curr+'/dataset2.mat')['y_true']
    th_true = loadmat(curr+'/dataset2.mat')['th_true']
    vicon_gt = np.concatenate([x_true, y_true, th_true], axis=1)
    # total timestamp
    K = t.shape[0];        T = 0.1  # time duration, 10 Hz

    # initial position 
    X0 = vicon_gt[0,:]
    # initial covariance
    P0 = np.diag([1, 1, 0.01])
    # input noise
    Q = np.diag([v_var[0,0], om_var[0,0]])
    # meas. noise
    R = np.diag([r_var[0,0], b_var[0,0]])
    # filter the measurements
    r_max = 10
    for i in range(r_meas.shape[0]):
        for j in range(r_meas.shape[1]):
            if r_meas[i,j] > r_max:
                r_meas[i,j] = 0.0

    smoother = RTS_Smoother_2D(P0, Q, R, l, d, K)

    # compute the operating point initially
    # compute operating points
    x_op = np.zeros((3*K, 1))    # column vector
    x_op[0:3] = X0.reshape(-1,1)
    for k in range(1, K):     # k = 1 : K-1 
        # compute operating point x_op (dead reckoning)
        x_op[3*k : 3*k+3] = smoother.motion_model(x_op[3*k-3 : 3*k], T, v[k], om[k])


    # Gauss-Newton 
    # in each iteration, 
    # (1) do one batch estimation for dx 
    # (2) update the operating point x_op
    # (3) check the convergence

    iter = 0;      delta_p = 1;    max_iter = 10; 
    while (iter < max_iter) and (delta_p > 0.0001):
        iter = iter + 1; 
        error = 0
        # full batch estimation

        # RTS smoother forward pass
        for k in range(1, K):     # k = 1 : K-1 
            # operting point 
            x_op_k1 = x_op[3*k-3, 3*k]
            x_op_k = x_op[3*k : 3*k+3]

            # measurements at timestamp k
            r_k = r_meas[k,:]
            b_k = b_meas[k,:]

            # forward pass
            smoother.forward(x_op_k1, x_op_k, v[k], om[k], r_k, b_k, T, k)

        # RTS smoother backward pass
        for k in range(K,-1,1):
            smoother.backward(k)

        # after forward and backward pass
        # update x_op_k one by one
        for i in range(k):
            x_op[3*k : 3*k+3] = x_op[3*k : 3*k+3] + smoother.dXpo[k,:].reshape(-1,1)
            # update delta x error (only check to x, y)
            error = error + np.sqrt(smoother.dXpo[k,0]**2 + smoother.dXpo[k,1]**2)

        delta_p = error / (K + 1)
