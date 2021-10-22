''' 2D Batch estimation using RTS smoother'''
import numpy as np
import os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
# import RTS-smoother
from util import wrapToPi
from RTS_smoother import RTS_Smoother_2D
from robot_2d import GroundRobot

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

    true_valid = loadmat(curr+'/dataset2.mat')['true_valid']

    # select a small amount of data for debugging
    w1 = 500; w2 = 550  # 12609
    t = t[w1 : w2];                   t = t - t[0,0]          #reset timestamp
    v = v[w1 : w2];                   om = om[w1 : w2]
    r_meas = r_meas[w1 : w2, :];      b_meas = b_meas[w1 : w2, :]
    vicon_gt = vicon_gt[w1 : w2,:]

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
    r_max = 5                                         
    for i in range(r_meas.shape[0]):
        for j in range(r_meas.shape[1]):
            if r_meas[i,j] > r_max:
                r_meas[i,j] = 0.0

    # ground robot
    robot = GroundRobot(Q, R, d, l ,T)
    # RTS smoother
    smoother = RTS_Smoother_2D(P0, robot, K)

    # compute the operating point initially
    # compute operating points
    x_dr = np.zeros((3*K, 1))    # column vector
    x_dr[0:3] = X0.reshape(-1,1)
    for k in range(1, K):     # k = 1 : K-1 
        # compute operating point x_op (dead reckoning)
        x_dr[3*k : 3*k+3] = robot.motion_model(x_dr[3*k-3 : 3*k], v[k], om[k])


    x_dr_v = x_dr.reshape(-1,3)

    # Gauss-Newton 
    # in each iteration, 
    # (1) do one batch estimation for dx 
    # (2) update the operating point x_op
    # (3) check the convergence
    iter = 0;       max_iter = 10; 
    delta_p = 1;    delta_an = 1; 
    x_op = np.copy(x_dr)

    while (iter < max_iter) and ((delta_p > 0.001) and (delta_an > 0.001)):
        iter = iter + 1; 
        error = 0;  an_error = 0
        print("\nIteration: #{0}\n".format(iter))
        # full batch estimation

        smoother.forward(x_op, v, om, r_meas, b_meas)

        smoother.backward()

        # after forward and backward pass
        # update x_op_k one by one
        for k in range(K):
            x_new = x_op[3*k : 3*k+3] + smoother.dXpo[k,:].reshape(-1,1)
            # wrap to Pi
            x_new[2] = wrapToPi(x_new[2])

            x_op[3*k : 3*k+3] = x_new
            # update delta x error 
            error = error + math.sqrt(smoother.dXpo[k,0]**2 + smoother.dXpo[k,1]**2)
            an_error = an_error + math.sqrt(smoother.dXpo[k,2]**2)

        delta_p = error / (K+1)
        delta_an = an_error / (K+1)
        print("pos error: {0}, angle error: {1}".format(delta_p, delta_an))

    # ------- End GN -------- #
    # vector to matrix
    x_op_v = x_op.reshape(-1,3)
    # remove invalid ground truth and estimation 
    REMOVE = False
    if REMOVE:
        gt_idx = np.where(true_valid==1)[0]
        t = t[gt_idx,:]
        vicon_gt = vicon_gt[gt_idx,:]
        x_op_v = x_op_v[gt_idx,:]
        Ppo = smoother.Ppo[gt_idx,:,:]
        K = vicon_gt.shape[0]

    # compute error
    x_error = x_op_v[:,0] - vicon_gt[:,0]
    y_error = x_op_v[:,1] - vicon_gt[:,1]
    th_error = np.zeros(K)
    sigma_x = np.zeros([K,1])
    sigma_y = np.zeros([K,1])
    sigma_th = np.zeros([K,1])
    for k in range(K):
        th_error[k] = x_op_v[k,2] - vicon_gt[k,2] 
        th_error[k] = wrapToPi(th_error[k])
        # extract the covariance matrix
        sigma_x[k,0] = np.sqrt(smoother.Ppo[k,0,0])
        sigma_y[k,0] = np.sqrt(smoother.Ppo[k,1,1])
        sigma_th[k,0] = np.sqrt(smoother.Ppo[k,2,2])

    # compute RMSE
    rms_x = math.sqrt(mean_squared_error(vicon_gt[:,0], x_op_v[:,0]))
    rms_y = math.sqrt(mean_squared_error(vicon_gt[:,1], x_op_v[:,1]))
    rms_th = 0
    for k in range(K):
        rms_th +=  th_error[k]**2     # square error

    rms_th = math.sqrt(rms_th/(K+1))  # root-mean-squared error 
    print('The RMS error for position x is %f [m]' % rms_x)
    print('The RMS error for position y is %f [m]' % rms_y)
    print('The RMS error for angle theta is %f [rad]' % rms_th)

    ## visual ground truth
    fig1 = plt.figure(facecolor="white")
    plt.plot(vicon_gt[:,0], vicon_gt[:,1], color='red')
    plt.plot(x_op_v[:,0], x_op_v[:,1], color = 'royalblue')

    fig2 = plt.figure(facecolor="white")
    ax = fig2.add_subplot(111)
    ax.plot(t, x_error, color='royalblue',linewidth=2.0, alpha=1.0)
    ax.plot(t, -3*sigma_x[:,0], '--', color='red')
    ax.plot(t,  3*sigma_x[:,0], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.ylim(-0.3,0.3)
    plt.title('error in x')

    fig3 = plt.figure(facecolor="white")
    bx = fig3.add_subplot(111)
    bx.plot(t, y_error, color='royalblue',linewidth=2.0, alpha=1.0)
    bx.plot(t, -3*sigma_y[:,0], '--', color='red')
    bx.plot(t, 3*sigma_y[:,0], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.ylim(-0.3,0.3)
    plt.title('error in y')

    fig4 = plt.figure(facecolor="white")
    cx = fig4.add_subplot(111)
    cx.plot(t, th_error, color='royalblue',linewidth=2.0, alpha=1.0)
    cx.plot(t, -3*sigma_th[:,0], '--', color='red')
    cx.plot(t,  3*sigma_th[:,0], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.ylim(-0.3,0.3)
    plt.title('error in theta')

    plt.show()