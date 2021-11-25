'''implement EKF for 2D robot'''
import numpy as np
import os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from scipy import linalg
# import RTS-smoother
from util import wrapToPi
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
    w1 = 0; w2 = 12609  # 12609
    t = t[w1 : w2];                   t = t - t[0,0]          #reset timestamp
    v = v[w1 : w2];                   om = om[w1 : w2]
    r_meas = r_meas[w1 : w2, :];      b_meas = b_meas[w1 : w2, :]
    vicon_gt = vicon_gt[w1 : w2,:]

    # total timestamp
    K = t.shape[0];        T = 0.1  # time duration, 10 Hz
    # initial position 
    X0 = vicon_gt[0,:]
    # initial covariance
    P0 = np.diag([0.0001, 0.0001, 0.0001])
    # input noise
    Q = np.diag([v_var[0,0], om_var[0,0]])
    # meas. noise
    R = np.diag([r_var[0,0], b_var[0,0]])

    # filter the measurements
    r_max = 1                                         
    for i in range(r_meas.shape[0]):
        for j in range(r_meas.shape[1]):
            if r_meas[i,j] > r_max:
                r_meas[i,j] = 0.0

    # ground robot
    robot = GroundRobot(Q, R, d, l ,T)

    Xpr = np.zeros((K, 3))
    Xpo = np.zeros((K, 3))
    Ppr = np.zeros((K, 3, 3))
    Ppo = np.zeros((K, 3, 3))      
    
    # init
    Xpr[0,:] = X0;      Ppr[0,:,:] = P0; 
    Xpo[0,:] = X0;      Ppo[0,:,:] = P0; 
    # EKF implementation
    for k in range(1, K):
        v_k = v[k];             om_k = om[k]
        r_k = r_meas[k,:];      b_k = b_meas[k,:]

        F_k1 = robot.compute_F(Xpo[k-1,:], v_k)
        Qv_k = robot.nv_prop(Xpo[k-1,:])
        # eq. 1
        Ppr[k,:,:] = F_k1.dot(Ppo[k-1,:,:]).dot(F_k1.T) + Qv_k
        # eq. 2
        Xpr[k,:] = np.squeeze(robot.motion_model(Xpo[k-1,:], v_k, om_k))

        # measurements
        l_k = np.nonzero(r_k);    l_k = np.asarray(l_k).reshape(-1,1)
        M = np.count_nonzero(r_k, axis=0)               # at timestamp k, we have M meas. in total

        # flag for measurement update
        update = True
        ey_k = np.empty(0);       Rk_y  = np.empty((0,0))
        if M: 
            G = np.empty((0,3))                            # empty G to contain all the Gs in timestep k
            for m in range(M): 
                l_xy = robot.landmark[l_k[m,0], :]         # the m-th landmark pos.
                r_l = r_k[l_k[m,0]]                        # [timestamp, id of landmark]
                b_l = b_k[l_k[m,0]]
                y = np.array([r_l, b_l]).reshape(-1,1)
                ey = y - robot.meas_model(Xpr[k,:], l_xy)  # y_k - g(x_check,0)
                # compute angle error, wrap to pi
                ey[1] = wrapToPi(ey[1])
                '''compute ey_k'''
                ey_k = np.append(ey_k, np.squeeze(ey))
                # --------- save the variance of meas. ------------ #
                Rk_y = linalg.block_diag(Rk_y, robot.Rm)
                # compute G
                G_i = robot.compute_G(Xpr[k,:], l_xy)
                G = np.concatenate((G, G_i), axis=0)
        else:
            # when no measurements, use Xpr as Xpo
            update = False

        if update:
            # eq. 3
            GM = G.dot(Ppr[k,:,:]).dot(G.T) + Rk_y 
            K_k = Ppr[k,:,:].dot(G.T).dot(linalg.inv(GM))
            # equ. 4
            Ppo[k,:,:] = (np.eye(3) - K_k.dot(G)).dot(Ppr[k,:,:])
            # equ. 5
            Xpo[k,:] = Xpr[k,:] + np.squeeze(K_k.dot(ey_k))
        else:
            Ppo[k,:,:] = Ppr[k,:,:]
            Xpo[k,:] =  Xpr[k,:]



    # compute error
    x_error = Xpo[:,0] - vicon_gt[:,0]
    y_error = Xpo[:,1] - vicon_gt[:,1]
    th_error = np.zeros(K)
    sigma_x = np.zeros([K,1])
    sigma_y = np.zeros([K,1])
    sigma_th = np.zeros([K,1])
    for k in range(K):
        th_error[k] = Xpo[k,2] - vicon_gt[k,2] 
        th_error[k] = wrapToPi(th_error[k])
        # extract the covariance matrix
        sigma_x[k,0] = np.sqrt(Ppo[k,0,0])
        sigma_y[k,0] = np.sqrt(Ppo[k,1,1])
        sigma_th[k,0] = np.sqrt(Ppo[k,2,2])

    # compute RMSE
    rms_x = math.sqrt(mean_squared_error(vicon_gt[:,0], Xpo[:,0]))
    rms_y = math.sqrt(mean_squared_error(vicon_gt[:,1], Xpo[:,1]))
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
    plt.plot(Xpo[:,0], Xpo[:,1], color = 'royalblue')

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


    fig5 = plt.figure(facecolor="white")
    ax1 = fig5.add_subplot(111)
    ax1.plot(t,  Xpo[:,0], color='royalblue',linewidth=2.0, alpha=1.0)
    ax1.plot(t, vicon_gt[:,0], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.title('error in x')

    fig6 = plt.figure(facecolor="white")
    bx1 = fig6.add_subplot(111)
    bx1.plot(t,  Xpo[:,1], color='royalblue',linewidth=2.0, alpha=1.0)
    bx1.plot(t, vicon_gt[:,1], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.title('error in y')

    fig7 = plt.figure(facecolor="white")
    cx1 = fig7.add_subplot(111)
    cx1.plot(t, Xpo[:,2], color='royalblue',linewidth=2.0, alpha=1.0)
    cx1.plot(t, vicon_gt[:,2], '--', color='red')
    plt.xlim(0.0, t[-1,0])
    plt.title('error in theta')

    plt.show()