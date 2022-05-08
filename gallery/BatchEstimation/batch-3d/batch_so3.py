'''3D batch estimation with SO3 representation'''
import numpy as np
import os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy import linalg
from robot_model_3d import Robot3D
from rts_smoother_3d import RTS_Smoother_3D

from so3_util import axisAngle_to_Rot, getTrans, getTrans_in, skew, circle
np.set_printoptions(precision=4)

FONTSIZE = 18;   TICK_SIZE = 16
matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

# LaTeX font
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Helvetica"]})


def getGroundtruth(K, theta_vk_i, r_i_vk_i):
    C_gt = np.zeros((K, 3, 3))
    r_gt = np.zeros((K, 3))
    for k in range(K):
        C_gt[k,:,:] = axisAngle_to_Rot(theta_vk_i[:,k])
        r_gt[k,:] = r_i_vk_i[:,k]

    return C_gt, r_gt

def visual_results(C_gt, r_gt, C_op, r_op, t, smoother, K):

    Er     = np.zeros((3,K))
    Eth    = np.zeros((3,K))
    var_r  = np.zeros((3,K))
    var_th = np.zeros((3,K))
    var_r_f  = np.zeros((3,K))
    var_th_f = np.zeros((3,K))


    for k in range(K):
        Er[:,k] = np.squeeze(r_op[k,:] - r_gt[k,:])
        delta_theta_skew = np.eye(3) - (C_op[k,:,:].dot(C_gt[k,:,:].T))
        Eth[0,k] = delta_theta_skew[2,1]; 
        Eth[1,k] = delta_theta_skew[0,2]; 
        Eth[2,k] = delta_theta_skew[1,0]; 

        var_r[0,k] = math.sqrt(smoother.Ppo[k,0,0])
        var_r[1,k] = math.sqrt(smoother.Ppo[k,1,1])
        var_r[2,k] = math.sqrt(smoother.Ppo[k,2,2])
        var_th[0,k] = math.sqrt(smoother.Ppo[k,3,3])
        var_th[1,k] = math.sqrt(smoother.Ppo[k,4,4])
        var_th[2,k] = math.sqrt(smoother.Ppo[k,5,5])

        var_r_f[0,k] = math.sqrt(smoother.Ppo_f[k,0,0])
        var_r_f[1,k] = math.sqrt(smoother.Ppo_f[k,1,1])
        var_r_f[2,k] = math.sqrt(smoother.Ppo_f[k,2,2])
        var_th_f[0,k] = math.sqrt(smoother.Ppo_f[k,3,3])
        var_th_f[1,k] = math.sqrt(smoother.Ppo_f[k,4,4])
        var_th_f[2,k] = math.sqrt(smoother.Ppo_f[k,5,5])

    fig = plt.figure(facecolor="white")
    ax_t = fig.add_subplot(111, projection = '3d')
    # make the panes transparent
    ax_t.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # change the color of the grid lines 
    ax_t.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    
    ax_t.plot(r_gt[:,0],r_gt[:,1],r_gt[:,2],color='steelblue',linewidth=1.9, alpha=0.9, label = 'GT Traj.')
    ax_t.plot(r_op[:,0],r_op[:,1],r_op[:,2],color='green',linewidth=1.9, alpha=0.9, label = 'est. Traj.')

    print("Error: x: mu:[%.3f] std:[%.3f] y: mu:[%.3f] std:[%.3f] z: mu:[%.3f] std:[%.3f]"%(
        np.mean(r_gt[:,0] - r_op[:,0]), np.std(r_gt[:,0] - r_op[:,0]),
        np.mean(r_gt[:,1] - r_op[:,1]), np.std(r_gt[:,1] - r_op[:,1]),
        np.mean(r_gt[:,2] - r_op[:,2]), np.std(r_gt[:,2] - r_op[:,2])))

    # use LaTeX fonts in the plot
    ax_t.set_xlabel(r'X [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_ylabel(r'Y [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_zlabel(r'Z [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.legend(loc='best', fontsize=FONTSIZE)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis


    fig1 = plt.figure(facecolor="white")
    ax1 = fig1.add_subplot(111)
    ax1.plot(t,Er[0,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax1.plot(t,3*var_r[0,:],color='red',linewidth=1.9, alpha=0.9)
    ax1.plot(t,-3*var_r[0,:],color='red',linewidth=1.9, alpha=0.9)
    # ax1.plot(t,3*var_r_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    # ax1.plot(t,-3*var_r_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in x")

    fig2 = plt.figure(facecolor="white")
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,Er[1,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax2.plot(t,3*var_r[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax2.plot(t,-3*var_r[1,:],color='red',linewidth=1.9, alpha=0.9)
    # ax2.plot(t,3*var_r_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    # ax2.plot(t,-3*var_r_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in y")

    fig3 = plt.figure(facecolor="white")
    ax3 = fig3.add_subplot(111)
    ax3.plot(t,Er[2,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax3.plot(t,3*var_r[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax3.plot(t,-3*var_r[2,:],color='red',linewidth=1.9, alpha=0.9)
    # ax3.plot(t,3*var_r_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    # ax3.plot(t,-3*var_r_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in z")


    fig4 = plt.figure(facecolor="white")
    ax4 = fig4.add_subplot(111)
    ax4.plot(t,Eth[0,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax4.plot(t,3*var_th[0,:],color='red',linewidth=1.9, alpha=0.9)
    ax4.plot(t,-3*var_th[0,:],color='red',linewidth=1.9, alpha=0.9)
    # ax4.plot(t,3*var_th_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    # ax4.plot(t,-3*var_th_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 1")

    fig5 = plt.figure(facecolor="white")
    ax5 = fig5.add_subplot(111)
    ax5.plot(t,Eth[1,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax5.plot(t,3*var_th[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax5.plot(t,-3*var_th[1,:],color='red',linewidth=1.9, alpha=0.9)
    # ax5.plot(t,3*var_th_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    # ax5.plot(t,-3*var_th_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 2")

    fig6 = plt.figure(facecolor="white")
    ax6 = fig6.add_subplot(111)
    ax6.plot(t,Eth[2,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax6.plot(t,3*var_th[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax6.plot(t,-3*var_th[2,:],color='red',linewidth=1.9, alpha=0.9)
    # ax6.plot(t,3*var_th_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    # ax6.plot(t,-3*var_th_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 3")

    plt.show()

# help function
def update_op(smoother, T_op, T_final, dr_step, dtheta_step, K):
    for k in range(K):
        perturb = smoother.pert_po[k,:]
        rho = perturb[0:3].reshape(-1,1)
        phi = perturb[3:6]
        phi_skew = skew(phi)
        zeta = np.block([
            [phi_skew, rho],
            [0,  0,  0,  0]
            ])
        Psi = linalg.expm(zeta)
        T_final[k,:,:] = Psi.dot(T_op[k,:,:])

        T_k = T_final[k,:,:]
        # update C_op, r_op
        C_op[k,:,:] = T_k[0:3,0:3]
        r_op[k,:]   = -1.0 * np.squeeze((C_op[k,:,:].T).dot(T_k[0:3,3].reshape(-1,1)))

        T_prev = T_op[k,:,:]
        C_prev[k,:,:] = T_prev[0:3, 0:3]
        r_prev[k,:]   = -1.0 * np.squeeze((C_prev[k,:,:].T).dot(T_prev[0:3,3].reshape(-1,1)))

        dr_step[k,:]  = np.squeeze(r_op[k,:].reshape(-1,1) - r_prev[k,:].reshape(-1,1))

        delta_theta_skew = np.eye(3) - C_op[k,:,:].dot(C_prev[k,:,:].T)

        dtheta_step[k,0] = delta_theta_skew[2,1]
        dtheta_step[k,1] = delta_theta_skew[0,2]
        dtheta_step[k,2] = delta_theta_skew[1,0]

    # update init. state
    X0 = np.block([dr_step[0,:], dtheta_step[0,:]]).reshape(-1,1)
    P0 = smoother.Ppo[0,:,:]
    # update T_op
    T_op = np.copy(T_final)              # need to use np.copy()
    return X0, P0, T_op, dr_step

if __name__ == "__main__":
    # load data
    os.chdir("/home/wenda/BayesFilterGallery/dataset/textbook-data")   
    curr = os.getcwd()
    # load ground truth data
    theta_vk_i = loadmat(curr+'/dataset3.mat')['theta_vk_i'];    r_i_vk_i = loadmat(curr+'/dataset3.mat')['r_i_vk_i']
    t = loadmat(curr+'/dataset3.mat')['t']; 
    # imu input
    w_vk_vk_i = loadmat(curr+'/dataset3.mat')['w_vk_vk_i'];      w_var = loadmat(curr+'/dataset3.mat')['w_var']
    v_vk_vk_i = loadmat(curr+'/dataset3.mat')['v_vk_vk_i'];      v_var = loadmat(curr+'/dataset3.mat')['v_var']

    # measurements
    rho_i_pj_i = loadmat(curr+'/dataset3.mat')['rho_i_pj_i'];    # landmark positions
    y_k_j = loadmat(curr+'/dataset3.mat')['y_k_j'];              y_var = loadmat(curr+'/dataset3.mat')['y_var']
    
    # calibration param
    C_c_v = loadmat(curr+'/dataset3.mat')['C_c_v'];              rho_v_c_v = loadmat(curr+'/dataset3.mat')['rho_v_c_v']
    fu = loadmat(curr+'/dataset3.mat')['fu'];                    fv = loadmat(curr+'/dataset3.mat')['fv']
    cu = loadmat(curr+'/dataset3.mat')['cu'];                    cv = loadmat(curr+'/dataset3.mat')['cv']
    b  = loadmat(curr+'/dataset3.mat')['b']; 

    # window
    w1 = 1214;        w2 = 1714  # Matlab: 1215 ~ 1714 --> Python: 1214 ~ 1713, set w2 = 1714 since a[w1:w2] doesn't get a[w2]
    # complete data   
    # w1 = 1214;           w2 = 1220

    # Data Process
    t = t[0, w1 : w2];                   t = t - t[0]          # reset timestamp
    theta_vk_i = theta_vk_i[:, w1:w2];   r_i_vk_i = r_i_vk_i[:,w1:w2]
    # inputs
    v_vk_vk_i = v_vk_vk_i[:, w1:w2];     w_vk_vk_i = w_vk_vk_i[:, w1:w2];

    # ----- debug ----- #
    DEBUG = False
    if DEBUG:
        fig1 = plt.figure(facecolor="white")
        ax1 = fig1.add_subplot(311)
        ax1.plot(t, v_vk_vk_i[0,:])
        bx1 = fig1.add_subplot(312)
        bx1.plot(t, v_vk_vk_i[1,:])
        cx1 = fig1.add_subplot(313)
        cx1.plot(t, v_vk_vk_i[2,:])
        plt.show()

    # measurements
    y_k_j = y_k_j[:,w1:w2,:]
    # total number of timestamp
    K = w2 - w1
    C_gt, r_gt = getGroundtruth(K, theta_vk_i, r_i_vk_i)

    # compute input and meas. variance
    R = np.diag(np.squeeze(y_var))              # measurement covariance
    var_imu = np.concatenate((v_var,w_var), axis=0)
    Q = np.diag(np.squeeze(var_imu))              # imu noise, Qi = Q * dt**2 
    # create robot
    robot = Robot3D(Q, R, C_c_v, rho_v_c_v, fu, fv, cu, cv, b, rho_i_pj_i)

    # init. state and covariance
    X0 = np.zeros((6,1))
    P0 = np.eye(6) * 0.0001
    # create RTS-smoother
    smoother = RTS_Smoother_3D(robot, K)

    # compute the operating point initially with dead-reckoning
    C_op = np.zeros((K, 3, 3))    # rotation matrix
    r_op = np.zeros((K, 3))       # translation vector
    T_op = np.zeros((K, 4, 4))    # transformation matrix
    # init
    C_op[0,:,:] = C_gt[0,:,:];   
    r_op[0,:]   = r_gt[0,:];  
    T_op[0,:,:] = getTrans(C_op[0,:,:], r_op[0,:])
    # dead reckoning: T_op
    for k in range(1, K):
        # input v(k-1)
        v_k = v_vk_vk_i[:,k-1];    w_k = w_vk_vk_i[:,k-1]
        # dt
        dt = t[k] - t[k-1]
        C_op[k,:,:], r_op[k,:] = robot.motion_model(C_op[k-1,:,:], r_op[k-1,:], v_k, w_k, dt)
        # compute the operating point for transformation matrix T_op
        T_op[k,:,:] = getTrans(C_op[k,:,:], r_op[k,:])


    # Gauss-Newton 
    # in each iteration, 
    # (1) do one batch estimation for dx 
    # (2) update the operating point x_op
    # (3) check the convergence
    iter = 0;       max_iter = 10; 
    delta_p = 1;    T_final = np.zeros((K, 4, 4))

    C_prev = np.zeros((K,3,3))
    r_prev = np.zeros((K,3))
    dr_step = np.zeros((K,3))
    dtheta_step = np.zeros((K,3))

    label = 1

    while (iter < max_iter) and (label != 0):
        iter = iter + 1 
        print("\nIteration: #{0}\n".format(iter))
        # full batch estimation using RTS-smoother
        # RTS forward
        smoother.forward(X0, P0, C_op, r_op, v_vk_vk_i, w_vk_vk_i, y_k_j, t)

        # smoother.forward_old(X0, P0, C_op, r_op, v_vk_vk_i, w_vk_vk_i, y_k_j, t)    # original forward pass

        # RTS backward
        smoother.backward()

        # update operating point
        X0, P0, T_op, dr_step = update_op(smoother, T_op, T_final, dr_step, dtheta_step, K)

        label = np.sum(dr_step > 0.01)
        print(label)
        if label == 0:
            print("Converged!\n")

    # visualize
    visual_results(C_gt, r_gt, C_op, r_op, t, smoother, K)





