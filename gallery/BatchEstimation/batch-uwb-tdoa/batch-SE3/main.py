'''
Batch estimation for UWB TDOA using Lie group SE(3)
The input is simulated body velocity: 3 in translation velocity, 3 in angular velocity
'''
import rosbag, os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import linalg, stats
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error

from drone_model import DroneModel
from rts_smoother_3d import RTS_Smoother_3D
from so3_util import getTrans, skew

# help function in EKF
from eskf_class_la import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj

# fix random seed
np.random.seed(7)

FONTSIZE = 18;   TICK_SIZE = 16
matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

def isin(t_np,t_k):
    # check if t_k is in the numpy array t_np. 
    # If t_k is in t_np, return the index and bool = Ture.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        b = True
        return res[0][0], b
    b = False
    return 0, b

def eskf_est(t, imu, uwb, anchor_position, t_gt_pose, gt_pos):
    K = t.shape[0]
    # Initial estimate for the state vector
    X0 = np.zeros((6,1))        
    X0[0] = 1.5;  X0[1] = 0.0;  X0[2] = 1.5
    q0 = Quaternion([1,0,0,0])  # initial quaternion
    # Initial posterior covariance
    std_xy0 = 0.1;       std_z0 = 0.1;      std_vel0 = 0.1
    std_rp0 = 0.1;       std_yaw0 = 0.1
    P0 = np.diag([std_xy0**2,  std_xy0**2,  std_z0**2,\
                  std_vel0**2, std_vel0**2, std_vel0**2,\
                  std_rp0**2,  std_rp0**2,  std_yaw0**2 ])
    # create the object of ESKF
    eskf = ESKF(X0, q0, P0, K)

    print('timestep: %f' % K)
    print('\nStart state estimation')
    # ESKF with sync. data (imu, uwb)
    for k in range(1,K):               # k = 1 ~ K-1
        # compute dt
        dt = t[k]-t[k-1]
        # ESKF Prediction
        eskf.predict(imu[k,:], dt, 1, k)
        # ESKF Correction
        eskf.UWB_correct(uwb[k,:], anchor_position, k)

    print('Finish the state estimation\n')

    ## compute the error    
    # interpolate Vicon measurements
    f_x = interpolate.splrep(t_gt_pose, gt_pos[:,0], s = 0.5)
    f_y = interpolate.splrep(t_gt_pose, gt_pos[:,1], s = 0.5)
    f_z = interpolate.splrep(t_gt_pose, gt_pos[:,2], s = 0.5)
    x_interp = interpolate.splev(t, f_x, der = 0)
    y_interp = interpolate.splev(t, f_y, der = 0)
    z_interp = interpolate.splev(t, f_z, der = 0)

    x_error = eskf.Xpo[:,0] - np.squeeze(x_interp)
    y_error = eskf.Xpo[:,1] - np.squeeze(y_interp)
    z_error = eskf.Xpo[:,2] - np.squeeze(z_interp)

    pos_error = np.concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)

    rms_x = math.sqrt(mean_squared_error(x_interp, eskf.Xpo[:,0]))
    rms_y = math.sqrt(mean_squared_error(y_interp, eskf.Xpo[:,1]))
    rms_z = math.sqrt(mean_squared_error(z_interp, eskf.Xpo[:,2]))
    print('The RMS error for position x is %f [m]' % rms_x)
    print('The RMS error for position y is %f [m]' % rms_y)
    print('The RMS error for position z is %f [m]' % rms_z)

    RMS_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)          
    print('The overall RMS error of position estimation is %f [m]\n' % RMS_all)

    # visualization
    plot_pos(t, eskf.Xpo, t_gt_pose, gt_pos)
    plot_pos_err(t, pos_error, eskf.Ppo)
    plot_traj(gt_pos, eskf.Xpo, anchor_position)
    plt.show()

# visualize traj.
def visual_traj(gt_pos, r_op, An):
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
    
    ax_t.plot(gt_pos[:,0],gt_pos[:,1], gt_pos[:,2],color='red',linewidth=1.9, alpha=1.0, label = 'GT Traj.')
    ax_t.plot(r_op[:,0],r_op[:,1],r_op[:,2],color='blue',linewidth=1.9, alpha=0.9, label = 'est. Traj.')
    ax_t.scatter(An[:,0], An[:,1], An[:,2], marker='o',color='red', label = 'Anchor position')
    ax_t.set_xlabel(r'X [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_ylabel(r'Y [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_zlabel(r'Z [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.legend(loc='best', fontsize=FONTSIZE)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis
    plt.show()

# need to synchronize so as to compute errors
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
    ax1.plot(t,3*var_r_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    ax1.plot(t,-3*var_r_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in x")

    fig2 = plt.figure(facecolor="white")
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,Er[1,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax2.plot(t,3*var_r[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax2.plot(t,-3*var_r[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax2.plot(t,3*var_r_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    ax2.plot(t,-3*var_r_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in y")

    fig3 = plt.figure(facecolor="white")
    ax3 = fig3.add_subplot(111)
    ax3.plot(t,Er[2,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax3.plot(t,3*var_r[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax3.plot(t,-3*var_r[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax3.plot(t,3*var_r_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    ax3.plot(t,-3*var_r_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in z")


    fig4 = plt.figure(facecolor="white")
    ax4 = fig4.add_subplot(111)
    ax4.plot(t,Eth[0,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax4.plot(t,3*var_th[0,:],color='red',linewidth=1.9, alpha=0.9)
    ax4.plot(t,-3*var_th[0,:],color='red',linewidth=1.9, alpha=0.9)
    ax4.plot(t,3*var_th_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    ax4.plot(t,-3*var_th_f[0,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 1")

    fig5 = plt.figure(facecolor="white")
    ax5 = fig5.add_subplot(111)
    ax5.plot(t,Eth[1,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax5.plot(t,3*var_th[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax5.plot(t,-3*var_th[1,:],color='red',linewidth=1.9, alpha=0.9)
    ax5.plot(t,3*var_th_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    ax5.plot(t,-3*var_th_f[1,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 2")

    fig6 = plt.figure(facecolor="white")
    ax6 = fig6.add_subplot(111)
    ax6.plot(t,Eth[2,:],color='steelblue',linewidth=1.9, alpha=0.9)
    ax6.plot(t,3*var_th[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax6.plot(t,-3*var_th[2,:],color='red',linewidth=1.9, alpha=0.9)
    ax6.plot(t,3*var_th_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    ax6.plot(t,-3*var_th_f[2,:],color='green',linewidth=1.9, alpha=0.9)
    plt.title("error in theta 3")

    plt.show()

'''help function'''
def zeta(phi):
    phi_norm = np.linalg.norm(phi)
    if phi_norm == 0:
        dq = np.array([1, 0, 0, 0])
    else:
        dq_xyz = (phi*(math.sin(0.5*phi_norm)))/phi_norm
        dq = np.array([math.cos(0.5*phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
    return dq

'''update operating point'''
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
    # address of the current script
    cwd = os.path.dirname(__file__)
    # load data
    data = np.load(os.path.join(cwd, "../data_npz/old/sim_uwb_batch2.npz"))
    t = data["t"];         imu = data["imu_syn"];      uwb = data["uwb"]
    t_gt = data["t_gt"];   gt_pos = data["gt_pos"];    gt_quat = data["gt_quat"]
    An = data["An"]; 

    # ground truth odometry as input     
    odom = data["odom"] 

    # ----------- apply ESKF with IMU as inputs ----------- #
    # eskf_est(t, imu, uwb, An, t_gt, gt_pos)

    VIS_DR = False

    K = t.shape[0]
    # we do not use imu meas. as inputs in this code
    acc = imu[0:K,0:3].T;                           # linear acceleration, shape(3,K), unit m/s^2
    gyro = imu[0:K,3:].T;                           # rotational velocity, shape(3,K), unit rad/s
    # gravity
    g = np.array([0.0, 0.0, -9.8]).reshape(-1,1)    # gravity in inertial frame
    # ---------- create quadrotor model ------------ #
    # input noise covariance matrix
    Q = np.diag([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])  
    R = 0.05**2 # UWB meas. variance
    # --- external calibration --- #
    # rotation from vehicle frame to UWB frame
    C_u_v = np.eye(3)      # uwb don't have orientation
    # translation vector from vehicle frame to UWB frame
    rho_v_u_v =  np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1) 
    drone = DroneModel(Q, R, C_u_v, rho_v_u_v, An)

    # init. state and covariance
    X0 = np.zeros((6,1))
    P0 = np.eye(6) * 0.0001
    # ----------- create RTS-smoother ----------- #
    smoother = RTS_Smoother_3D(drone, K)

    # compute the operating point initially with dead-reckoning
    C_op = np.zeros((K, 3, 3))              # rotation matrix
    r_op = np.zeros((K, 3))                 # translation vector
    T_op = np.zeros((K, 4, 4))              # transformation matrix
    # init
    C_op[0,:,:] = np.eye(3)                 # init. rotation 
    r_op[0,:] = np.array([1.5,0.0,1.5])     # init. translation  
    T_op[0,:,:] = getTrans(C_op[0,:,:], r_op[0,:])
    # init input
    v_k = odom[:, 0:3]
    w_k = odom[:, 3:6]
    # dead reckoning: T_op
    for k in range(1,K):
        # using gt odomety, we have very good dead reckoning. (consider to add noise on the gt inputs)
        # we add noise to the ground truth input (translation velocity, angular velocity)
        Norm = stats.norm(0, 0.2)

        v_k[k,:] = odom[k,0:3] + np.squeeze(np.array([Norm.rvs(), Norm.rvs(), Norm.rvs()])) 
        w_k[k,:] = odom[k,3:6] + np.squeeze(np.array([Norm.rvs(), Norm.rvs(), Norm.rvs()]))

        # compute dt
        dt = t[k] - t[k-1]
        C_op[k,:,:], r_op[k,:] = drone.motion_model(C_op[k-1,:,:], r_op[k-1,:], v_k[k], w_k[k], dt)
        # compute the operating point for transformation matrix T_op
        T_op[k,:,:] = getTrans(C_op[k,:,:], r_op[k,:])

    if VIS_DR:
        visual_traj(gt_pos, r_op)

    # ---------- Gauss-Newton ---------- #
    iter = 0;       max_iter = 10; 
    delta_p = 1;    T_final = np.zeros((K, 4, 4))

    C_prev = np.zeros((K,3,3))
    r_prev = np.zeros((K,3))
    dr_step = np.zeros((K,3))
    dtheta_step = np.zeros((K,3))
    # convergence label
    label = 1
    # uwb meas.
    y_k = uwb 
    while (iter < max_iter) and (label != 0):
        iter = iter + 1
        print("\nIteration: #{0}\n".format(iter))
        # full batch estimation using RTS-smoother
        # RTS forward 
        smoother.forward(X0, P0, C_op, r_op, v_k, w_k, y_k, t)
        # RTS backward
        smoother.backward()

        # update operating point
        X0, P0, T_op, dr_step = update_op(smoother, T_op, T_final, dr_step, dtheta_step, K)
        label = np.sum(dr_step > 0.005)
        print(label)
        if label == 0:
            print("Converged!\n")

    # interpolate Vicon measurements
    f_x = interpolate.splrep(t_gt, gt_pos[:,0], s = 0.5)
    f_y = interpolate.splrep(t_gt, gt_pos[:,1], s = 0.5)
    f_z = interpolate.splrep(t_gt, gt_pos[:,2], s = 0.5)
    x_interp = interpolate.splev(t, f_x, der = 0)
    y_interp = interpolate.splev(t, f_y, der = 0)
    z_interp = interpolate.splev(t, f_z, der = 0)
    x_error = r_op[:,0] - np.squeeze(x_interp)
    y_error = r_op[:,1] - np.squeeze(y_interp)
    z_error = r_op[:,2] - np.squeeze(z_interp)

    pos_error = np.concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)

    rms_x = math.sqrt(mean_squared_error(x_interp, r_op[:,0]))
    rms_y = math.sqrt(mean_squared_error(y_interp, r_op[:,1]))
    rms_z = math.sqrt(mean_squared_error(z_interp, r_op[:,2]))
    print('The RMS error for position x is %f [m]' % rms_x)
    print('The RMS error for position y is %f [m]' % rms_y)
    print('The RMS error for position z is %f [m]' % rms_z)

    # visual batch estimation results
    visual_traj(gt_pos, r_op, drone.An)
    # visual batch estimation results
    # compute C_gt and r_gt
    # need to synchronize before compute the error
    # visual_results(C_gt, r_gt, C_op, r_op, t, smoother, K)

