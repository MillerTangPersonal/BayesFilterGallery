'''
Batch estimation for UWB TDOA 
IMU and UWB
'''
from pickle import TRUE
import rosbag, os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import linalg, stats
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error

from robot_model import DroneModel
from rts_smoother_cf import RTS_Smoother         
# help function in EKF
from eskf_la_test import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj
from rot_util import zeta

FONTSIZE = 18;   TICK_SIZE = 16
matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

'''timestamp check'''
def isin(t_np, t_k):
    # check if t_k is in the numpy array t_np. 
    # If t_k is in t_np, return the index and bool = Ture.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        b = True
        return res[0][0], b
    b = False
    return 0, b

'''test the data with eskf'''
def eskf_est(t, imu, uwb, anchor_position, t_gt_pose, gt_pos):
    K = t.shape[0]
    # Initial estimate for the state vector
    X0 = np.zeros((6,1))        
    X0[0] = 1.36;  X0[1] = 0.0;  X0[2] = 0.16
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

    ## visualization
    # plot_pos(t, eskf.Xpo, t_gt_pose, gt_pos)
    # plot_pos_err(t, pos_error, eskf.Ppo)
    # plot_traj(gt_pos, eskf.Xpo, anchor_position)
    # plt.show()
    return eskf.Xpo, eskf.q_list

'''update operating point'''
def update_op(smoother, X_op, X_final, dp_step, dv_step, dtheta_step, K):
    for k in range(K):
        perturb = smoother.pert_po[k,:]   # perturb (or error state) [dx,dy,dz, dvx,dvy,dvz, dtheta1,dtheta2,dtheta3]
        
        dp = perturb[0:3]
        dv = perturb[3:6]
        dtheta = perturb[6:]
        # update position
        X_final[k,0:3] = X_op[k,0:3] + dp
        # update velocity
        X_final[k,3:6] = X_op[k,3:6] + dv
        # update quaternion  
        # consider the direction here: q_op_k_inv * q_k_check = dq_k --> q_k_check = q_op_k * dq_k
        dqk = Quaternion(zeta(dtheta))   # convert incremental rotation vector to quaternion
        q_op = Quaternion([X_op[k,6], X_op[k,7], X_op[k,8], X_op[k,9]])
        q_update = q_op * dqk            # update the quaternion with incremental rotation
        X_final[k,6:] = np.array([q_update[0], q_update[1], q_update[2], q_update[3]])

        # save the d step
        dp_step[k,:] = dp
        dv_step[k,:] = dv
        dtheta_step[k,:] = dtheta

    # update init. state
    pert_x0 = np.block([dp_step[0,:], dv_step[0,:], dtheta_step[0,:]]).reshape(-1,1)
    P0= smoother.Ppo[0,:,:]
    # update X_op
    X_op = np.copy(X_final)                       # need to use np.copy()
    return pert_x0, P0, X_op, dp_step

''' visualize traj. '''
def visual_traj(gt_pos, X_final, An):
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
    ax_t.plot(X_final[:,0],X_final[:,1],X_final[:,2],color='blue',linewidth=1.9, alpha=0.9, label = 'est. Traj.')
    ax_t.scatter(An[:,0], An[:,1], An[:,2], marker='o',color='red', label = 'Anchor position')
    ax_t.set_xlabel(r'X [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_ylabel(r'Y [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.set_zlabel(r'Z [m]',fontsize=FONTSIZE, linespacing=30.0)
    ax_t.legend(loc='best', fontsize=FONTSIZE)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis
    plt.show()

'''visual x,y,z'''
def visual_xyz(t_gt, gt_pos, t, X_final):
    fig1 = plt.figure(facecolor="white")
    ax = fig1.add_subplot(311)
    ax.plot(t_gt, gt_pos[:,0], color = 'red',  label = "gt")
    ax.plot(t, X_final[:,0],   color = 'blue', label = "est")
    ax.set_ylabel(r'X [m]',fontsize = FONTSIZE) 
    ax.legend(loc='best')

    bx = fig1.add_subplot(312)
    bx.plot(t_gt, gt_pos[:,1], color = 'red')
    bx.plot(t, X_final[:,1],   color = 'blue')
    bx.set_ylabel(r'Y [m]',fontsize = FONTSIZE) 

    cx = fig1.add_subplot(313)
    cx.plot(t_gt, gt_pos[:,2], color = 'red')
    cx.plot(t, X_final[:,2],   color = 'blue')
    cx.set_ylabel(r'Y [m]',fontsize = FONTSIZE) 
    cx.set_xlabel(r'Time [s]',fontsize = FONTSIZE)

    plt.show()


if __name__ == "__main__":   
    # directory of the current script
    cwd = os.path.dirname(__file__)
    # load data
    # data = np.load(os.path.join(cwd, "data_npz/1130_simData.npz"))
    data = np.load(os.path.join(cwd, "dataset_npz/const1-trial1.npz"))

    t = data["t_sensor"];  imu = data["imu_syn"];      uwb = data["uwb"]
    t_gt = data["t_gt"];   gt_pos = data["gt_pos"];    gt_quat = data["gt_quat"]
    An = data["An"]; 

    # ----------- apply ESKF with IMU as inputs [debug] ----------- #
    DEBUG = True
    if DEBUG:
        X_po_ekf, q_po_ekf = eskf_est(t, imu, uwb, An, t_gt, gt_pos)
        X_op_ekf = np.block([X_po_ekf,q_po_ekf])     # shape: K x 10
    # ------------------------------------------------------------- #

    K = t.shape[0]
    # imu inputs
    acc = imu[0:K, 0:3]     # shape: K x 3
    gyro = imu[0:K, 3:]     # shape: K x 3

    # ------- create quadrotor model 
    # input noise 
    std_acc = 2.0;   std_gyro = 0.1;   std_uwb = 0.05
    # UWB meas. variance
    R = std_uwb * 2
    # --- external calibration
    # rotation from vehicle frame to UWB frame (for completeness)
    C_u_v = np.eye(3)
    # translation vector from vehicle frame to UWB frame
    rho_v_u_v =  np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1) 
    # create drone model
    drone = DroneModel(std_acc, std_gyro, R, C_u_v, rho_v_u_v, An)
    # init. error state and covariance
    pert_x0 = np.zeros((9,1))
    P0 = np.eye(9) * 0.0001
    # ----- create RTS smoother
    smoother = RTS_Smoother(drone, K)
    # compute the operating points initially with dead-reckoning
    X_op = np.zeros((K,10))     # [x,y,z,vx,vy,vz,qw,qx,qy,qz]
    # init
    X_op[0,:] = np.array([1.25, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    for k in range(1, K):
        X_op_k1 = X_op[k-1,:]
        # inputs
        acc_k = acc[k,:];  gyro_k = gyro[k, :]
        # compute dt
        dt = t[k] - t[k-1]
        X_op[k, :] = drone.motion_model(X_op_k1, acc_k, gyro_k, dt)


    # [debug] use the eskf estimate as the operating point 
    if DEBUG:
        X_op[0:K,:] = X_op_ekf[0:K,:]

    # ------ visualize dead-reckoning results ------ #
    # visual_traj(gt_pos, X_op, drone.An)
    visual_xyz(t_gt, gt_pos, t, X_op)

    # ----- Gauss-Newton
    iter = 0;       max_iter = 200; 
    delta_p = 1; 
    X_final = np.zeros((K, 10))     # final position, velocity and quaternion
    # convergence label
    label = 1 

    dp_step = np.zeros((K,3))      # delta position
    dv_step = np.zeros((K,3))      # delta velocity 
    dtheta_step = np.zeros((K,3))  # delta angles
    # input data: acc, gyro
    # meas. data: uwb
    while (iter < max_iter) and (label != 0):
        iter = iter + 1
        print("\nIteration: #{0}\n".format(iter))
        # aplly full batch estimation using RTS smoother
        # RTS forward
        smoother.forward(pert_x0, P0, X_op, acc, gyro, uwb, t)
        # RTS backward
        smoother.backward()

        # update operating point 
        pert_x0, P0, X_op, dp_step = update_op(smoother, X_op, X_final, dp_step, dv_step, dtheta_step, K) 
        label = np.sum(abs(dp_step) >0.005)
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
    x_error = X_final[:,0] - np.squeeze(x_interp)
    y_error = X_final[:,1] - np.squeeze(y_interp)
    z_error = X_final[:,2] - np.squeeze(z_interp)

    pos_error = np.concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)

    rms_x = math.sqrt(mean_squared_error(x_interp, X_final[:,0]))
    rms_y = math.sqrt(mean_squared_error(y_interp, X_final[:,1]))
    rms_z = math.sqrt(mean_squared_error(z_interp, X_final[:,2]))
    print('The RMS error for position x is %f [m]' % rms_x)
    print('The RMS error for position y is %f [m]' % rms_y)
    print('The RMS error for position z is %f [m]' % rms_z)

    RMS_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)          
    print('The overall RMS error of position estimation is %f [m]\n' % RMS_all)
    
    visual_traj(gt_pos, X_final, drone.An)
    visual_xyz(t_gt, gt_pos, t, X_final)