'''
Batch estimation for UWB TDOA using Lie group
'''
import rosbag
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, stats
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error

from drone_model import DroneModel
from rts_smoother_3d import RTS_Smoother_3D
from so3_util import axisAngle_to_Rot, getTrans, skew

# help function in EKF
from eskf_class_la import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj

# fix random seed
np.random.seed(7)

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

def visual_traj(gt_pos, r_op):
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
    
    ax_t.plot(gt_pos[:,0],gt_pos[:,1], gt_pos[:,2],color='steelblue',linewidth=1.9, alpha=0.9, label = 'GT Traj.')
    ax_t.plot(r_op[:,0],r_op[:,1],r_op[:,2],color='green',linewidth=1.9, alpha=0.9, label = 'est. Traj.')
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
    # load data
    data = np.load("data_npz/sim_uwb_batch2.npz")
    t = data["t"];         imu = data["imu_syn"];      uwb = data["uwb"]
    t_gt = data["t_gt"];   gt_pos = data["gt_pos"];    gt_quat = data["gt_quat"]
    An = data["An"]; 

    # ground truth odometry as input     
    odom = data["odom"] 

    # ----------- apply ESKF ----------- #
    # eskf_est(t, imu, uwb, An, t_gt, gt_pos)

    # ----- debug ----- #
    DEBUG = False

    K = t.shape[0]
    acc = imu[0:K,0:3].T;                           # linear acceleration, shape(3,K), unit m/s^2
    gyro = imu[0:K,3:].T;                           # rotational velocity, shape(3,K), unit rad/s
    g = np.array([0.0, 0.0, -9.8]).reshape(-1,1)      # gravity in inertial frame
    # --------- compute the input data ----------- #
    # IMU measurements: 
    # acc: m/s^2    acc. in x, y, and z 
    # gyro: rad/s   angular velocity 
    v_vk_vk_i = np.zeros((3, K))          # translation velocity of vehicle frame with respect to inertial frame, expressed in vehicle frame 
    q_list = np.zeros((K,4))              # quaternion list
    R_list = np.zeros((K,3,3))            # Rotation matrix list (from body frame to inertial frame) 
    q_list[0,:] = np.array([1.0, 0.0, 0.0, 0.0])
    R_list[0,:,:] = np.eye(3)
    # compute the v_vk_vk_i (imu dea reckoning drifts quickly)
    for idx in range(1,K):
        # compute dt
        dt = t[idx] - t[idx-1] 
        # rotation from inertial to body frame (to convert g to body frame)
        R_iv = R_list[idx-1,:,:]
        C_vi = R_iv.T
        # compute input v_k 
        v_vk_vk_i[:, idx] = v_vk_vk_i[:,idx-1] + (acc[:,idx-1] + np.squeeze(C_vi.dot(g))) * dt

        # quaternion prop. (dead reckoning) 
        # problem #1: gryo[2,:] and odom[:,5] are different. The DR drifts very quickly. 
        # dw   = gyro[:,idx-1] * dt     
        # the difference between gyro[:,idx-1] and odom[idx-1,3:6]) are [0.001, -0.0004, -0.0008]
        dw = odom[idx-1,3:6] * dt
        dqk  = Quaternion(zeta(dw))           # convert incremental rotation vector to quaternion
        dR = dqk.rotation_matrix
        R_list[idx,:,:] = R_iv.dot(dR)

    if DEBUG:
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(311)
        ax.plot(t[0:K], gyro[0,:])
        ax.plot(t[0:K], odom[:,3],'--')
        plt.ylim(-0.1, 0.1)
        bx = fig.add_subplot(312)
        bx.plot(t[0:K], gyro[1,:])
        bx.plot(t[0:K], odom[:,4],'--')
        plt.ylim(-0.1, 0.1)
        cx = fig.add_subplot(313)
        cx.plot(t[0:K], gyro[2,:])
        cx.plot(t[0:K], odom[:,5],'--')
        plt.ylim(-0.1, 0.1)

        fig1 = plt.figure(facecolor="white")
        ax1 = fig1.add_subplot(311)
        ax1.plot(t[0:K], v_vk_vk_i[0,:])
        ax1.plot(t[0:K], odom[:,0],'--')
        bx1 = fig1.add_subplot(312)
        bx1.plot(t[0:K], v_vk_vk_i[1,:])
        bx1.plot(t[0:K], odom[:,1],'--')
        cx1 = fig1.add_subplot(313)
        cx1.plot(t[0:K], v_vk_vk_i[2,:])
        cx1.plot(t[0:K], odom[:,2],'--')
        plt.show()

    # ---------- create quadrotor model ------------ #
    Q = np.diag([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])  
    R = 0.05 # UWB meas. variance
    # --- external calibration --- #
    # rotation from vehicel frame to UWB frame
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
    # r_op[0,:] = np.array([0.0,0.0,1.0])   # init. translation  (with level arm)
    r_op[0,:] = np.array([1.5,0.0,1.5])     # init. translation  (without level arm)
    T_op[0,:,:] = getTrans(C_op[0,:,:], r_op[0,:])
    # dead reckoning: T_op
    for k in range(1,K):
        # input v(k-1) (computed)
        # v_k = v_vk_vk_i[:,k-1];  w_k = odom[k-1,3:6]

        # the gt imu is also biased around 0.001
        # w_k = gyro[:, k-1]; 

        # using gt odomety, we have very good dead reckoning. (consider to add noise on the gt inputs)
        # we add noise to the ground truth input (translation velocity, angular velocity)
        Norm = stats.norm(0, 0.2)

        v_k = odom[k-1,0:3] + np.squeeze(np.array([Norm.rvs(), Norm.rvs(), Norm.rvs()])) 
        w_k = odom[k-1,3:6] + np.squeeze(np.array([Norm.rvs(), Norm.rvs(), Norm.rvs()]))

        # compute dt
        dt = t[k] - t[k-1]
        C_op[k,:,:], r_op[k,:] = drone.motion_model(C_op[k-1,:,:], r_op[k-1,:], v_k, w_k, dt)
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

        # RTS backward

        # update operating point
        X0, P0, T_op, dr_step = update_op(smoother, T_op, T_final, dr_step, dtheta_step, K)

        label = np.sum(dr_step > 0.01)
        print(label)
        if label == 0:
            print("Converged!\n")

    # check the dead reckoning
    visual_traj(gt_pos, r_op)