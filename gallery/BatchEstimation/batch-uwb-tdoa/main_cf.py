'''
Batch estimation for UWB TDOA 
IMU and UWB
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

# from robot_model import DroneModel

# from rts_smoother_3d import RTS_Smoother_3D

# help function in EKF
from eskf_class_la import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj


FONTSIZE = 18;   TICK_SIZE = 16
matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

'''timestamp check'''
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

'''test the data with eskf'''
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



if __name__ == "__main__":   
    # address of the current script
    cwd = os.path.dirname(__file__)
    # load data
    data = np.load(os.path.join(cwd, "data_npz/1130_simData.npz"))
    t = data["t_sensor"];  imu = data["imu_syn"];      uwb = data["uwb"]
    t_gt = data["t_gt"];   gt_pos = data["gt_pos"];    gt_quat = data["gt_quat"]
    An = data["An"]; 

    # ----------- apply ESKF with IMU as inputs ----------- #
    eskf_est(t, imu, uwb, An, t_gt, gt_pos)
