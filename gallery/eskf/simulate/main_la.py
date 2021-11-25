'''
main file for eskf estimation for simulation (with lever-arm)
no UWB bias estimation
'''
#!/usr/bin/env python3
import argparse
import os, sys
import rosbag
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error

from eskf_class_la import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj

'''help function for timestamp'''
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

if __name__ == "__main__":    
    # anchor positions (simulated)
    anchor_position = np.array([[ 4.0,  4.3,  3.0],
                                [-4.0,  4.3,  3.0],
                                [ 4.0, -4.3,  3.0],
                                [-4.0, -4.3,  3.0],
                                [ 4.0,  4.3,  0.1],
                                [-4.0,  4.3,  0.1],
                                [ 4.0, -4.3,  0.1],
                                [-4.0, -4.3,  0.1]])
    # access rosbag file
    ros_bag = "simData_lever_arm.bag"    # with lever arm
    bag = rosbag.Bag(ros_bag)
    # -------------------- start extract the rosbag ------------------------ #
    gt_pos = []; gt_quat=[]; t_gt_pose = []
    imu = []; t_imu = []; uwb = []; t_uwb = []

    for topic, msg, t in bag.read_messages(["/firefly/ground_truth/pose", "/firefly/imu", "/uwb_tdoa"]):
        if topic == '/firefly/ground_truth/pose':
            gt_pos.append([msg.position.x,  msg.position.y,  msg.position.z])
            gt_quat.append([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
            t_gt_pose.append(t.secs + t.nsecs * 1e-9)
        if topic == '/firefly/imu':
            # imu unit: m/s^2 in acc, rad/sec in gyro
            imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                        msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)       
        if topic == '/uwb_tdoa':
            uwb.append([msg.id_A, msg.id_B, msg.dist_diff])
            t_uwb.append(msg.stamp.secs + msg.stamp.nsecs * 1e-9)

    min_t = min(t_imu + t_uwb)
    # get the vicon information from min_t
    t_gt_pose = np.array(t_gt_pose); idx = np.argwhere(t_gt_pose > min_t); t_gt_pose = t_gt_pose[idx]
    gt_pos = np.array(gt_pos)[idx,:];    gt_pos = np.squeeze(gt_pos)
    gt_quat = np.array(gt_quat)[idx,:];  gt_quat = np.squeeze(gt_quat)

    # sensor
    t_uwb = np.array(t_uwb);  t_imu = np.array(t_imu)
    uwb = np.array(uwb);      imu = np.array(imu)
    # reset ROS time base
    t_gt_pose = (t_gt_pose - min_t).reshape(-1,1)
    t_imu = (t_imu - min_t).reshape(-1,1)
    t_uwb = (t_uwb - min_t).reshape(-1,1)

    # ----------------------- INITIALIZATION OF EKF -------------------------#
    # Create a compound vector t with a sorted merge of all the sensor time bases
    time = np.sort(np.concatenate((t_imu, t_uwb)))
    t = np.unique(time)
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
    for k in range(1,K):               # k = 1 ~ K-1
        # Find what measurements are available at the current time (help function: isin() )
        imu_k,  imu_check  = isin(t_imu,   t[k-1])
        uwb_k,  uwb_check  = isin(t_uwb,   t[k-1])
        dt = t[k]-t[k-1]

        # ESKF Prediction
        eskf.predict(imu[imu_k,:], dt, imu_check, k)
        # ESKF Correction
        if uwb_check:            # if we have UWB measurement
            eskf.UWB_correct(uwb[uwb_k,:], anchor_position, k)

    print('Finish the state estimation\n')

    ## compute the error    
    # interpolate Vicon measurements
    f_x = interpolate.splrep(t_gt_pose, gt_pos[:,0], s = 0.5)
    f_y = interpolate.splrep(t_gt_pose, gt_pos[:,1], s = 0.5)
    f_z = interpolate.splrep(t_gt_pose, gt_pos[:,2], s = 0.5)
    x_interp = interpolate.splev(t, f_x, der = 0)
    y_interp = interpolate.splev(t, f_y, der = 0)
    z_interp = interpolate.splev(t, f_z, der = 0)

    x_error = eskf.Xpo[:,0] - x_interp
    y_error = eskf.Xpo[:,1] - y_interp
    z_error = eskf.Xpo[:,2] - z_interp

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