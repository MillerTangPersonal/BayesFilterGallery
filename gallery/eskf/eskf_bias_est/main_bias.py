'''
main file for eskf estimation
used for the real data collected from CF_Bolt
Add UWB bias estimation 
'''
#!/usr/bin/env python3
import argparse
import os, sys
import rosbag
from cf_msgs.msg import Accel, Gyro, Flow, Tdoa, Tof 
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error

from eskf_class import ESKF
from plot_util import plot_pos, plot_pos_err, plot_traj, getTDOA_gt, visual_bias, get_uwbp

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

    # select anchor constellations
    survey_path = "../../../dataset/flight-dataset/survey-results/"
    anchor_npz = "anchor_const1.npz"
    anchor_survey = np.load(os.path.join(survey_path, anchor_npz))
    anchor_position = anchor_survey['an_pos']
    print("\nselecting " + str(anchor_npz) + "\n")

    # access rosbag file
    ros_bag_path = "../../../dataset/flight-dataset/rosbag-data/const1/"
    bag_file = "const1-log2.bag"
    bag = rosbag.Bag(os.path.join(ros_bag_path, bag_file))
    print("visualizing rosbag: " + str(bag_file) + "\n")
    # rosbag 1110 NLOS
    # bag = rosbag.Bag("/home/wenda/dsl__projects__uwbOML/data/rosbag/1110/1110_los_path.bag")

    # ----- init. bias std: 0.1, process noise = 1.0----- #
    # better: log1 {0.090896 [m]}  -> {0.104528 [m]} no bias est.
    #         log2 {0.105933 [m]}  -> {0.120910 [m]} no bias est.
    #         log6 {0.105345 [m]}  -> {0.119539 [m]} no bias est.

    # worse:  log3 {0.112 [m]}  -> {0.107 [m]} no bias est.
    #         log4 {0.110 [m]}  -> {0.109 [m]} no bias est.
    #         log5 {0.175 [m]}  -> {0.125 [m]} no bias est.        
    #         log5(ds) {0.1761 [m]} -> {0.141 [m]} no bias est.

    # ----- init. bias std: 0.1, process noise = 4.0----- #
    # better: 
    #         log4 {0.107409 [m]}    -> {0.109 [m]} no bias est.
    #         log5 (ds) {0.1385 [m]} -> {0.141 [m]} no bias est.
    # worse: 
    #         log2 {0.213346 [m]} ->  {0.120910 [m]} no bias est.
    #         log3 {0.17306 [m]}  ->  {0.107 [m]} no bias est.
    #         log5 {0.12956 [m]}  ->  {0.125 [m]} no bias est. 


    # -------------------- start extract the rosbag ------------------------ #
    pos_vicon=[];      t_vicon=[]; 
    uwb=[];            t_uwb=[]; 
    imu=[];            t_imu=[]; 
    for topic, msg, t in bag.read_messages(['/pose_data', '/tdoa_data', '/imu_data']):
        if topic == '/pose_data':
            pos_vicon.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                              msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                              msg.pose.pose.orientation.z, msg.pose.pose.orientation.w ])
            t_vicon.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        if topic == '/tdoa_data':
            uwb.append([msg.idA, msg.idB, msg.data])
            t_uwb.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        if topic == '/imu_data':
            imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y,  msg.linear_acceleration.z,\
                        msg.angular_velocity.x,    msg.angular_velocity.y,     msg.angular_velocity.z     ])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

    DOWN_SAMP = False
    # sensor
    if DOWN_SAMP:
        # down-sample the imu data
        # https://stackoverflow.com/questions/12433695/extract-elements-of-list-at-odd-positions
        t_imu = t_imu[0::2];           imu = imu[0::2]    # keep the even index 
        t_imu = t_imu[0::2];           imu = imu[0::2]    # downsample imu twice
        t_uwb = t_uwb[0::2];           uwb = uwb[0::2]

    # get the start time
    min_t = min(t_uwb + t_imu + t_vicon)

    t_imu = np.array(t_imu);       imu = np.array(imu);  
    t_uwb = np.array(t_uwb);       uwb = np.array(uwb);  

    # get the vicon information from min_t
    t_vicon = np.array(t_vicon);              
    idx = np.argwhere(t_vicon > min_t);     
    t_vicon = t_vicon[idx]; 
    pos_vicon = np.squeeze(np.array(pos_vicon)[idx,:])

    # reset ROS time base
    t_vicon = (t_vicon - min_t).reshape(-1,1)
    t_imu = (t_imu - min_t).reshape(-1,1)
    t_uwb = (t_uwb - min_t).reshape(-1,1)

    # ----------------------- INITIALIZATION OF EKF -------------------------#
    # Create a compound vector t with a sorted merge of all the sensor time bases
    time = np.sort(np.concatenate((t_imu, t_uwb)))
    t = np.unique(time)
    K = t.shape[0]


    # Initial estimate for the state vector
    X0 = np.zeros((14,1))        
    X0[0] = 1.25;  X0[1] = 0.0;  X0[2] = 0.07
        
    q0 = Quaternion([1,0,0,0])  # initial quaternion

    # Initial posterior covariance
    std_xy0 = 0.1;       std_z0 = 0.1;      std_vel0 = 0.1
    std_rp0 = 0.1;       std_yaw0 = 0.1
    # UWB bias std
    std_bias = 0.1

    P0 = np.diag([std_xy0**2,  std_xy0**2,  std_z0**2,\
                std_vel0**2, std_vel0**2, std_vel0**2,\
                std_rp0**2,  std_rp0**2,  std_yaw0**2,\
                std_bias**2, std_bias**2, std_bias**2, std_bias**2, std_bias**2, std_bias**2, std_bias**2, std_bias**2])


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
    f_x = interpolate.splrep(t_vicon, pos_vicon[:,0], s = 0)
    f_y = interpolate.splrep(t_vicon, pos_vicon[:,1], s = 0)
    f_z = interpolate.splrep(t_vicon, pos_vicon[:,2], s = 0)
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
    # get uwb_p
    uwb_p = get_uwbp(pos_vicon)
    # get TDOA and gt
    [tdoa_meas, d_gt] = getTDOA_gt(t_uwb, uwb, uwb_p, anchor_position)
    # visualize bias
    for id in range(8):
       visual_bias(tdoa_meas, d_gt, t_vicon, t, eskf, id)

    plot_pos(t, eskf.Xpo, t_vicon, pos_vicon)
    plot_pos_err(t, pos_error, eskf.Ppo)
    plot_traj(pos_vicon, eskf.Xpo, anchor_position)
    plt.show()


