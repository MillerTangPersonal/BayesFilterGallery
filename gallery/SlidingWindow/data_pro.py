'''
load and sync imu and uwb data
'''
import rosbag
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion
from scipy import interpolate  

''' help function'''
def interp_meas(t1, meas1, t2):
    f_meas1 = interpolate.splrep(t1, meas1, s = 0.5)
    # synchronized meas 1 w.r.t. t2
    syn_m1 = interpolate.splev(t2, f_meas1, der = 0)
    return syn_m1

def synchnize_imu(t_imu, imu, t_uwb):
    acc_x  = interp_meas(t_imu, imu[:,0], t_uwb).reshape(-1,1)
    acc_y  = interp_meas(t_imu, imu[:,1], t_uwb).reshape(-1,1)
    acc_z  = interp_meas(t_imu, imu[:,2], t_uwb).reshape(-1,1)
    gyro_x = interp_meas(t_imu, imu[:,3], t_uwb).reshape(-1,1)
    gyro_y = interp_meas(t_imu, imu[:,4], t_uwb).reshape(-1,1)
    gyro_z = interp_meas(t_imu, imu[:,5], t_uwb).reshape(-1,1)
    imu_syn = np.concatenate((acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z), axis = 1)
    return imu_syn

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
    # ros_bag = "sim-data/simData_lever_arm.bag"       # with lever-arm
    # ros_bag = "sim-data/sim_data.bag"                # without lever-arm
    ros_bag = "sim-data/sim_uwb_batch2.bag"             # with lever-arm (with gt imu)
    bag = rosbag.Bag(ros_bag)

    # -------------------- start extract the rosbag ------------------------ #
    gt_pos = [];    gt_quat=[];    t_gt_pose = []
    imu = [];       t_imu = [];    uwb = [];       t_uwb = []
    # gt imu for testing
    t_imu_gt = [];  imu_gt = []
    # gt odometry
    t_odom = [];     odom_gt = []
    for topic, msg, t in bag.read_messages(["/firefly/ground_truth/pose", "/firefly/ground_truth/imu", 
                                            "/firefly/ground_truth/odometry", "/firefly/imu", "/uwb_tdoa"]):
        if topic == '/firefly/ground_truth/pose':
            gt_pos.append([msg.position.x,  msg.position.y,  msg.position.z])
            gt_quat.append([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
            t_gt_pose.append(t.secs + t.nsecs * 1e-9)

        if topic == '/firefly/ground_truth/imu':
            imu_gt.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                           msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z])
            t_imu_gt.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)   

        if topic == '/firefly/ground_truth/odometry':
            odom_gt.append([msg.twist.twist.linear.x,   msg.twist.twist.linear.y,  msg.twist.twist.linear.z,
                            msg.twist.twist.angular.x,  msg.twist.twist.angular.y, msg.twist.twist.angular.z])
            t_odom.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

        if topic == '/firefly/imu':
            # imu unit: m/s^2 in acc, rad/sec in gyro
            imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                        msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)    

        if topic == '/uwb_tdoa':
            uwb.append([msg.id_A, msg.id_B, msg.dist_diff])
            t_uwb.append(msg.stamp.secs + msg.stamp.nsecs * 1e-9)

    # sensor
    t_uwb = np.array(t_uwb);         t_imu = np.array(t_imu)
    uwb = np.array(uwb);             imu = np.array(imu)

    t_imu_gt = np.array(t_imu_gt);   imu_gt = np.array(imu_gt)

    # data process: get the unique timestamp
    t_uwb_u, idx_uwb = np.unique(t_uwb, return_index=True)
    # t_imu_u, idx_imu = np.unique(t_imu, return_index=True)  # imu has unique timestamp

    # get uwb at unique time: {t_uwb_u, uwb_u}
    uwb_u = uwb[idx_uwb,:]
    # synchronize IMU data to UWB time
    IMU_GT = True
    if IMU_GT:
        imu_syn = synchnize_imu(t_imu_gt, imu_gt, t_uwb_u)
    else:
        imu_syn = synchnize_imu(t_imu, imu, t_uwb_u)

    t = t_uwb_u
    min_t = min(t)
    # get the vicon information from min_t
    t_gt_pose = np.array(t_gt_pose);        idx = np.argwhere(t_gt_pose > min_t); t_gt_pose = t_gt_pose[idx]
    gt_pos = np.array(gt_pos)[idx,:];       gt_pos = np.squeeze(gt_pos)
    gt_quat = np.array(gt_quat)[idx,:];     gt_quat = np.squeeze(gt_quat)
    # gt of the odometry
    t_odom = np.array(t_odom);              idx = np.argwhere(t_odom > min_t); t_odom = t_odom[idx]
    odom_gt = np.array(odom_gt)[idx,:];     odom_gt = np.squeeze(odom_gt)

    # synchronize the odometry
    lin_x = interp_meas(t_odom, odom_gt[:,0], t).reshape(-1,1)
    lin_y = interp_meas(t_odom, odom_gt[:,1], t).reshape(-1,1)
    lin_z = interp_meas(t_odom, odom_gt[:,2], t).reshape(-1,1)
    ang_x = interp_meas(t_odom, odom_gt[:,3], t).reshape(-1,1)
    ang_y = interp_meas(t_odom, odom_gt[:,4], t).reshape(-1,1)
    ang_z = interp_meas(t_odom, odom_gt[:,5], t).reshape(-1,1)

    odom_syn = np.concatenate((lin_x, lin_y, lin_z, ang_x, ang_y, ang_z), axis = 1)

    # reset ROS time base
    t_gt_pose = (t_gt_pose - min_t).reshape(-1,1)
    t = (t - min_t).reshape(-1,1)

    # save t, imu, uwb, t_gt_pose, gt_pos, gt_quat 
    np.savez('sim_uwb_batch2.npz', t=t, imu_syn=imu_syn, uwb = uwb_u, odom = odom_syn,\
             t_gt=t_gt_pose, gt_pos=gt_pos, gt_quat=gt_quat, An=anchor_position)