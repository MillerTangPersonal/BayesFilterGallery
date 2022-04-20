'''
load and sync imu and uwb data from original dataset type
'''
import rosbag
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion
from scipy import interpolate  

GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD  = math.pi/180.0

''' help function'''
def interp_meas(t1, meas1, t2):
    # synchronized meas 1 w.r.t. t2
    syn_m1 = np.interp(t2, t1, meas1)
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
    dataset_path = "/home/wenda/repositories/util-uwb-dataset/dataset/flight-dataset/"
    anchor_npz = dataset_path + "survey-results/anchor_const1.npz"
    # access the survey results
    anchor_survey = np.load(anchor_npz)
    anchor_position = anchor_survey['an_pos']

    # access rosbag file
    ros_bag  = dataset_path + "rosbag-data/const1/const1-trial1-tdoa2.bag"
    bag = rosbag.Bag(ros_bag)

    # -------------------- start extract the rosbag ------------------------ #
    gt_pos = [];    gt_quat=[];    t_gt_pose = []
    imu = [];       t_imu = [];    uwb = [];       t_uwb = []

    for topic, msg, t in bag.read_messages(['/tdoa_data', '/pose_data', "/imu_data"]):
        if topic == '/pose_data':
            gt_pos.append([msg.pose.pose.position.x,   msg.pose.pose.position.y,  msg.pose.pose.position.z])
            gt_quat.append([ msg.pose.pose.orientation.w ,  msg.pose.pose.orientation.x,  msg.pose.pose.orientation.y,  msg.pose.pose.orientation.z])
            t_gt_pose.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

        if topic == '/imu_data':
            # the rosbag data need to be converted
            # imu unit: m/s^2 in acc, rad/sec in gyro
            imu.append([msg.linear_acceleration.x * GRAVITY_MAGNITUDE,
                        msg.linear_acceleration.y * GRAVITY_MAGNITUDE, 
                        msg.linear_acceleration.z * GRAVITY_MAGNITUDE,
                        msg.angular_velocity.x * DEG_TO_RAD,    
                        msg.angular_velocity.y * DEG_TO_RAD,    
                        msg.angular_velocity.z * DEG_TO_RAD])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)    

        if topic == '/tdoa_data':
            uwb.append([msg.idA, msg.idB, msg.data])
            t_uwb.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

    # sensor
    t_uwb = np.array(t_uwb);         t_imu = np.array(t_imu)
    uwb = np.array(uwb);             imu = np.array(imu)

    # downsample UWB and IMU data
    # only select the odd rows
    # downsample two times 
    uwb = uwb[::2];   t_uwb = t_uwb[::2]
    uwb = uwb[::2];   t_uwb = t_uwb[::2]

    imu = imu[::2];   t_imu = t_imu[::2]
    imu = imu[::2];   t_imu = t_imu[::2]

    print("start syn!\n")
    # synchronize IMU data to UWB time
    imu_syn = synchnize_imu(t_imu, imu, t_uwb)
    print("down syn!\n")

    t_sensor = t_uwb                 # t of the sensor (imu is sync to uwb)
    min_t = min(t_sensor)

    # get the vicon information from min_t
    t_gt_pose = np.array(t_gt_pose);        idx = np.argwhere(t_gt_pose > min_t); t_gt_pose = t_gt_pose[idx]
    gt_pos = np.array(gt_pos)[idx,:];       gt_pos = np.squeeze(gt_pos)
    gt_quat = np.array(gt_quat)[idx,:];     gt_quat = np.squeeze(gt_quat)

    # reset ROS time base
    t_gt_pose = (t_gt_pose - min_t).reshape(-1,1)
    t_sensor = (t_sensor - min_t).reshape(-1,1)

    # save t_sensor, imu, uwb, t_gt_pose, gt_pos, gt_quat 
    np.savez('const1-trial1.npz', t_sensor = t_sensor, imu_syn = imu_syn, uwb = uwb,\
             t_gt=t_gt_pose, gt_pos=gt_pos, gt_quat=gt_quat, An=anchor_position)