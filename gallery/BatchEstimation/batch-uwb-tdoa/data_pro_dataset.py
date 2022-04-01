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

def array2id(array):
    if array ==  "anchor_0":
        return 0
    elif array == "anchor_1":
        return 1
    elif array == "anchor_2":
        return 2
    elif array == "anchor_3":
        return 3
    elif array == "anchor_4":
        return 4
    elif array == "anchor_5":
        return 5
    elif array == "anchor_6":
        return 6
    elif array == "anchor_7":
        return 7
    else:
        print(array)
        print("error in array2id\n")
        return -1

if __name__ == "__main__":    
 
    # ros_bag = "sim-data/1130_simData_larm.bag"             #with lever-arm
    anchor_npz = "/home/wenda/utias_uwb_dataset/dataset/flight-dataset/survey-results/anchor_const1.npz"
    # access the survey results
    anchor_survey = np.load(anchor_npz)
    anchor_position = anchor_survey['an_pos']

    # access rosbag file
    ros_bag  = "/home/wenda/data/uwb_dataset/rosbag-data/const1/bag/test_ds5_cv_imu.bag"
    bag = rosbag.Bag(ros_bag)

    # -------------------- start extract the rosbag ------------------------ #
    gt_pos = [];    gt_quat=[];    t_gt_pose = []
    imu = [];       t_imu = [];    uwb = [];       t_uwb = []

    # anchor list array
    an_list = []; 

    for topic, msg, t in bag.read_messages(['/tdoa_data', '/pose_data', "/imu_data"]):
        if topic == '/pose_data':
            gt_pos.append([msg.pose.pose.position.x,   msg.pose.pose.position.y,  msg.pose.pose.position.z])
            gt_quat.append([ msg.pose.pose.orientation.w ,  msg.pose.pose.orientation.x,  msg.pose.pose.orientation.y,  msg.pose.pose.orientation.z])
            t_gt_pose.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

        if topic == '/imu_data':
            # imu unit: m/s^2 in acc, rad/sec in gyro
            imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                        msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)    

        if topic == '/tdoa_data':
            uwb.append([-1, -1, msg.data[0]])
            an_list.append([msg.anchor_i[0], msg.anchor_j[0]])
            t_uwb.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)



    # sensor
    t_uwb = np.array(t_uwb);         t_imu = np.array(t_imu); an_list = np.array(an_list)
    uwb = np.array(uwb);             imu = np.array(imu)


    # convert the anchor list array to int
    for i in range(len(t_uwb)):
        uwb[i,0] = array2id(an_list[i,0])
        uwb[i,1] = array2id(an_list[i,1])
        
    # data process: get the unique timestamp
    t_uwb_u, idx_uwb = np.unique(t_uwb, return_index=True)
    # t_imu_u, idx_imu = np.unique(t_imu, return_index=True)  # imu has unique timestamp

    # get uwb at unique time: {t_uwb_u, uwb_u}
    uwb_u = uwb[idx_uwb,:]
    uwb_u = uwb_u[::2]
    t_uwb_u = t_uwb_u[::2]

    # downsample imu: only select the odd rows
    t_imu_d1 = t_imu[::2]
    imu_d1 = imu[::2]
    t_imu_d2 = t_imu_d1[::2]
    imu_d2 = imu_d1[::2]
    t_imu_ds = t_imu_d2[::2]
    imu_ds = imu_d2[::2]
    t_imu_ds = t_imu_ds[::2]
    imu_ds = imu_ds[::2]

    
    print("start syn!\n")
    # synchronize IMU data to UWB time
    imu_syn = synchnize_imu(t_imu_ds, imu_ds, t_uwb_u)
    print("down syn!\n")

    t_sensor = t_uwb_u       # t of the sensor (imu is sync to uwb)
    min_t = min(t_sensor)
    # get the vicon information from min_t
    t_gt_pose = np.array(t_gt_pose);        idx = np.argwhere(t_gt_pose > min_t); t_gt_pose = t_gt_pose[idx]
    gt_pos = np.array(gt_pos)[idx,:];       gt_pos = np.squeeze(gt_pos)
    gt_quat = np.array(gt_quat)[idx,:];     gt_quat = np.squeeze(gt_quat)


    # reset ROS time base
    t_gt_pose = (t_gt_pose - min_t).reshape(-1,1)
    t_sensor = (t_sensor - min_t).reshape(-1,1)

    # save t_sensor, imu, uwb, t_gt_pose, gt_pos, gt_quat 
    np.savez('test_dataset.npz', t_sensor = t_sensor, imu_syn = imu_syn, uwb = uwb_u,\
             t_gt=t_gt_pose, gt_pos=gt_pos, gt_quat=gt_quat, An=anchor_position)