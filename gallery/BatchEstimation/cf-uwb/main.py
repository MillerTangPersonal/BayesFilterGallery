'''
    main file for Batch estimation in S03 (test with simulation data)
    no lever-arm --> select the data "sim_ekf_test.bag"
'''
import rosbag, sys
import time, os
import numpy as np
import math
from scipy import linalg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

curr = os.path.dirname(sys.argv[0])
bagFile = os.path.join(curr, "sim_ekf_test.bag")
print(bagFile)
# access rosbag of Crazyflie
bag = rosbag.Bag(bagFile)
base = os.path.basename(bagFile)
selected_bag = os.path.splitext(base)[0]

anchor_position = np.array([[ 4.0,  4.3,  3.0],
                            [-4.0,  4.3,  3.0],
                            [ 4.0, -4.3,  3.0],
                            [-4.0, -4.3,  3.0],
                            [ 4.0,  4.3,  0.1],
                            [-4.0,  4.3,  0.1],
                            [ 4.0, -4.3,  0.1],
                            [-4.0, -4.3,  0.1]
                            ])


# -------------------- rosbag data process ------------------------ #
gt_pos = []; gt_quat=[]; t_gt_pose = []
imu = []; t_imu = []; tdoa = []; t_tdoa = []

for topic, msg, t in bag.read_messages(["/firefly/ground_truth/pose", "/firefly/imu", "/uwb_tdoa"]):
    if topic == '/firefly/ground_truth/pose':
        gt_pos.append([msg.position.x,    
                       msg.position.y,    
                       msg.position.z])
        gt_quat.append([msg.orientation.w,
                        msg.orientation.x, 
                        msg.orientation.y,
                        msg.orientation.z 
                        ])
        t_gt_pose.append(t.secs + t.nsecs * 1e-9)
        
    if topic == '/firefly/imu':
        # imu unit: m/s^2 in acc, rad/sec in gyro
        imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                    msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z])
        t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        
    if topic == '/uwb_tdoa':
        tdoa.append([msg.id_A, msg.id_B, msg.dist_diff])
        t_tdoa.append(msg.stamp.secs + msg.stamp.nsecs * 1e-9)

min_t = min(t_imu + t_tdoa)
# get the vicon information from min_t
t_gt_pose = np.array(t_gt_pose); idx = np.argwhere(t_gt_pose > min_t); t_gt_pose = t_gt_pose[idx]
gt_pos = np.array(gt_pos)[idx,:];    gt_pos = np.squeeze(gt_pos)
gt_quat = np.array(gt_quat)[idx,:];  gt_quat = np.squeeze(gt_quat)

# sensor
t_tdoa = np.array(t_tdoa);  t_imu = np.array(t_imu)
tdoa = np.array(tdoa);      imu = np.array(imu)
# reset ROS time base
t_gt_pose = (t_gt_pose - min_t).reshape(-1,1)
t_imu = (t_imu - min_t).reshape(-1,1)
t_tdoa = (t_tdoa - min_t).reshape(-1,1)

# --------- compute the input data ----------- #
# IMU measurements: 
# acc: m/s^2    acc. in x, y, and z 
# gyro: rad/s   angular velocity 




