'''
test eskf for ros simulation
no lever-arm --> select the data "sim_data.bag"
'''
import rosbag, sys
import time, os
import numpy as np
import math
from scipy import linalg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter.filedialog import askopenfilename
from pyquaternion import Quaternion      # package for quaternion
from ekf_util import isin, cross, update_state
from ekf_util import error_state_update, calculateRPY, zeta
from plot_util import plot_pos, plot_traj, plot_pos_err
from scipy import interpolate            # interpolate vicon

# extract the rosbag 
# get the path of the current script
curr = os.path.dirname(sys.argv[0])
bagFile = askopenfilename(initialdir = curr, title = "Select rosbag")
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

# -------------------- start extract the rosbag ------------------------ #
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
#------------------------ INITIAL CONFIG --------------------------#  
USE_IMU  = True;         USE_UWB_tdoa = True

#------------------------ CONFIG END --------------------------#  
# std deviations of inital states
std_xy0 = 0.05;    std_z0 = 0.05;      std_vel0 = 0.05
std_rp0 = 0.05;    std_yaw0 = 0.05
# Process noise
w_accxyz = 2.0;      w_gyro_rpy = 0.1    # rad/sec
w_vel = 0;           w_pos = 0;          w_att = 0;        
# Constants
GRAVITY_MAGNITUE = 9.81
DEG_TO_RAD  = math.pi/180.0
e3 = np.array([0, 0, 1]).reshape(-1,1)       
# Standard devirations of each sensor (tuning parameter)
# UWB measurements
std_uwb_tdoa = np.sqrt(0.05)
# ----------------------- INITIALIZATION OF EKF -------------------------#
# Create a compound vector t with a sorted merge of all the sensor time bases
time = np.sort(np.concatenate((t_imu, t_tdoa)))
t = np.unique(time)
K=t.shape[0]
# Initial states/inputs
f_k = np.zeros((3,1))       # Initial accelerometer input
f = np.zeros((K, 3))
f[0] = f_k.transpose()
omega0 = np.zeros((3,1))    # Initial angular velocity input
omega = np.zeros((K,3))
omega[0] = omega0.transpose()

# --------------------- Initial position ----------------------- #
X0 = np.zeros((6,1))        # Initial estimate for the state vector
X0[0] = 1.5                  
X0[1] = 0.0
X0[2] = 1.5
q0 = Quaternion([1,0,0,0])  # initial quaternion
R = q0.rotation_matrix
q_list = np.zeros((K,4))    # quaternion list
q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
R_list = np.zeros((K,3,3))  # Rotation matrix list (from body frame to inertial frame) 
# Initial posterior covariance
P0 = np.diag([std_xy0**2, std_xy0**2, std_z0**2,\
              std_vel0**2, std_vel0**2, std_vel0**2,\
              std_rp0**2, std_rp0**2, std_yaw0**2 ])
# nominal-state X = [x, y, z, vx, vy, vz]
Xpr = np.zeros((K,6));    Xpo = np.zeros((K,6))
Xpr[0] = X0.transpose();  Xpo[0] = X0.transpose()

Ppo = np.zeros((K, 9, 9));  Ppr = np.zeros((K, 9, 9))
Ppr[0] = P0;                Ppo[0] = P0

# ----------------------- MAIN EKF LOOP ---------------------#
print('timestep: %f' % len(t))
print('Start state estimation')

for k in range(len(t)-1):                 # k = 0 ~ N-1
    k=k+1                                 # k = 1 ~ N
    # Find what measurements are available at the current time (help function: isin() )
    imu_k, imu_check = isin(t_imu,  t[k-1])
    # separate uwb1 and uwb2
    uwb_k, uwb_check = isin(t_tdoa, t[k-1])
    dt = t[k]-t[k-1]
    # # Process noise
    Fi = np.block([
        [np.zeros((3,3)),   np.zeros((3,3))],
        [np.eye(3),         np.zeros((3,3))],
        [np.zeros((3,3)),   np.eye(3)      ]
    ])
    Vi = (w_accxyz**2)*(dt**2)*np.eye(3)
    Thetai = (w_gyro_rpy**2)*(dt**2)*np.eye(3)
    Qi = np.block([
        [Vi,               np.zeros((3,3)) ],
        [np.zeros((3,3)),  Thetai          ]
    ])

    if imu_check and USE_IMU:
        # We have a new IMU measurement
        # update the prior Xpr based on accelerometer and gyroscope data
        # in simulation, gyro is rad/sec
        omega_k = imu[imu_k,3:] # * DEG_TO_RAD  
        omega[k] = omega_k
        Vpo = Xpo[k-1,3:6]
        ## Acc: G --> m/s^2
        # f_k = imu[imu_k,0:3] * GRAVITY_MAGNITUE
        ## in simulation, acc is in m/s^2
        f_k = imu[imu_k,0:3]
        f[k] = f_k
        dw = omega_k * dt                      # Attitude error
        # nominal state motion model
        # position prediction 
        Xpr[k,0:3] = Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUE*e3) * dt**2
        # velocity prediction
        Xpr[k,3:6] = Xpo[k-1, 3:6] + np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUE*e3) * dt
        # if CF is on the ground
        if Xpr[k, 2] < 0:  
            Xpr[k, 2:6] = np.zeros((1,4))    
        # quaternion update
        qk_1 = Quaternion(q_list[k-1,:])
        dqk  = Quaternion(zeta(dw))       # convert incremental rotation vector to quaternion
        q_pr = qk_1 * dqk                 # compute quaternion multiplication with package
        q_list[k,:] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])  # save quaternion in q_list
        R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list
        # error state covariance matrix 
        # use the rotation matrix from timestep k-1
        R = qk_1.rotation_matrix          
        # Jacobian matrix
        Fx = np.block([
            [np.eye(3),         dt*np.eye(3),      -0.5*dt**2*R.dot(cross(f_k))],
            [np.zeros((3,3)),   np.eye(3),         -dt*R.dot(cross(f_k))       ],
            [np.zeros((3,3)),   np.zeros((3,3)),   linalg.expm(cross(dw)).T    ]            
        ])
        # Process noise matrix Fi, Qi are defined above
        Ppr[k] = Fx.dot(Ppo[k-1]).dot(Fx.T) + Fi.dot(Qi).dot(Fi.T) 
        # Enforce symmetry
        Ppr[k] = 0.5*(Ppr[k] + Ppr[k].T)   
        # print("in imu ",k)
        # print("The variance of x, y, z are [{0}, {1}, {2}]".format(Ppr[k,0,0], Ppr[k,1,1], Ppr[k,2,2]))
    else:
        # if we don't have IMU data
        Ppr[k] = Ppo[k-1] + Fi.dot(Qi).dot(Fi.T)
        # Enforce symmetry
        Ppr[k] = 0.5*(Ppr[k] + Ppr[k].T)  
         
        omega[k] = omega[k-1]
        f[k] = f[k-1]
        dw = omega[k] * dt                      # Attitude error
        # nominal state motion model
        # position prediction 
        Vpo = Xpo[k-1,3:6]
        Xpr[k,0:3] = Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUE*e3) * dt**2
        # velocity prediction
        Xpr[k,3:6] = Xpo[k-1, 3:6] + np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUE*e3) * dt
        # if CF is on the ground
        # if Xpr[k, 2] < 0:  
        #     Xpr[k, 2:6] = np.zeros((1,4))    
        # quaternion update
        qk_1 = Quaternion(q_list[k-1,:])
        dqk  = Quaternion(zeta(dw))       # convert incremental rotation vector to quaternion
        q_pr = qk_1 * dqk                 # compute quaternion multiplication with package
        q_list[k] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])  # save quaternion in q_list
        R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list

    # End of Prediction

    # Initially take our posterior estimates as the prior estimates
    # These are updated if we have sensor measurements (UWB)
    Xpo[k] = Xpr[k]
    Ppo[k] = Ppr[k]
    
    # Update with UWB tdoa measurements  
    if uwb_check and USE_UWB_tdoa:

        an_0 = anchor_position[int(tdoa[uwb_k, 0]), :]
        an_1 = anchor_position[int(tdoa[uwb_k, 1]), :]
        dx0 = Xpr[k,0] - an_0[0];  dx1 = Xpr[k,0] - an_1[0]
        dy0 = Xpr[k,1] - an_0[1];  dy1 = Xpr[k,1] - an_1[1]
        dz0 = Xpr[k,2] - an_0[2];  dz1 = Xpr[k,2] - an_1[2]
        dis_0 = linalg.norm(an_0 - Xpr[k,0:3])   
        dis_1 = linalg.norm(an_1 - Xpr[k,0:3])
        predicted = dis_1 - dis_0                          
        err_uwb = tdoa[uwb_k, 2] - predicted   # error is large at the beginning
        
        # err_uwb = tdoa_meas_fake - predicted
        # print(err_uwb)
        # G is 1 x 9
        G = np.append(np.array([dx1/dis_1 - dx0/dis_0,  
                                dy1/dis_1 - dy0/dis_0, 
                                dz1/dis_1 - dz0/dis_0]).reshape(1,-1), np.zeros((1,6)))
        # uwb covariance
        Q = std_uwb_tdoa**2

        # --- error_state_update --- #
        M = G.dot(Ppr[k]).dot(G.T) + Q
        # print("in uwb ",k)
        # print("The variance of x, y, z are [{0}, {1}, {2}]".format(Ppr[k,0,0], Ppr[k,1,1], Ppr[k,2,2]))
        d_m = math.sqrt(err_uwb**2/M)
        # -------------------- Statistical Validation -------------------- #
        # if d_m < 10.0:
        Kk = (Ppr[k].dot(G.T) / M).reshape(-1,1)
        
        Ppo[k] = (np.eye(9) - Kk.dot(G.reshape(1,-1))).dot(Ppr[k])
        
        Ppo[k] = 0.5 * (Ppo[k] + Ppo[k].T)
        derror = Kk.dot(err_uwb)
        
        # -------------------------- #
        # update nominal states 
        # position, velocity
        Xpo[k] = Xpr[k] + np.squeeze(derror[0:6])
        dq_k = Quaternion(zeta(np.squeeze(derror[6:])))
        
        #update quaternion: q_list
        qk_pr = Quaternion(q_list[k])
        qk_po = qk_pr * dq_k
        q_list[k] = np.array([qk_po.w, qk_po.x, qk_po.y, qk_po.z])  # update quaternion list
        R_list[k]   = qk_po.rotation_matrix   
        
        
print('Finish the state estimation')

## compute the error    
# interpolate Vicon measurements
f_x = interpolate.splrep(t_gt_pose, gt_pos[:,0], s = 0.5)
f_y = interpolate.splrep(t_gt_pose, gt_pos[:,1], s = 0.5)
f_z = interpolate.splrep(t_gt_pose, gt_pos[:,2], s = 0.5)
x_interp = interpolate.splev(t, f_x, der = 0).reshape(-1,1)
y_interp = interpolate.splev(t, f_y, der = 0).reshape(-1,1)
z_interp = interpolate.splev(t, f_z, der = 0).reshape(-1,1)

x_error = Xpo[:,0] - np.squeeze(x_interp)
y_error = Xpo[:,1] - np.squeeze(y_interp)
z_error = Xpo[:,2] - np.squeeze(z_interp)

pos_error = np.concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)

rms_x = math.sqrt(mean_squared_error(x_interp, Xpo[:,0]))
rms_y = math.sqrt(mean_squared_error(y_interp, Xpo[:,1]))
rms_z = math.sqrt(mean_squared_error(z_interp, Xpo[:,2]))
print('The RMS error for position x is %f [m]' % rms_x)
print('The RMS error for position y is %f [m]' % rms_y)
print('The RMS error for position z is %f [m]' % rms_z)

RMS_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)          
print('The overall RMS error of position estimation is %f [m]\n' % RMS_all)
        
plot_pos(t, Xpo, t_gt_pose, gt_pos)
plot_pos_err(t, pos_error, Ppo)
plt.show()
        
        
        
