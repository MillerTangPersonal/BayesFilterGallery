'''
An offline UKF with IMU, UWB, and flow-deck measurements 
'''
#!/usr/bin/env python3
import argparse
import rosbag
import time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import math
from cf_msgs.msg import Accel, Gyro, Flow, Tdoa, Tof
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from scipy import linalg
from sklearn.metrics import mean_squared_error
from scipy import interpolate      # interpolate vicon
from pyquaternion import Quaternion
# import packages for NN
from ukf_util import isin, cross, getSigmaP, getAlpha, plot_pos, plot_traj, plot_pos_err 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', nargs=2)
    args = parser.parse_args()
    
    # access the survey results
    anchor_npz = args.i[0]
    anchor_survey = np.load(anchor_npz)
    # select anchor constellations
    anchor_position = anchor_survey['an_pos']
    # print out
    anchor_file = os.path.split(sys.argv[-2])[1]
    print("\n selecting anchor constellation " + str(anchor_file) + "\n")

    # access rosbag file
    ros_bag = args.i[1]
    bag = rosbag.Bag(ros_bag)
    bagFile = os.path.split(sys.argv[-1])[1]

    # print out
    bag_name = os.path.splitext(bagFile)[0]
    print("visualizing rosbag: " + str(bagFile) + "\n")

    # -------------------- start extract the rosbag ------------------------ #
    pos_vicon=[];      t_vicon=[]; 
    uwb=[];            t_uwb=[]; 
    imu=[];            t_imu=[]; 
    for topic, msg, t in bag.read_messages(['/pose_data', '/tdoa_data', '/imu_data']):
        if topic == '/pose_data':
            pos_vicon.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            t_vicon.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        if topic == '/tdoa_data':
            uwb.append([msg.idA, msg.idB, msg.data])
            t_uwb.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        if topic == '/imu_data':
            imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y,  msg.linear_acceleration.z,\
                        msg.angular_velocity.x,    msg.angular_velocity.y,     msg.angular_velocity.z     ])
            t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)


    min_t = min(t_uwb + t_imu + t_vicon)
    # get the vicon information from min_t
    t_vicon = np.array(t_vicon);              
    idx = np.argwhere(t_vicon > min_t);     
    t_vicon = t_vicon[idx]; 
    pos_vicon = np.squeeze(np.array(pos_vicon)[idx,:])

    # sensor
    t_imu = np.array(t_imu);       imu = np.array(imu);  
    t_uwb = np.array(t_uwb);       uwb = np.array(uwb);     

    # reset ROS time base
    t_vicon = (t_vicon - min_t).reshape(-1,1)
    t_imu = (t_imu - min_t).reshape(-1,1)
    t_uwb = (t_uwb - min_t).reshape(-1,1)

    # ---------------------- Parameter ---------------------- #
    USE_IMU  = True;    USE_UWB_tdoa = True; 

    std_xy0 = 0.1;    std_z0 = 0.1;      std_vel0 = 0.1
    std_rp0 = 0.1;    std_yaw0 = 0.1
    # Process noise
    w_accxyz = 2.0;      w_gyro_rpy = 0.1    # rad/sec
    w_vel = 0;           w_pos = 0;          w_att = 0;        
    # Constants
    GRAVITY_MAGNITUDE = 9.81
    DEG_TO_RAD  = math.pi/180.0
    e3 = np.array([0, 0, 1]).reshape(-1,1)     

    # Standard devirations of each sensor (tuning parameter)
    # UWB measurements: sigma**2 = 0.05 
    std_uwb_tdoa = np.sqrt(0.05)
    std_flow = 0.1;          std_tof = 0.0001

    # sigma points
    kappa = 5

    # external calibration: translation vector from the quadcopter to UWB tag
    t_uv = np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1)  

    # ----------------------- INITIALIZATION OF EKF -------------------------#
    # Create a compound vector t with a sorted merge of all the sensor time bases
    time = np.sort(np.concatenate((t_imu, t_uwb)))
    t = np.unique(time)
    K=t.shape[0]
    # Initial states/inputs
    f_k = np.zeros((3,1))       # Initial accelerometer input
    f = np.zeros((K, 3))
    f[0] = f_k.transpose()
    omega0 = np.zeros((3,1))    # Initial angular velocity input
    omega = np.zeros((K,3))
    omega[0] = omega0.transpose()
    # nominal-state: x, y, z, vx, vy, vz, q = [qw, qx, qy, qz]
    # error state vector: [dx, dy, dz, dvx, dvy, dvz, \delta pi_1, \delta pi_2, \delta pi_3]
    # [\delta pi_1, \delta pi_2, \delta pi_3] is the 3 dimension rotational correction vector, 
    # which can be used to correct the rotational matrix

    # --------------------- Initial position ----------------------- #
    X0 = np.zeros((6,1))        # Initial estimate for the state vector
    X0[0] = 1.25                  
    X0[1] = 0.0
    X0[2] = 0.07
    q0 = Quaternion([1,0,0,0])  # initial quaternion
    R = q0.rotation_matrix
    q_list = np.zeros((K,4))    # quaternion list
    q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
    R_list = np.zeros((K,3,3))  # Rotation matrix list (from body frame to inertial frame) 
    # Initial posterior covariance
    P0 = np.diag([std_xy0**2,  std_xy0**2,  std_z0**2,\
                std_vel0**2, std_vel0**2, std_vel0**2,\
                std_rp0**2,  std_rp0**2,  std_yaw0**2 ])
    # nominal-state X = [x, y, z, vx, vy, vz]
    Xpr = np.zeros((K,6));      Xpo = np.zeros((K,6))
    Xpr[0] = X0.transpose();    Xpo[0] = X0.transpose()

    Ppo = np.zeros((K, 9, 9));  Ppr = np.zeros((K, 9, 9))
    Ppr[0] = P0;                Ppo[0] = P0

    # ----------------------- MAIN UKF LOOP ---------------------#
    print('timestep: %f' % len(t))
    print('\nStart state estimation')
    for k in range(len(t)-1):                 # k = 0 ~ N-1
        k=k+1                                 # k = 1 ~ N
        # Find what measurements are available at the current time (help function: isin() )
        imu_k,  imu_check  = isin(t_imu,   t[k-1])
        uwb_k,  uwb_check  = isin(t_uwb,   t[k-1])
        dt = t[k]-t[k-1]
        # # Process noise
        Fi = np.block([
            [np.zeros((3,3)),   np.zeros((3,3))],
            [np.eye(3),         np.zeros((3,3))],
            [np.zeros((3,3)),   np.eye(3)      ]
        ])
        Vi = (w_accxyz**2)*(dt**2)*np.eye(3)
        Thetai = (w_gyro_rpy**2)*(dt**2)*np.eye(3)

        # TODO: check the process noise in UKF
        Qi = np.block([
            [Vi,               np.zeros((3,3)) ],
            [np.zeros((3,3)),  Thetai          ]
        ])

        if imu_check and USE_IMU:
            # We have a new IMU measurement
            # make prediction Xpr, Ppr based on accelerometer and gyroscope data
            omega_k = imu[imu_k, 3:] * DEG_TO_RAD
            omega[k] = omega_k        # used when no IMU data is coming
            Vpo = Xpo[k-1, 3:6]
            # Acc: G --> m/s^2
            f_k = imu[imu_k, 0:3] * GRAVITY_MAGNITUDE 
            f[k] = f_k                # used when no IMU data is coming
            # ------------------------------------------#
            Ppo_sigma = Ppo[k-1] + np.identity(9)*0.001  # add a small positive num on the idag of Ppo 

            ###### find the problem: Q is not correct ####################
            Sigma_zz = linalg.block_diag(Ppo_sigma, Qi)   
            L = linalg.cholesky(Sigma_zz, lower = True)                 # not positive definite
            # get sigmapoint 
            Z_sp, sp_num, L_num = getSigmaP(Xpo[k-1], L, kappa, 9)   # dimension of the state is 9
            # extract sigma points into state and motion noise
            sp_X = Z_sp[0:9, :] 
            sp_w = Z_sp[9:, :]
            # Make Prediction: send sigma points into nonlinear motion model
            # input: d_omega_k, f_k
            sp_Xpr = np.zeros_like(sp_X)

            # Problem #1
            for idx in range(sp_num):
                # Rotation prediction 
                # d_omega_k: row vector
                # Attitude error with sigmapoint noise, in Q the noise has timed dt
                d_omega_k = omega_k*dt  + sp_w[6:, idx].transpose()  
                R = linalg.expm(cross(d_omega_k)).dot(R)
                # R = R.dot(linalg.expm(cross(d_omega_k)))
                R_list[k] = R
                # Position prediction (Inertial frame)
                sp_Xpr[0:3, idx] = sp_X[0:3, idx] + sp_X[3:6, idx] * dt + 0.5 * np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2 + sp_w[0:3, idx]
                # Velocity prediction (Inertial frame)
                sp_Xpr[3:6, idx] = sp_X[3:6, idx] + np.squeeze(R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt + sp_w[3:6, idx]
                
            # get Xpr[k]  
            for idx in range(sp_num):
                alpha = getAlpha(idx, L_num, kappa)
                Xpr[k] = Xpr[k] + alpha * sp_Xpr[:,idx] 
                        
            # get Ppr[k]
            for idx in range(sp_num):
                alpha = getAlpha(idx, L_num, kappa)
                delat_X = (sp_Xpr[:,idx] - Xpr[k]).reshape(-1,1)      # column vector
                Ppr[k] = Ppr[k] + alpha * delat_X.dot(delat_X.transpose())
            # Enforce symmetry
            Ppr[k] = 0.5*(Ppr[k] + Ppr[k].transpose())    
        else:
            # If we don't have IMU data, integrate the last acceleration to give prior estimate
            Ppr[k] = Ppo[k-1] + Qi
            Ppr[k] = 0.5*(Ppr[k-1] + Ppr[k-1].transpose())    
            # use IMU data from last timestamp
            omega[k] = omega[k-1]
            f[k] = f[k-1]
            # Rotation prediction
            d_omega_k = omega[k]*dt  
            R = linalg.expm(cross(d_omega_k)).dot(R)
            # R = R.dot(linalg.expm(cross(d_omega_k)))
            R_list[k] = R
            # Position prediction
            Xpr[k, 0:3] = Xpo[k-1, 0:3] + Xpo[k-1, 3:6]*dt + 0.5 * np.squeeze(R.dot(f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2      
            # Velocity prediction
            Xpr[k, 3:6] = Xpo[k-1, 3:6] + np.squeeze(R.dot(f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt
        # end of prediction
        
        # Initially take our posterior estimates as the prior estimates
        # These are updated if we have sensor measurements (UWB)
        Xpo[k] = Xpr[k]
        Ppo[k] = Ppr[k]

        # Update with UWB tdoa measurements       


    print('Finish the state estimation\n')

    # # update with UWB range measurement (TWR)
    # if uwb_check and USE_UWB_TWR:
    #     Ppo_sigma = Ppr[k] + np.identity(9)*0.0001  # add a small positive num on the diag of Ppo 
    #     Rk = std_uwb**2*np.eye(uwb_num)
    #     Sigma_zz = linalg.block_diag(Ppo_sigma, Rk) 
    #     L_update = linalg.cholesky(Sigma_zz, lower = True)
    #     # get sigmapoint 
    #     Z_sp, sp_num, L_num = getSigmaP(Xpr[k], L_update, kappa, uwb_num)
    #     # extract sigma points into state and motion noise
    #     sp_X = Z_sp[0:9, :] 
    #     sp_w = Z_sp[9:, :]
    #     # Correction: send sigma points into nonlinear measurement model
    #     # yk_sigma = g(xk_sigma, nk_sigma)
    #     sp_y = np.zeros((uwb_num, sp_num))
    #     # sp_y = [y_uwb0_sigma0,  y_uwb0_sigma1, ... , y_uwb0_sigma_sp_num
    #     #         y_uwb1_sigma0,  y_uwb1_sigma1, ... , y_uwb1_sigma_sp_num
    #     #         :
    #     #         y_uwb7_sigma0,  y_uwb7_sigma1, ... , y_uwb7_sigma_sp_num  ]
    #     for idx in range(sp_num):
    #         for i in range(uwb_num):
    #             dxi = sp_X[0,idx] - anchor_pos[i, 0]
    #             dyi = sp_X[1,idx] - anchor_pos[i, 1]
    #             dzi = sp_X[2,idx] - anchor_pos[i, 2]
    #             sp_y[i,idx] = math.sqrt(dxi**2 + dyi**2 + dzi**2) + sp_w[i, idx]
        
    #     # get mu_yk
    #     mu_yk = np.zeros((uwb_num,1))
    #     for idx in range(sp_num):
    #         alpha = getAlpha(idx, L_num, kappa)
    #         mu_yk = mu_yk + alpha * sp_y[:,idx].reshape(-1,1)
    #     # get Sigma_yy_k and Sigma_xy_k
    #     Sigma_yy_k = np.zeros((uwb_num,uwb_num))
    #     Sigma_xy_k = np.zeros((9, uwb_num))
    #     for idx in range(sp_num):
    #         alpha = getAlpha(idx, L_num, kappa)
    #         delat_y = sp_y[:,idx].reshape(-1,1) - mu_yk      # column vector
    #         delat_x = (sp_X[:,idx] - Xpr[k]).reshape(-1,1)
            
    #         Sigma_yy_k = Sigma_yy_k + alpha * delat_y.dot(delat_y.transpose())
    #         Sigma_xy_k = Sigma_xy_k + alpha * delat_x.dot(delat_y.transpose())
    #     # Update Xpo[k] and Ppo[k]
    #     # Calculate Kalman Gain 
    #     Kk = Sigma_xy_k.dot(linalg.inv(Sigma_yy_k))
    #     Ppo[k] = Ppr[k] - Kk.dot(Sigma_xy_k.transpose())
    #     # Enforce symmetry of covariance matrix
    #     Ppo[k] = 0.5 * (Ppo[k] + Ppo[k].transpose())
    #     # innovation error term
    #     Err_uwb[uwb_k,:] = uwb[uwb_k,:] - np.squeeze(mu_yk)
        
    #     Xpo[k] = Xpr[k] + np.squeeze(Kk.dot(Err_uwb[uwb_k,:].reshape(-1,1)))
    #     updated = False
        
    # if updated:
    # # update rotation only
    # # follow the EKF rotation update, not sure if this part is useful
    #     v = Xpo[k][6:9]*dt
    #     A = linalg.block_diag(np.eye(3), np.eye(3),linalg.expm(cross(v/2.0)))
    #     Ppo[k] = A.dot(Ppo[k]).dot(A.transpose())
    #     Ppo[k] = 0.5*(Ppo[k]+Ppo[k].transpose())
    #     R = linalg.expm(cross(v)).dot(R)
    #     Xpo[k][6:9] = np.array([0,0,0])
        
    plot_pos(t,Xpo,t_vicon,pos_vicon)
        
    plt.show()