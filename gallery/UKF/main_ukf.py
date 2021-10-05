'''
An offline UKF with IMU, UWB measurements (TWR and TDoA) and flowdeck
Need to be organize
'''
import rosbag, sys
import time, string, os
import numpy as np
import math
import pandas as pd
from scipy import linalg
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy import interpolate      # interpolate vicon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as style
from tkinter.filedialog import askopenfilename
# import packages for NN
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
# help functions
from plot_util import plot_8uwb, plot_pos, plot_vel, plot_traj, plot_pos_err, plot_debug
from ukf_util import isin, cross, denormalize, getSigmaP, getAlpha
# ----------------------      load and process the data      ------------------- #
style.use('ggplot')
os.chdir("/home/william/Documents/Bias_testing")
curr = os.getcwd()
bagFile = askopenfilename(initialdir = curr, title = "Select rosbag")
# access rosbag
bag = rosbag.Bag(bagFile)
base = os.path.basename(bagFile)
selected_bag = os.path.splitext(base)[0]
anchor_pos = loadmat(curr+'/Vicon10.mat')['anchor_pos'].T

pos_vicon=[]; t_vicon=[]; uwb1=[]; t_uwb1=[]; uwb2=[]; t_uwb2=[]; imu=[]; t_imu=[]; flow=[]; t_flow=[]
for topic, msg, t in bag.read_messages(['/full_state','/log/uwb1','/log/uwb2','/log/imu','/log/flowdeck']):
    if topic == '/full_state':
        pos_vicon.append(msg.fullstate[0].pos)
        t_vicon.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    if topic == '/log/uwb1':
        uwb1.append(msg.logdata[0].values)
        t_uwb1.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    if topic == '/log/uwb2':
        uwb2.append(msg.logdata[0].values)
        t_uwb2.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    if topic == '/log/imu':
        imu.append(msg.logdata[0].values)
        t_imu.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    if topic == "/log/flowdeck":
        flow.append(msg.logdata[0].values)
        t_flow.append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        
min_t = min(t_uwb1 + t_uwb2 + t_imu + t_flow)
# get the vicon infor from min_t
t_vicon = np.array(t_vicon);               idx = np.argwhere(t_vicon > min_t);      t_vicon = t_vicon[idx]
pos_vicon = np.array(pos_vicon)[idx,:];    pos_vicon = np.squeeze(pos_vicon)

t_uwb1 = np.array(t_uwb1);  t_uwb2 = np.array(t_uwb2);   t_imu = np.array(t_imu);   t_flow = np.array(t_flow)
uwb1 = np.array(uwb1);      uwb2 = np.array(uwb2);       imu = np.array(imu);       flow = np.array(flow) 
# Reset ROS time base
t_vicon = (t_vicon - min_t).reshape(-1,1)
t_uwb1 = (t_uwb1 - min_t).reshape(-1,1);    t_uwb2 = (t_uwb2 - min_t).reshape(-1,1)
t_imu = (t_imu - min_t).reshape(-1,1);      t_flow = (t_flow - min_t).reshape(-1,1)

# Create a unified time base for all UWB measurements
if t_uwb1.shape[0] < t_uwb2.shape[0]:
    idex_uwb2 =[]
    for i in range(uwb1.shape[0]):
        idex_uwb2.append((np.abs(t_uwb1[i]-t_uwb2)).argmin())
    uwb2 = uwb2[idex_uwb2][:]
    idx_uwb = np.argwhere(abs(t_uwb1-t_uwb2[idex_uwb2])<=0.033)[:,0]   # get the first column
else:
    idex_uwb1 =[]
    for i in range(uwb2.shape[0]):
        idex_uwb1.append((np.abs(t_uwb2[i]-t_uwb1)).argmin())
    uwb1 = uwb1[idex_uwb1][:]
    idx_uwb = np.argwhere(abs(t_uwb2-t_uwb1[idex_uwb1])<=0.033)[:,0]   # get the first column 
uwb1 = uwb1[idx_uwb][:]
uwb2 = uwb2[idx_uwb][:]
t_uwb = t_uwb1[idx_uwb]
uwb = np.concatenate((uwb1, uwb2), axis=1)          # combine uwb data into the shape: num x 8 
# -----------------------------    Param Setting    ------------------------------- #
# Decide which sensor to fuse 
USE_IMU = True;      USE_UWB_TWR = True;    USE_UWB_tdoa = False
USE_FLOW = False;    USE_ZRANGER = False
# use neural network to compensate and outlier rejection
NN_COM = False;      NN_tdoa_COM = False;     Constant_Bias = True
Outlier_Rej = True;  Outlier_Prob = True
# Visualization 
PLOT_est = True;     PLOT_UWB = False
ComVel = False;      VisTraj = False;       Vis_PosErr = False
# load the DNN for TWR
if NN_COM:
    os.chdir("/home/william/dsl__projects__uwbBias/NN_BiasLearning/NN_pytorch")
    NN = torch.load('BiasNet.pkl')
    scaler_x = joblib.load("scaler_x")       # scaler for input x feature
    scaler_y = joblib.load("scaler_y")       # scaler for output (UWB bias)
if NN_tdoa_COM:
    os.chdir("/home/william/dsl__projects__uwbBias/NN_BiasLearning/NN_tdoa_pytorch")
    DNN_tdoa = torch.load('BiasNet_tdoa.pkl')
    scaler_tdoa_x = joblib.load("scaler_x")       # scaler for input x feature
    scaler_tdoa_y = joblib.load("scaler_y")       # scaler for output (UWB bias)
# -----------------------------    Initial Config     ----------------------------- #
# std deviations fo initial states
std_xy0 = 0.15;     std_z0 = 0.15;     std_vel0 = 0.05
std_rp0 = 0.05;     std_yaw0 = 0.05
# Process noise
w_accxy = 2;        w_accz = 1;        w_vel = 0;        w_pos = 0
w_att = 0;          w_gyro_rp = 0.1;   w_gyro_yaw = 0.1
# Constants
GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD = math.pi/180.0
e3 = np.array([0, 0, 1]).reshape(-1,1)
# Flowdeck constants
N_pix = 30.0               # number of pixels in the square image
theta_pix = 4.2 * DEG_TO_RAD
omega_factor = 1.25
# Standard devirations of each sensor (tuning parameter)
std_uwb = 0.03;      std_uwb_tdoa = 0.2
std_flow = 0.1;       std_tof = 0.0001
# ----------------------------- Initialization of UKF ----------------------------- #
# Create a compound vector t with a sorted merge of all the sensor time bases
time = np.sort(np.concatenate((t_imu, t_uwb, t_flow))) 
t = np.unique(time)
K = t.shape[0]
# Initial states/inputs
f_0 = np.zeros((1,3))      # initial accelerometer input
f = np.zeros((K,3));       f[0] = f_0  
omega0 = np.zeros((1,3))   # initial angular velocity input
omega = np.zeros((K,3));   omega[0] = omega0
# [x, y, z, vx, vy, vz, roll, pitch, yaw]
# [0, 1, 2, 3,  4,  5,  6,    7,     8]
# --------- Initial Position -------- #
X0 = np.zeros((9,1))
X0[0] = 1.5          # initial x
X0[1] = 0.0          # initial y
# ------------ hyperparam ----------- #
kappa = 2
# ----------------------------------- #
R = np.eye(3)   # Rotation matrix from body frame to inertial frame
R_list = np.zeros((K,3,3))
R_list[0] = R
# Initial posterior covariance
P0 = np.diag([std_xy0**2,    std_xy0**2,   std_z0*2,\
              std_vel0**2,   std_vel0**2,  std_vel0**2,\
              std_rp0**2,    std_rp0**2,   std_yaw0**2 ])

Xpr = np.zeros((K,9));        Xpo = np.zeros((K,9))
Xpr[0] = X0.transpose();      Xpo[0] = X0.transpose()

Ppo = np.zeros((K, 9, 9));    Ppr = np.zeros((K, 9, 9))
Ppo[0] = P0;                  Ppr[0] = P0
# -------------------- UWB TWR constant bias ---------------------- #
anchor_id = [1,2,3,4,5,6,7,8]
uwb_num = len(anchor_id)
con_bias=np.array([-0.298,  -0.232,  -0.196,  -0.109, 
                   -0.184,  -0.216,  -0.169,  -0.288]).reshape(1,-1)
Err_uwb = np.zeros_like(uwb)                                              # save the uwb error for visualization
# ---------------------- velocity from vicon ---------------------- # 
len_v = t_vicon.shape[0]
vel_vicon = np.zeros((len_v,3))
for k,_ in enumerate(t_vicon[1:],start=1):  # only get index, start from the second value
    dt = t_vicon[k] - t_vicon[k-1]
    vel_vicon[k] = (pos_vicon[k] - pos_vicon[k-1])/dt

# ----------------------- MAIN UKF LOOP ----------------------- #
print('timestep: {0}'.format(len(t)))
print('Start State Estimation!!!')
for k,_ in enumerate(t[1:], start=1):     # not the best way writing for loop start from 1
    # Find what measurements are available at the current time (help function: isin() )
    imu_k,  imu_check  = isin(t_imu,   t[k-1])
    uwb_k,  uwb_check  = isin(t_uwb,   t[k-1])
    flow_k, flow_check = isin(t_flow,  t[k-1])
    dt = t[k]-t[k-1]
    # Process noise in EKF
    Q = np.diag([(w_accxy*dt**2 + w_vel*dt + w_pos)**2,\
              (w_accxy*dt**2 + w_vel*dt + w_pos)**2,\
              (w_accz*dt**2 + w_vel*dt)**2,\
              (w_accxy*dt + w_vel)**2,\
              (w_accxy*dt + w_vel)**2,\
              (w_accz*dt + w_vel)**2,\
              (w_gyro_rp*dt + w_att)**2,\
              (w_gyro_rp*dt + w_att)**2,\
              (w_gyro_yaw*dt + w_att)**2])
    # print(k)
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
        Sigma_zz = linalg.block_diag(Ppo_sigma,Q)   
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
        # R = R.dot(linalg.expm(cross(d_omega_k)))  or  R = linalg.expm(cross(d_omega_k)).dot(R)
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
            alpha = getAlpha(idx,L_num, kappa)
            Xpr[k] = Xpr[k] + alpha * sp_Xpr[:,idx] 
            
        if Xpr[k,2] < 0: # on the ground
            Xpr[k, 2:6] = np.zeros((1,4))
            
        # get Ppr[k]
        for idx in range(sp_num):
            alpha = getAlpha(idx,L_num, kappa)
            delat_X = (sp_Xpr[:,idx] - Xpr[k]).reshape(-1,1)      # column vector
            Ppr[k] = Ppr[k] + alpha * delat_X.dot(delat_X.transpose())
        # Enforce symmetry
        Ppr[k] = 0.5*(Ppr[k] + Ppr[k].transpose())    
    else:
        # If we don't have IMU data, integrate the last acceleration to give prior estimate
        Ppr[k] = Ppo[k-1] + Q
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
    updated = False
    
    # update with UWB range measurement (TWR)
    if uwb_check and USE_UWB_TWR:
        Ppo_sigma = Ppr[k] + np.identity(9)*0.0001  # add a small positive num on the diag of Ppo 
        Rk = std_uwb**2*np.eye(uwb_num)
        Sigma_zz = linalg.block_diag(Ppo_sigma, Rk) 
        L_update = linalg.cholesky(Sigma_zz, lower = True)
        # get sigmapoint 
        Z_sp, sp_num, L_num = getSigmaP(Xpr[k], L_update, kappa, uwb_num)
        # extract sigma points into state and motion noise
        sp_X = Z_sp[0:9, :] 
        sp_w = Z_sp[9:, :]
        # Correction: send sigma points into nonlinear measurement model
        # yk_sigma = g(xk_sigma, nk_sigma)
        sp_y = np.zeros((uwb_num, sp_num))
        # sp_y = [y_uwb0_sigma0,  y_uwb0_sigma1, ... , y_uwb0_sigma_sp_num
        #         y_uwb1_sigma0,  y_uwb1_sigma1, ... , y_uwb1_sigma_sp_num
        #         :
        #         y_uwb7_sigma0,  y_uwb7_sigma1, ... , y_uwb7_sigma_sp_num  ]
        for idx in range(sp_num):
            for i in range(uwb_num):
                dxi = sp_X[0,idx] - anchor_pos[i, 0]
                dyi = sp_X[1,idx] - anchor_pos[i, 1]
                dzi = sp_X[2,idx] - anchor_pos[i, 2]
                sp_y[i,idx] = math.sqrt(dxi**2 + dyi**2 + dzi**2) + sp_w[i, idx]
        
        # get mu_yk
        mu_yk = np.zeros((uwb_num,1))
        for idx in range(sp_num):
            alpha = getAlpha(idx, L_num, kappa)
            mu_yk = mu_yk + alpha * sp_y[:,idx].reshape(-1,1)
        # get Sigma_yy_k and Sigma_xy_k
        Sigma_yy_k = np.zeros((uwb_num,uwb_num))
        Sigma_xy_k = np.zeros((9, uwb_num))
        for idx in range(sp_num):
            alpha = getAlpha(idx, L_num, kappa)
            delat_y = sp_y[:,idx].reshape(-1,1) - mu_yk      # column vector
            delat_x = (sp_X[:,idx] - Xpr[k]).reshape(-1,1)
            
            Sigma_yy_k = Sigma_yy_k + alpha * delat_y.dot(delat_y.transpose())
            Sigma_xy_k = Sigma_xy_k + alpha * delat_x.dot(delat_y.transpose())
        # Update Xpo[k] and Ppo[k]
        # Calculate Kalman Gain 
        Kk = Sigma_xy_k.dot(linalg.inv(Sigma_yy_k))
        Ppo[k] = Ppr[k] - Kk.dot(Sigma_xy_k.transpose())
        # Enforce symmetry of covariance matrix
        Ppo[k] = 0.5 * (Ppo[k] + Ppo[k].transpose())
        # innovation error term
        Err_uwb[uwb_k,:] = uwb[uwb_k,:] - np.squeeze(mu_yk)
        
        Xpo[k] = Xpr[k] + np.squeeze(Kk.dot(Err_uwb[uwb_k,:].reshape(-1,1)))
        updated = False
        
    if updated:
    # update rotation only
    # follow the EKF rotation update, not sure if this part is useful
        v = Xpo[k][6:9]*dt
        A = linalg.block_diag(np.eye(3), np.eye(3),linalg.expm(cross(v/2.0)))
        Ppo[k] = A.dot(Ppo[k]).dot(A.transpose())
        Ppo[k] = 0.5*(Ppo[k]+Ppo[k].transpose())
        R = linalg.expm(cross(v)).dot(R)
        Xpo[k][6:9] = np.array([0,0,0])
        
print('State Estimation Finished.')  

if PLOT_est:
    plot_pos(t,Xpo,t_vicon,pos_vicon)
    
plt.show()