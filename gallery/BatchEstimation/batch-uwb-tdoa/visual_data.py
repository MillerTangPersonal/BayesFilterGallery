from pickle import TRUE
import rosbag, os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import linalg, stats
import math
from pyquaternion import Quaternion
from scipy import interpolate            
from sklearn.metrics import mean_squared_error


FONTSIZE = 18;   TICK_SIZE = 16
matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 


if __name__ == "__main__":   
    # address of the current script
    cwd = os.path.dirname(__file__)
    # load data
    data = np.load(os.path.join(cwd, "dataset_npz/const4-trial1-traj1.npz"))
    # translation vector from the quadcopter to UWB tag
    t_uv = np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1) 
    t = data["t_sensor"];  imu = data["imu_syn"];      uwb = data["uwb"]
    t_gt = data["t_gt"];   gt_pos = data["gt_pos"];    gt_quat = data["gt_quat"]
    An = data["An"]; 

        # external calibration: convert the gt_position to UWB antenna center
    uwb_p = np.zeros((len(gt_pos), 3))
    for idx in range(len(gt_pos)):
        q_cf =Quaternion([gt_quat[idx,0], gt_quat[idx,1], gt_quat[idx,2], gt_quat[idx,3]])    # [q_w, q_x, q_y, q_z]
        C_iv = q_cf.rotation_matrix       # rotation matrix from vehicle body frame to inertial frame

        uwb_ac = C_iv.dot(t_uv) + gt_pos[idx,0:3].reshape(-1,1)
        uwb_p[idx,:] = uwb_ac.reshape(1,-1)     # gt of uwb tag

    tdoa = np.concatenate((t,uwb), axis =1)
    # select the anchor pair for visualization
    # possible anchor ID = [0,1,2,3,4,5,6,7] 

    # get the id for tdoa_ij measurements
    tdoa_70 = np.where((tdoa[:,1]==[7])&(tdoa[:,2]==[0]))
    tdoa_meas_70 = np.squeeze(tdoa[tdoa_70, :])

    tdoa_01 = np.where((tdoa[:,1]==[0])&(tdoa[:,2]==[1]))
    tdoa_meas_01 = np.squeeze(tdoa[tdoa_01, :])

    tdoa_12 = np.where((tdoa[:,1]==[1])&(tdoa[:,2]==[2]))
    tdoa_meas_12 = np.squeeze(tdoa[tdoa_12, :])

    tdoa_23 = np.where((tdoa[:,1]==[2])&(tdoa[:,2]==[3]))
    tdoa_meas_23 = np.squeeze(tdoa[tdoa_23, :])

    tdoa_34 = np.where((tdoa[:,1]==[3])&(tdoa[:,2]==[4]))
    tdoa_meas_34 = np.squeeze(tdoa[tdoa_34, :])

    tdoa_45 = np.where((tdoa[:,1]==[4])&(tdoa[:,2]==[5]))
    tdoa_meas_45 = np.squeeze(tdoa[tdoa_45, :])

    tdoa_56 = np.where((tdoa[:,1]==[5])&(tdoa[:,2]==[6]))
    tdoa_meas_56 = np.squeeze(tdoa[tdoa_56, :])

    tdoa_67 = np.where((tdoa[:,1]==[6])&(tdoa[:,2]==[7]))
    tdoa_meas_67 = np.squeeze(tdoa[tdoa_67, :])


    # compute the ground truth for tdoa_ij
    d = []
    for i in range(8):
        d.append(linalg.norm(An[i,:].reshape(1,-1) - uwb_p, axis = 1))
    
    # shape of d is 8 x n
    d = np.array(d)

    # measurement model
    d_70 = d[0,:] - d[7,:]
    d_01 = d[1,:] - d[0,:]
    d_12 = d[2,:] - d[1,:]
    d_23 = d[3,:] - d[2,:]

    d_34 = d[4,:] - d[3,:]
    d_45 = d[5,:] - d[4,:]
    d_56 = d[6,:] - d[5,:]
    d_67 = d[7,:] - d[6,:]

    # visualization 
    # UWB TDOA
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(221)
    ax1.scatter(tdoa_meas_70[:,0], tdoa_meas_70[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    ax1.plot(t_gt, d_70, color='red',linewidth=1.5, label = "Vicon ground truth")
    ax1.legend(loc='best')
    ax1.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    ax1.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An7, An0)", fontsize=FONTSIZE, fontweight=0, color='black')

    bx1 = fig1.add_subplot(222)
    bx1.scatter(tdoa_meas_01[:,0], tdoa_meas_01[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    bx1.plot(t_gt, d_01, color='red',linewidth=1.5, label = "Vicon ground truth")
    bx1.legend(loc='best')
    bx1.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    bx1.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An0, An1)", fontsize=FONTSIZE, fontweight=0, color='black')

    cx1 = fig1.add_subplot(223)
    cx1.scatter(tdoa_meas_12[:,0], tdoa_meas_12[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    cx1.plot(t_gt, d_12, color='red',linewidth=1.5, label = "Vicon ground truth")
    cx1.legend(loc='best')
    cx1.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    cx1.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An1, An2)", fontsize=FONTSIZE, fontweight=0, color='black')


    dx1 = fig1.add_subplot(224)
    dx1.scatter(tdoa_meas_23[:,0], tdoa_meas_23[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    dx1.plot(t_gt, d_23, color='red',linewidth=1.5, label = "Vicon ground truth")
    dx1.legend(loc='best')
    dx1.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    dx1.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An2, An3)", fontsize=FONTSIZE, fontweight=0, color='black')

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(221)
    ax2.scatter(tdoa_meas_34[:,0], tdoa_meas_34[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    ax2.plot(t_gt, d_34, color='red',linewidth=1.5, label = "Vicon ground truth")
    ax2.legend(loc='best')
    ax2.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    ax2.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An3, An4)", fontsize=FONTSIZE, fontweight=0, color='black')

    bx2 = fig2.add_subplot(222)
    bx2.scatter(tdoa_meas_45[:,0], tdoa_meas_45[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    bx2.plot(t_gt, d_45, color='red',linewidth=1.5, label = "Vicon ground truth")
    bx2.legend(loc='best')
    bx2.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    bx2.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An4, An5)", fontsize=FONTSIZE, fontweight=0, color='black')

    cx2 = fig2.add_subplot(223)
    cx2.scatter(tdoa_meas_56[:,0], tdoa_meas_56[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    cx2.plot(t_gt, d_56, color='red',linewidth=1.5, label = "Vicon ground truth")
    cx2.legend(loc='best')
    cx2.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    cx2.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An5, An6)", fontsize=FONTSIZE, fontweight=0, color='black')

    dx2 = fig2.add_subplot(224)
    dx2.scatter(tdoa_meas_67[:,0], tdoa_meas_67[:,3], color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
    dx2.plot(t_gt, d_67, color='red',linewidth=1.5, label = "Vicon ground truth")
    dx2.legend(loc='best')
    dx2.set_xlabel(r'Time [s]',fontsize = FONTSIZE)
    dx2.set_ylabel(r'TDOA measurement [m]',fontsize = FONTSIZE) 
    plt.title(r"UWB TDOA measurements, (An6, An7)", fontsize=FONTSIZE, fontweight=0, color='black')

    fig7 = plt.figure(figsize=(10, 8))
    a_x = fig7.add_subplot(311)
    plt.title(r"Ground truth of the trajectory", fontsize=FONTSIZE, fontweight=0, color='black', style='italic', y=1.02)
    a_x.plot(t,imu[:,0],color='steelblue',linewidth=1.9, alpha=0.9, label = "acc x")
    a_x.legend(loc='best')
    a_x.set_ylabel(r'm^2/s',fontsize = FONTSIZE) 
    a_y = fig7.add_subplot(312)
    a_y.plot(t,imu[:,1],color='steelblue',linewidth=1.9, alpha=0.9, label = "acc y")
    a_y.legend(loc='best')
    a_y.set_ylabel(r'm^2/s',fontsize = FONTSIZE) 
    a_z = fig7.add_subplot(313)
    a_z.plot(t,imu[:,2],color='steelblue',linewidth=1.9, alpha=0.9, label = "acc z")
    a_z.legend(loc='best')
    a_z.set_ylabel(r'm^2/s',fontsize = FONTSIZE)
    a_z.set_xlabel(r'Time [s]',fontsize = FONTSIZE)

    plt.show()