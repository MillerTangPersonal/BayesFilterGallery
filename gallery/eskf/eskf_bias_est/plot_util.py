'''
plotting functions
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy.stats import norm
from scipy import interpolate
from pyquaternion import Quaternion
import matplotlib.mlab as mlab
import math
import matplotlib

FONTSIZE = 18;   TICK_SIZE = 16

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'

matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

def plot_pos(t,Xpo,t_vicon,pos_vicon):
    fig = plt.figure(facecolor="white",figsize=(10, 8))
    ax = fig.add_subplot(311)
    ax.plot(t_vicon, pos_vicon[:,0], color='orangered',linewidth=2.5,alpha=0.8)
    ax.plot(t, Xpo[:,0], color='royalblue',linewidth=2.5, alpha=1.0)
    ax.set_ylabel(r'X [m]',fontsize=FONTSIZE)
    plt.legend(['Vicon ground truth','Estimate'])
    plt.title(r"Estimation results", fontsize=FONTSIZE,  color='black')
    plt.xlim(0, max(t))

    ax = fig.add_subplot(312)
    ax.plot(t_vicon, pos_vicon[:,1], color='orangered',linewidth=2.5,alpha=0.8)
    ax.plot(t, Xpo[:,1], color='royalblue',linewidth=2.5, alpha=1.0)
    ax.set_ylabel(r'Y [m]',fontsize=FONTSIZE)
    plt.xlim(0, max(t))

    ax = fig.add_subplot(313)
    ax.plot(t_vicon, pos_vicon[:,2], color='orangered',linewidth=2.5,alpha=0.8)
    ax.plot(t, Xpo[:,2], color='royalblue',linewidth=2.5, alpha=1.0)
    ax.set_xlabel(r'time [s]',fontsize=FONTSIZE)
    ax.set_ylabel(r'Z [m]',fontsize=FONTSIZE)
    plt.xlim(0, max(t))

def plot_pos_err(t,pos_error, Ppo=np.zeros((0, 9, 9))):   
    # extract the variance
    D = Ppo.shape[0]       
    delta_x = np.zeros([D,1])
    delta_y = np.zeros([D,1])
    delta_z = np.zeros([D,1])
    for i in range(D):
        delta_x[i,0] = math.sqrt(Ppo[i,0,0])
        delta_y[i,0] = math.sqrt(Ppo[i,1,1])
        delta_z[i,0] = math.sqrt(Ppo[i,2,2])

    fig = plt.figure(facecolor="white",figsize=(10, 8))
    ax = fig.add_subplot(311)
    plt.title(r"Estimation Error", fontsize=FONTSIZE, fontweight=0, color='black')
    ax.plot(t, pos_error[:,0], color='royalblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_x[:,0], 3*delta_x[:,0],facecolor="pink",alpha=0.3)
    ax.set_ylabel(r'error x [m]',fontsize=FONTSIZE)
    plt.xlim(0, max(t))
    plt.ylim(-0.25,0.25)

    ax = fig.add_subplot(312)
    ax.plot(t, pos_error[:,1], color='royalblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_y[:,0], 3*delta_y[:,0],facecolor="pink",alpha=0.3)
    ax.set_ylabel(r'error y [m]',fontsize=FONTSIZE)
    plt.xlim(0, max(t))
    plt.ylim(-0.25,0.25)

    ax = fig.add_subplot(313)
    ax.plot(t, pos_error[:,2], color='royalblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_z[:,0], 3*delta_z[:,0],facecolor="pink",alpha=0.3)
    ax.set_xlabel(r'time [s]',fontsize=FONTSIZE)
    ax.set_ylabel(r'error z [m]',fontsize=FONTSIZE)
    plt.xlim(0, max(t))
    plt.ylim(-0.25,0.25)

def plot_traj(pos_vicon, Xpo, anchor_pos):
    fig_traj = plt.figure(facecolor = "white",figsize=(10, 8))
    ax_t = fig_traj.add_subplot(projection='3d')
    # make the panes transparent
    ax_t.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # change the color of the grid lines 
    ax_t.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)

    ax_t.plot(pos_vicon[:,0],pos_vicon[:,1],pos_vicon[:,2],color='orangered',linewidth=2.0, alpha=0.9, label='ground truth')
    ax_t.plot(Xpo[:,0], Xpo[:,1], Xpo[:,2],color='royalblue', linewidth=2.0, alpha=1.0, label = 'estimation')
    ax_t.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2],color='Teal', s = 100, alpha = 0.5, label = 'anchors')
    ax_t.set_xlim([-3.5,3.5])
    ax_t.set_ylim([-3.9,3.9])
    ax_t.set_zlim([-0.0,3.0])
    # use LaTeX fonts in the plot
    ax_t.set_xlabel(r'X [m]',fontsize=FONTSIZE)
    ax_t.set_ylabel(r'Y [m]',fontsize=FONTSIZE)
    ax_t.set_zlabel(r'Z [m]',fontsize=FONTSIZE)
    ax_t.legend(loc='best', fontsize=FONTSIZE)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis
    plt.title(r"Trajectory of the experiment", fontsize=FONTSIZE, fontweight=0, color='black', style='italic', y=1.02 )
    

# split TDOA meas and get the corresponding gt
def getTDOA_gt(t_uwb, tdoa, uwb_p, anchor_pos):

    # combine uwb tdoa timestep and meas. for visualization
    tdoa = np.concatenate((t_uwb, tdoa),axis=1) 

    # get the id for tdoa_ij measurements
    tdoa_meas_70 = np.where((tdoa[:,1]==[7])&(tdoa[:,2]==[0]))
    tdoa_70 = np.squeeze(tdoa[tdoa_meas_70, :])

    tdoa_meas_01 = np.where((tdoa[:,1]==[0])&(tdoa[:,2]==[1]))
    tdoa_01 = np.squeeze(tdoa[tdoa_meas_01, :])

    tdoa_meas_12 = np.where((tdoa[:,1]==[1])&(tdoa[:,2]==[2]))
    tdoa_12 = np.squeeze(tdoa[tdoa_meas_12, :])

    tdoa_meas_23 = np.where((tdoa[:,1]==[2])&(tdoa[:,2]==[3]))
    tdoa_23 = np.squeeze(tdoa[tdoa_meas_23, :])

    tdoa_meas_34 = np.where((tdoa[:,1]==[3])&(tdoa[:,2]==[4]))
    tdoa_34 = np.squeeze(tdoa[tdoa_meas_34, :])

    tdoa_meas_45 = np.where((tdoa[:,1]==[4])&(tdoa[:,2]==[5]))
    tdoa_45 = np.squeeze(tdoa[tdoa_meas_45, :])

    tdoa_meas_56 = np.where((tdoa[:,1]==[5])&(tdoa[:,2]==[6]))
    tdoa_56 = np.squeeze(tdoa[tdoa_meas_56, :])

    tdoa_meas_67 = np.where((tdoa[:,1]==[6])&(tdoa[:,2]==[7]))
    tdoa_67 = np.squeeze(tdoa[tdoa_meas_67, :])

    tdoa_meas = np.array([tdoa_70, tdoa_01, tdoa_12, tdoa_23,
                          tdoa_34, tdoa_45, tdoa_56, tdoa_67], dtype=object)
    # compute the ground truth for tdoa_ij
    d = []
    for i in range(8):
        d.append(linalg.norm(anchor_pos[i,:].reshape(1,-1) - uwb_p, axis = 1))
    
    # shape of d is 8 x n
    d = np.array(d)
    # measurement model
    d_70 = d[0,:] - d[7,:]; d_01 = d[1,:] - d[0,:]; d_12 = d[2,:] - d[1,:]; d_23 = d[3,:] - d[2,:]
    d_34 = d[4,:] - d[3,:]; d_45 = d[5,:] - d[4,:]; d_56 = d[6,:] - d[5,:]; d_67 = d[7,:] - d[6,:]
    
    d_gt = np.array([d_70, d_01, d_12, d_23, d_34, d_45, d_56, d_67], dtype=object)

    return tdoa_meas, d_gt

# visualize bias (by id)
def visual_bias(tdoa_meas, d_gt, t_vicon, t, eskf, id):
    gt_time = t_vicon
    # get synchronized gt
    d_int = interp_gt(gt_time, d_gt[id], tdoa_meas[id][:,0])
    # bias = tdoa - gt
    bias = tdoa_meas[id][:,3] - d_int

    plt.figure()
    if id == 0:
        plt.title(r"UWB TDOA meas biass, (An7, An0)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 1:
        plt.title(r"UWB TDOA meas biass, (An0, An1)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 2:
        plt.title(r"UWB TDOA meas biass, (An1, An2)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 3:
        plt.title(r"UWB TDOA meas biass, (An2, An3)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 4:
        plt.title(r"UWB TDOA meas biass, (An3, An4)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 5:
        plt.title(r"UWB TDOA meas biass, (An4, An5)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 6:
        plt.title(r"UWB TDOA meas biass, (An5, An6)", fontsize=FONTSIZE, fontweight=0, color='black')
    elif id == 7:
        plt.title(r"UWB TDOA meas biass, (An6, An7)", fontsize=FONTSIZE, fontweight=0, color='black')

    plt.scatter(tdoa_meas[id][:,0], bias, color = "steelblue", s = 6.5, alpha = 0.9, label = "tdoa biases")
    plt.plot(t, eskf.Xpo[:,6+id], color = "orange", label = "est. biases")
    plt.legend(loc='best',fontsize = FONTSIZE)
    plt.xlabel(r'Time [s]',fontsize = FONTSIZE)
    plt.ylabel(r'TDOA bias [m]',fontsize = FONTSIZE) 
    plt.ylim([-0.7, 0.7]) 

# interpolate gt meas. using UWB time
def interp_gt(gt_time, gt_m, tdoa_time):
    f_gt = interpolate.splrep(gt_time, gt_m, s = 0.5)
    # synchronized gt meas.
    m_int = interpolate.splev(tdoa_time, f_gt, der = 0)
    return m_int

def get_uwbp(gt_pose):
    # gt_pose: [x, y, z, qx, qy, qz, qw]
    # external calibration: convert the gt_position to UWB antenna center
    t_uv = np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1) 
    uwb_p = np.zeros((len(gt_pose), 3))
    for idx in range(len(gt_pose)):
        q_cf =Quaternion([gt_pose[idx,6], gt_pose[idx,3], gt_pose[idx,4], gt_pose[idx,5]])    # [q_w, q_x, q_y, q_z]
        C_iv = q_cf.rotation_matrix                       # rotation matrix from vehicle body frame to inertial frame
        uwb_ac = C_iv.dot(t_uv) + gt_pose[idx,0:3].reshape(-1,1)
        uwb_p[idx,:] = uwb_ac.reshape(1,-1)     # gt of uwb tag
        
    return uwb_p