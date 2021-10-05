'''
plotting functions
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import math
import os
import pandas as pd
# help function
from ukf_util import wrapToPi_vector

# debug plotting: visualize the original uwb and the nn compensated uwb
def plot_debug(t_uwb, uwb, uwb_tfL, bias, i):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(211)
    plt.title(r"uwb tf compensation of Anchor %i" % i, fontsize=18, fontweight=2, color='black')
    ax.plot(t_uwb, uwb, color='orangered', linewidth=2.0, alpha=0.9)
    ax.plot(t_uwb, uwb_tfL, color='darkred', linewidth = 2.0, alpha=0.9)
    ax.set_ylabel(r'Range measurement [m]',fontsize=15) 
    
    bx = fig.add_subplot(212)
    bx.plot(t_uwb, bias, color='dodgerblue',linewidth=1.0,alpha=0.7)
    bx.set_xlabel(r'time [s]',fontsize=15)
    bx.set_ylabel(r'bias [m]',fontsize=15) 

def plot_8uwb(t_uwb, uwb, error, outlier, uwb_outlier_prob, ID, TDOA, TWR):
    # plotting function for the 8 UWB anchors
    # fig = plt.figure(facecolor="white")
    # ax = fig.add_subplot(211)
    # plt.title(r"UWB measurement of Anchor %i" % i, fontsize=18, fontweight=2, color='black')
    # ax.plot(t_uwb, uwb, color='orangered', linewidth=2.0, alpha=0.9)
    # # ax.plot(t_uwb, dist_vicon, color='orangered', linewidth=1.5, alpha=0.9)
    # # ax.set_xlabel(r'time [s]')
    # ax.set_ylabel(r'Range measurement [m]',fontsize=15)  
    
    fig = plt.figure(facecolor="white")
    bx = fig.add_subplot(111)
    bx.scatter(t_uwb, error, color='dodgerblue',linewidth=1.0,alpha=0.7)
    num = outlier.shape[0]
    for i in range(num):
        if outlier[i]!=0:
            bx.scatter(t_uwb[i], outlier[i], linewidth=1.0, facecolors='none', edgecolors='r')
        if uwb_outlier_prob[i]!=0:
            bx.scatter(t_uwb[i], uwb_outlier_prob[i], linewidth=1.0, facecolors='none', edgecolors='orange')
    bx.set_xlabel(r'time [s]',fontsize=15)
    bx.set_ylabel(r'innovation [m]',fontsize=15) 
    if TWR:
        plt.title("UWB tdoa performance %i" % ID)
        SAVE = True
        if ID == 7 and SAVE:
            # save tdoa d1-2 data (tdoa_testing_data1) with DNN3
            os.chdir("/home/william/Desktop/IROS_data/figure5/twr")
            ## save to csv file
            dataFrame = np.append(t_uwb.reshape(-1,1), error.reshape(-1,1), axis = 1)
            dataFrame = np.append(dataFrame, outlier.reshape(-1,1), axis = 1)
            dataFrame = np.append(dataFrame, uwb_outlier_prob.reshape(-1,1), axis = 1)
            df = pd.DataFrame(dataFrame, columns=['time_twr','twr_error','model_based_outlier','outlier_prob'])
            df.to_csv("twr_OutlierRej.csv",index=None)    
            
    if TDOA:
        ID_list = ['d7-0', 'd0-1','d1-2','d2-3','d3-4','d4-5','d5-6','d6-7']
        plt.title("UWB tdoa performance " + ID_list[ID])
        SAVE =False
        if ID == 2 and SAVE:
            # save tdoa d1-2 data (tdoa_testing_data1) with DNN3
            os.chdir("/home/william/Desktop/IROS_data/figure5/tdoa")
            ## save to csv file
            dataFrame = np.append(t_uwb.reshape(-1,1), error.reshape(-1,1), axis = 1)
            dataFrame = np.append(dataFrame, outlier.reshape(-1,1), axis = 1)
            dataFrame = np.append(dataFrame, uwb_outlier_prob.reshape(-1,1), axis = 1)
            df = pd.DataFrame(dataFrame, columns=['time_tdoa','tdoa_error','model_based_outlier','outlier_prob'])
            df.to_csv("tdoa_OutlierRej.csv",index=None)

def plot_pos(t,Xpo,t_vicon,pos_vicon):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(311)
    ax.plot(t_vicon, pos_vicon[:,0], color='orangered',linewidth=1.9,alpha=1.0)
    ax.plot(t, Xpo[:,0], color='steelblue',linewidth=1.9, alpha=0.8)
    ax.set_ylabel(r'X [m]',fontsize=15)
    plt.legend(['Vicon ground truth','Estimate'])
    plt.title(r"Estimation results", fontsize=18,  color='black')
    # plt.ylim((-2.25, 2.25))
    
    ax = fig.add_subplot(312)
    ax.plot(t_vicon, pos_vicon[:,1], color='orangered',linewidth=1.9,alpha=1.0)
    ax.plot(t, Xpo[:,1], color='steelblue',linewidth=1.9, alpha=0.8)
    ax.set_ylabel(r'Y [m]',fontsize=15)
    # plt.ylim((-2.25, 2.25))
    
    ax = fig.add_subplot(313)
    ax.plot(t_vicon, pos_vicon[:,2], color='orangered',linewidth=1.9,alpha=1.0)
    ax.plot(t, Xpo[:,2], color='steelblue',linewidth=1.9, alpha=0.8)
    ax.set_xlabel(r'time [s]')
    ax.set_ylabel(r'Z [m]',fontsize=15)
    # plt.ylim((-0.25, 2.25))
    
def plot_2d(t,Xpo,Ppo,vicon):
    # plot function for 2D problem
    # There is a small problem: Ppo have some small values e-05   
    # a wrap around
    Ppo[:,0,0] = np.clip(Ppo[:,0,0], 0, 1)
    Ppo[:,1,1] = np.clip(Ppo[:,1,1], 0, 1)
    Ppo[:,2,2] = np.clip(Ppo[:,2,2], 0, 1)
    
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(311)
    # ax.plot(t, vicon[:,0], color='orangered',linewidth=1.9,alpha=2.0)
    ax.plot(t,vicon[:,0] - Xpo[0,:], color='steelblue',linewidth=1.9, alpha=0.8)
    ax.plot(t,  3*np.sqrt(Ppo[:,0,0]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    ax.plot(t, -3*np.sqrt(Ppo[:,0,0]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r'e_x [m]',fontsize=15)
    # plt.legend(['Vicon ground truth','Estimate'])
    plt.title(r"Estimation results", fontsize=18,  color='black')
    
    bx = fig.add_subplot(312)
    # bx.plot(t, vicon[:,1], color='orangered',linewidth=1.9,alpha=2.0)
    bx.plot(t, vicon[:,1] - Xpo[1,:], color='steelblue',linewidth=1.9, alpha=0.8)
    bx.plot(t,  3*np.sqrt(Ppo[:,1,1]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    bx.plot(t, -3*np.sqrt(Ppo[:,1,1]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    bx.set_ylabel(r'e_y [m]',fontsize=15)
    
    ## wrap to pi function not working well 
    cx = fig.add_subplot(313)
    # cx.plot(t, vicon[:,2], color='orangered',linewidth=1.9,alpha=2.0)
    cx.plot(t, wrapToPi_vector(vicon[:,2] - Xpo[2,:]), color='steelblue',linewidth=1.9, alpha=0.8)
    cx.plot(t,  3*np.sqrt(Ppo[:,2,2]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    cx.plot(t, -3*np.sqrt(Ppo[:,2,2]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    cx.set_xlabel(r'time [s]')
    cx.set_ylabel(r'e_th [rad]',fontsize=15)

    fig1 = plt.figure(facecolor="white")
    dx = fig1.add_subplot(111)
    dx.plot(Xpo[0,:], Xpo[1,:], color='orangered',linewidth=1.0,alpha=1.0)
    dx.plot(vicon[:,0], vicon[:,1], color='steelblue',linewidth=1.0,alpha=1.0)
    
def plot_pos_err(t,pos_error, NN_COM=True, Constant_Bias=False, OUTLIER_REJ = True, Ppo=np.zeros((0, 9, 9))):     
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(311)
    plt.title(r"Estimation Error with NN_bias= %r,Constant_bias= %r,Outlier_rej= %r" % (NN_COM, Constant_Bias, OUTLIER_REJ),\
         fontsize=11, fontweight=0, color='black')
    SHOW_VAR = Ppo.shape[0]        # when no variance send inside, do not show the variance
    # extract the variance
    D = Ppo.shape[0]       
    delta_x = np.zeros([D,1])
    delta_y = np.zeros([D,1])
    delta_z = np.zeros([D,1])
    for i in range(D):
        delta_x[i,0] = math.sqrt(Ppo[i,0,0])
        delta_y[i,0] = math.sqrt(Ppo[i,1,1])
        delta_z[i,0] = math.sqrt(Ppo[i,2,2])
    ax.plot(t, pos_error[:,0], color='steelblue',linewidth=1.9, alpha=0.9)
    if SHOW_VAR:
        ax.plot(t, 3*delta_x, color='orangered',linewidth=1.9,alpha=0.9)
        ax.plot(t, -3*delta_x, color='orangered',linewidth=1.9,alpha=0.9)
    ax.set_ylabel(r'err_x [m]',fontsize=15)

    ax = fig.add_subplot(312)
    ax.plot(t, pos_error[:,1], color='steelblue',linewidth=1.9, alpha=0.9)
    if SHOW_VAR:
        ax.plot(t, 3*delta_y, color='orangered',linewidth=1.9,alpha=0.9)
        ax.plot(t, -3*delta_y, color='orangered',linewidth=1.9,alpha=0.9)
    ax.set_ylabel(r'err_y [m]',fontsize=15)

    ax = fig.add_subplot(313)
    ax.plot(t, pos_error[:,2], color='steelblue',linewidth=1.9, alpha=0.9)
    if SHOW_VAR:
        ax.plot(t, 3*delta_z, color='orangered',linewidth=1.9,alpha=0.9)
        ax.plot(t, -3*delta_z, color='orangered',linewidth=1.9,alpha=0.9)
    ax.set_xlabel(r'time [s]',fontsize=15)
    ax.set_ylabel(r'err_y [m]',fontsize=15)


    
def plot_vel(t,Xpo,t_vicon,vel_vicon):    
    fig1 = plt.figure(facecolor="white")
    ax = fig1.add_subplot(311)
    ax.plot(t, Xpo[:,3], color='steelblue',linewidth=1.9, alpha=0.9)    
    ax.plot(t_vicon, vel_vicon[:,0], color='orangered',linewidth=1.9,alpha=0.9)
    ax.set_ylabel(r'Vx [m/s]')
    plt.legend(['Estimate','Vicon ground truth'])
    # plt.ylim((-2.25, 2.25))

    ax = fig1.add_subplot(312)
    ax.plot(t, Xpo[:,4], color='steelblue',linewidth=1.9, alpha=0.9)
    ax.plot(t_vicon, vel_vicon[:,1], color='orangered',linewidth=1.9,alpha=0.9)
    ax.set_ylabel(r'Vy [m/s]')
    # plt.ylim((-2.25, 2.25))

    ax = fig1.add_subplot(313)
    ax.plot(t, Xpo[:,5], color='steelblue',linewidth=1.9, alpha=0.9)
    ax.plot(t_vicon, vel_vicon[:,2], color='orangered',linewidth=1.9,alpha=0.9)
    # plt.ylim((-1,3))
    ax.set_xlabel(r'time [s]')
    ax.set_ylabel(r'Vz [m/s]')
    
def plot_traj(pos_vicon, Xpo, anchor_pos):
    fig_traj = plt.figure(facecolor = "white")
    ax_t = fig_traj.add_subplot(111, projection = '3d')
    ax_t.plot(pos_vicon[:,0],pos_vicon[:,1],pos_vicon[:,2],color='steelblue',linewidth=1.9, alpha=0.9)
    ax_t.scatter( Xpo[:,0], Xpo[:,1], Xpo[:,2],color='darkblue',s=0.5, alpha=0.9,linestyle='--')
    ax_t.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2], marker='o',color='red')
    ax_t.set_xlim3d(np.amin(anchor_pos[:,0])-0.5, np.amax(anchor_pos[:,0])+0.5)  
    ax_t.set_ylim3d(np.amin(anchor_pos[:,1])-0.5, np.amax(anchor_pos[:,1])+0.5)  
    ax_t.set_zlim3d(np.amin(pos_vicon[:,2])-0.5, np.amax(pos_vicon[:,2])+0.5)  
    # use LaTeX fonts in the plot
    ax_t.set_xlabel(r'X [m]')
    ax_t.set_ylabel(r'Y [m]')
    ax_t.set_zlabel(r'Z [m]')
    plt.title(r"Trajectory of the experiment", fontsize=13, fontweight=0, color='black', style='italic', y=1.02 )
    
def plot_compare(t, error_nn, error_constant, error_outlier):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(311)
    # ax.plot(t, error_constant[:,0]**2, color='darkred',linewidth=1.9,alpha=0.9)    # without outlier rejection, the error^2 is too large 
    ax.plot(t, error_outlier[:,0]**2, color='orange',linewidth=1.9,alpha=0.9)
    ax.plot(t, error_nn[:,0]**2, color='steelblue',linewidth=1.9, alpha=0.9)
    ax.set_ylabel(r'err_x^2 [m]')
    plt.legend(['constant bias + outlier reject','nn + outlier reject'])

    ax = fig.add_subplot(312)
    # ax.plot(t, error_constant[:,1], color='darkred',linewidth=1.9,alpha=0.9)
    ax.plot(t, error_outlier[:,1]**2, color='orange',linewidth=1.9,alpha=0.9)
    ax.plot(t, error_nn[:,1]**2, color='steelblue',linewidth=1.9, alpha=0.9)
    ax.set_ylabel(r'err_y^2 [m]')

    ax = fig.add_subplot(313)
    # ax.plot(t, error_constant[:,2], color='darkred',linewidth=1.9,alpha=0.9)
    ax.plot(t, error_outlier[:,2]**2, color='orange',linewidth=1.9,alpha=0.9)
    ax.plot(t, error_nn[:,2]**2, color='steelblue',linewidth=1.9, alpha=0.9)
    ax.set_xlabel(r'time [s]')
    ax.set_ylabel(r'err_z^2 [m]')