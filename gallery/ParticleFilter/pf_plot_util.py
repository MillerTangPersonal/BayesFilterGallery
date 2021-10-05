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
from pf_util import wrapToPi_vector

def plot_2d(t,Xpo,vicon):
# plot function for 2D problem
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(311)
    # ax.plot(t, vicon[:,0], color='orangered',linewidth=1.9,alpha=2.0)
    ax.plot(t,vicon[:,0] - Xpo[0,:], color='steelblue',linewidth=1.9, alpha=0.8)
    # ax.plot(t,  3*np.sqrt(Ppo[:,0,0]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    # ax.plot(t, -3*np.sqrt(Ppo[:,0,0]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r'e_x [m]',fontsize=15)
    # plt.legend(['Vicon ground truth','Estimate'])
    plt.title(r"Estimation errors", fontsize=18,  color='black')
    
    bx = fig.add_subplot(312)
    # bx.plot(t, vicon[:,1], color='orangered',linewidth=1.9,alpha=2.0)
    bx.plot(t, vicon[:,1] - Xpo[1,:], color='steelblue',linewidth=1.9, alpha=0.8)
    # bx.plot(t,  3*np.sqrt(Ppo[:,1,1]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    # bx.plot(t, -3*np.sqrt(Ppo[:,1,1]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    bx.set_ylabel(r'e_y [m]',fontsize=15)
    
    ## wrap to pi function not working well 
    cx = fig.add_subplot(313)
    # cx.plot(t, vicon[:,2], color='orangered',linewidth=1.9,alpha=2.0)
    cx.plot(t, wrapToPi_vector(vicon[:,2] - Xpo[2,:]), color='steelblue',linewidth=1.9, alpha=0.8)
    # cx.plot(t,  3*np.sqrt(Ppo[:,2,2]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    # cx.plot(t, -3*np.sqrt(Ppo[:,2,2]), color='red',linestyle='dashed',linewidth=1.0, alpha=0.8)
    cx.set_xlabel(r'time [s]')
    cx.set_ylabel(r'e_th [rad]',fontsize=15)

    fig1 = plt.figure(facecolor="white")
    dx = fig1.add_subplot(111)
    dx.plot(Xpo[0,:], Xpo[1,:], color='orangered',linewidth=1.0,alpha=1.0)
    dx.plot(vicon[:,0], vicon[:,1], color='steelblue',linewidth=1.0,alpha=1.0)
    
    
    fig2 = plt.figure(facecolor="white")
    ax2 = fig2.add_subplot(311)
    ax2.plot(t, vicon[:,0], color='orangered',linewidth=1.9,alpha=1.0)
    ax2.plot(t, Xpo[0,:], color='steelblue',linewidth=1.9, alpha=0.8)
    ax2.set_ylabel(r'x [m]',fontsize=15)
    plt.title(r"Estimation results", fontsize=18,  color='black')
    
    bx2 = fig2.add_subplot(312)
    bx2.plot(t, vicon[:,1], color='orangered',linewidth=1.9,alpha=1.0)
    bx2.plot(t, Xpo[1,:], color='steelblue',linewidth=1.9, alpha=0.8)
    bx2.set_ylabel(r'y [m]',fontsize=15)
    
    cx2 = fig2.add_subplot(313)
    cx2.plot(t, vicon[:,2], color='orangered',linewidth=1.9,alpha=1.0)
    cx2.plot(t, Xpo[2,:], color='steelblue',linewidth=1.9, alpha=0.8)
    cx2.set_xlabel(r'time [s]')
    cx2.set_ylabel(r'theta [rad]',fontsize=15)