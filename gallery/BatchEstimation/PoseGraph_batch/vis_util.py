import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import numpy as np
import math

FONTSIZE = 18;   TICK_SIZE = 16

def visual_traj(traj, landmarks):
    fig = plt.figure(0)
    ax = fig.add_subplot(projection='3d')
    # Plot estimate position
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='b',
        label='Ground Truth')
    ax.plot(traj[:, 3], traj[:, 4], traj[:, 5], color='g',
        label='Estimate')
    ax.scatter(landmarks[0,:], landmarks[1,:], landmarks[2,:],
        marker='o', color='r',label='Landmarks')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Stereo based localization')
    ax.legend()


def visual_est(t, pos_error, Ppo=np.zeros((0, 9, 9))):   
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
    ax.plot(t, pos_error[:,0], color='steelblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_x[:,0], 3*delta_x[:,0],facecolor="teal",alpha=0.3)
    ax.set_ylabel(r'error x [m]',fontsize=FONTSIZE)

    ax = fig.add_subplot(312)
    ax.plot(t, pos_error[:,1], color='steelblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_y[:,0], 3*delta_y[:,0],facecolor="teal",alpha=0.3)
    ax.set_ylabel(r'error y [m]',fontsize=FONTSIZE)

    ax = fig.add_subplot(313)
    ax.plot(t, pos_error[:,2], color='steelblue',linewidth=2.0, alpha=1.0)
    # plot variance
    plt.fill_between(t,-3*delta_z[:,0], 3*delta_z[:,0],facecolor="teal",alpha=0.3)
    ax.set_xlabel(r'time [s]',fontsize=FONTSIZE)
    ax.set_ylabel(r'error z [m]',fontsize=FONTSIZE)
