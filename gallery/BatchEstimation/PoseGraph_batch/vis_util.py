import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

def visual_traj(traj, landmarks):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
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