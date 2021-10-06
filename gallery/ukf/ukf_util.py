'''
some helpling functions for the UKF
'''
import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
import matplotlib
FONTSIZE = 18;   TICK_SIZE = 16

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'

matplotlib.rc('xtick', labelsize=TICK_SIZE) 
matplotlib.rc('ytick', labelsize=TICK_SIZE) 

def isin(t_np,t_k):
    # check if t_k is in the numpy array t_np. If t_k is in t_np, return the index and bool = Ture.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        b = True
        return res[0][0], b
    b = False
    return 0, b

def cross(v):    # input: 3x1 vector, output: 3x3 matrix
    vx = np.array([
        [ 0,    -v[2], v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1],  v[0], 0 ] 
    ])
    return vx

def denormalize(scl, norm_data):
    # @param: scl: the saved scaler   norm_data: 
    norm_data = norm_data.reshape(-1,1)
    new = scl.inverse_transform(norm_data)
    return new


def getSigmaP(X, L, kappa, dim):
    # get sigma points
    # return sigma_points, the num of sigma points
    X = X.reshape(-1,1)
    w = np.zeros((dim,1))
    mu = np.concatenate((X, w), axis=0)
    L_num = mu.shape[0]
    Z_SP= mu

    for idx in range(L_num):
        # i=idx: 1,...,L 
        z_i  = mu + math.sqrt(L_num + kappa) * L[:, idx].reshape(-1,1)
        z_iL = mu - math.sqrt(L_num + kappa) * L[:, idx].reshape(-1,1)
        Z_SP = np.hstack((Z_SP, z_i))
        Z_SP = np.hstack((Z_SP, z_iL))
        
    # Z_SP = [z_0; z_1, z_(1+L), ..., z_i, z_(i+L)]
    return Z_SP, Z_SP.shape[1], L_num

def getAlpha(idx,L_num,kappa):
    if idx == 0:
        alpha = kappa/(L_num + kappa)
    else:
        alpha = 1/(2*(L_num + kappa))
    return alpha

def wrapToPi(err_th):
    # wrap the theta error to [-pi, pi]
    # wrap a scalar angle error
    while err_th < -math.pi:
        err_th += 2 * math.pi
    while err_th > math.pi:
        err_th -= 2 * math.pi
    return err_th

def wrapToPi_vector(err_th):
    # wrap a vector angle error
    # wrap to [-pi, pi]
    err_th = (err_th + np.pi) % (2 * np.pi) - np.pi
    return err_th


# help functions for visualization
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