''' Particle Filter code for a 2D estimation'''
import numpy as np
import math
from numpy.random import seed                   # fix the randomness
import string, os
import scipy
from scipy import linalg
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.style as style
# import help functions
from pf_util import wrapToPi,wrapToPi_vector
from pf_plot_util import plot_2d
# select the matplotlib plotting style
style.use('ggplot')
# fix random seed
seed(30)
np.set_printoptions(precision=3)
os.chdir("/home/wenda/BayesFilterGallery/gallery/ParticleFilter")
curr = os.getcwd()
# load .mat data
t = loadmat(curr+'/dataset2.mat')['t']
x_true = loadmat(curr+'/dataset2.mat')['x_true']
y_true = loadmat(curr+'/dataset2.mat')['y_true']
th_true = loadmat(curr+'/dataset2.mat')['th_true']
l = loadmat(curr+'/dataset2.mat')['l']
# measurements
r_meas = loadmat(curr+'/dataset2.mat')['r'];   r_var = loadmat(curr+'/dataset2.mat')['r_var']
r_std = np.squeeze(np.sqrt(r_var))
b_meas = loadmat(curr+'/dataset2.mat')['b'];   b_var = loadmat(curr+'/dataset2.mat')['b_var']
b_std = np.squeeze(np.sqrt(b_var))
# inputs
v = loadmat(curr+'/dataset2.mat')['v'];   v_var = loadmat(curr+'/dataset2.mat')['v_var']
om = loadmat(curr+'/dataset2.mat')['om']; om_var = loadmat(curr+'/dataset2.mat')['om_var']
d = loadmat(curr+'/dataset2.mat')['d']

vicon_gt = np.concatenate([x_true, y_true], axis=1)
vicon_gt = np.concatenate([vicon_gt, th_true], axis=1)
K = t.shape[0]
T = 0.1
# -------------------- initial Position ---------------------- #
X0 = np.zeros(3)
X0[0] = x_true[0]
X0[1] = y_true[0]
X0[2] = th_true[0]
Xpr = np.zeros((3,K));  Xpo = np.zeros((3,K))
Xpr[:,0] = X0;    Xpo[:,0] = X0
# Initial covariance
P0 = np.diag([0.01, 0.01, 0.01])
Ppr = np.zeros((K, 3, 3));      Ppo = np.zeros((K, 3, 3))
Ppr[0] = P0;                    Ppo[0] = P0    
# -------------------- Parameters ---------------------- #     
REAL_TIME = False           # real time plot
weight_gain = 1000
# motion noise
Q = np.diag([v_var[0,0], om_var[0,0]])
# measurement noise
R = np.diag([r_var[0,0], b_var[0,0]])
state_num = 3;                  input_dim = 2        
# Set Max range
r_max = 5
for i in range(r_meas.shape[0]):
    for j in range(r_meas.shape[1]):
        if r_meas[i,j] > r_max:
            r_meas[i,j] = 0.0
# ----------------------- Initial Particles ----------------------- #
# uniform sampling N_sample points as particles
N_sample = 10
particles = np.empty((N_sample, state_num))
particles[:,0] = np.random.uniform(x_true[0]-1.0, x_true[0]+1.0, N_sample)
particles[:,1] = np.random.uniform(y_true[0]-1.0, y_true[0]+1.0, N_sample)
particles[:,2] = np.random.uniform(th_true[0]-0.1, th_true[0]+0.1, N_sample)
# motion noise
motion_noise = np.empty((N_sample, input_dim))
# weights
weights = np.empty((N_sample, 1))
print('Start PF State Estimation!!!')

if REAL_TIME:
    plt.ion()
    plt.show()
    plt.cla()
    k=0
    plt.scatter(vicon_gt[k,0], vicon_gt[k,1], marker='o', c='red')
    plt.scatter(particles[:,0], particles[:,1], marker = 'x', c='steelblue',alpha=0.6, s=5)
    plt.xlim((-2, 10))
    plt.ylim((-3, 4))
    plt.pause(0.3)
# ----------------------- MAIN PF LOOP ----------------------- #
for k in range(K-1):
    k=k+1
    # PF: prediction
    # noises for v_k and om_k
    motion_noise[:,0] = np.random.normal(0, np.sqrt(v_var[0,0]), N_sample)
    motion_noise[:,1] = np.random.normal(0, np.sqrt(om_var[0,0]), N_sample)
    
    v_sample = v[k-1,0] + motion_noise[:,0]
    u_sample = motion_noise[:,0]
    
    # here the elemently mutiplication
    particles[:,0] += T * np.cos(particles[:,2]) * (v[k-1,0] + motion_noise[:,0]) 
    particles[:,1] += T * np.sin(particles[:,2]) * (v[k-1,0] + motion_noise[:,0])
    # particles[:,0] += T * (v_sample * np.cos(particles[:,2]) - u_sample*np.sin(particles[:,2]))
    # particles[:,1] += T * (v_sample * np.sin(particles[:,2]) + u_sample*np.cos(particles[:,2]))
    particles[:,2] += T * (om[k-1,0] + motion_noise[:,1])
    # wrap the angle
    particles[:,2] = wrapToPi_vector(particles[:,2])
    
    Xpr[:,k] = sum(particles * weights)
    
    # If no measurement data, extact the mean and variance from the prior particles
    if np.all(r_meas[k,:] == 0):
        # print("No measurements at time step",k)
        Xpo[:,k] = Xpr[:,k]
        
        # real time plotting
        if REAL_TIME:
            plt.cla()
            plt.scatter(vicon_gt[k,0], vicon_gt[k,1], marker='o', c='red')
            plt.scatter(Xpo[0,k],      Xpo[1,k],      marker='o', c='blue')
            plt.scatter(particles[:,0], particles[:,1], marker = 'x', c='steelblue',alpha=0.6, s=5)
            plt.xlim((-2, 10))
            plt.ylim((-3, 4))
            plt.pause(0.01)
        
    # If there are measurements
    else:
        # PF: correction
        landmark_idx = np.nonzero(r_meas[k,:])
        landmark_idx = np.asarray(landmark_idx).reshape(1,-1)
        # measurement number at timestep k
        R_num = np.count_nonzero(r_meas[k,:], axis=0)
        # yk_m = [r_particles, theta_particles]   (N_sample x R_num x 2)
        yk_m = np.zeros((N_sample, R_num, 2))
        # reset weights to 1.0
        weights.fill(1.)
        for r_idx in range(R_num):
            landmark_x = l[landmark_idx[0, r_idx],  0]
            landmark_y = l[landmark_idx[0, r_idx],  1]
            # yk_m = g(xk_m, 0)
            yk_m[:,r_idx,0] = np.sqrt((landmark_x - particles[:,0] - d*np.cos(particles[:,2]))**2 + \
                                    (landmark_y - particles[:,1] - d*np.sin(particles[:,2]))**2)
            
            y_tan = landmark_y - particles[:,1] - d * np.sin(particles[:,2])
            x_tan = landmark_x - particles[:,0] - d * np.cos(particles[:,2])
            # wrap the angle
            arctan = wrapToPi_vector( np.arctan2(y_tan, x_tan) )
            yk_m[:,r_idx,1] = wrapToPi_vector(arctan - particles[:,2])
            
            r_meas_k = r_meas[k,landmark_idx[0,r_idx]]
            b_meas_k = b_meas[k,landmark_idx[0,r_idx]]  # already [-pi, pi]
            ## debug
            # p(r_i,theta_i) = p(r_i)*p(theta_i)
            # print("prob of r")
            # print(scipy.stats.norm(yk_m[:,r_idx,0], r_std).pdf(r_meas_k))
            # print("yk_m[:,r_idx,1] ",yk_m[:,r_idx,1])
            # print("b_meas_k", b_meas_k)
            # print("prob of b")
            # print(scipy.stats.norm(yk_m[:,r_idx,1], b_std).pdf(b_meas_k))
            
            weights[:,0] *= weight_gain * scipy.stats.norm(yk_m[:,r_idx,0], r_std).pdf(r_meas_k) 
            # Problem: if considering the bearing measurements, PF estimation has the bias
                    #   * np.exp(-0.5*(wrapToPi_vector(yk_m[:,r_idx,1] - b_meas_k))**2/b_var).reshape(-1)
                    #   * scipy.stats.norm(yk_m[:,r_idx,1], b_std).pdf(b_meas_k)
            # print(sum(weights))

        if sum(weights):        
            weights /= sum(weights)          # normalize the weights
        else:
            # if sum(weights) = 0, set equal weights to all the sample points
            # update the weights line 153-154 solve this problem
            print("The sum of weights are zero at time step ",k)    
            weights.fill(1./N_sample)  
        
        # Resample the posterior particles, create N_sample bins
        bins = np.cumsum(weights)      
        sample_tmp = np.random.uniform(0, 1, N_sample)
        particle_idx = np.digitize(sample_tmp, bins)
        
        # posterior particles
        particles = particles[particle_idx,:]
        weights = weights[particle_idx]
        weights /= sum(weights)
        
        # Extact the mean and variance from the posterior particles
        Xpo[:,k] = sum(particles * weights)
        
        # real time plot
        if REAL_TIME:
            plt.cla()
            plt.scatter(vicon_gt[k,0], vicon_gt[k,1], marker='o', c='red')
            plt.scatter(Xpo[0,k],      Xpo[1,k],      marker='o', c='orange')
            plt.scatter(particles[:,0], particles[:,1], marker = 'x', c='steelblue',alpha=0.6, s=5)
            plt.xlim((-2, 10))
            plt.ylim((-3, 4))
            plt.pause(0.01)

print('State Estimation Finished')
# visualization
if REAL_TIME:
    plt.ioff()
    plt.show()

plot_2d(t,Xpo,vicon_gt)

plt.show()

