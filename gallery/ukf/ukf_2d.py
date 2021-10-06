''' UKF Testing code for a 2D estimation'''
import numpy as np
import math
import string, os
from scipy import linalg
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.style as style
# import help functions
from ukf_util import getSigmaP, getAlpha, wrapToPi
from plot_util import plot_2d
# select the matplotlib plotting style
style.use('ggplot')
np.set_printoptions(precision=3)
os.chdir("/home/wenda/BayesFilterGallery/gallery/ukf")   
curr = os.getcwd()
# load .mat data
t = loadmat(curr+'/dataset2.mat')['t']
x_true = loadmat(curr+'/dataset2.mat')['x_true']
y_true = loadmat(curr+'/dataset2.mat')['y_true']
th_true = loadmat(curr+'/dataset2.mat')['th_true']
l = loadmat(curr+'/dataset2.mat')['l']
# measurements
r_meas = loadmat(curr+'/dataset2.mat')['r'];   r_var = loadmat(curr+'/dataset2.mat')['r_var']
b_meas = loadmat(curr+'/dataset2.mat')['b'];   b_var = loadmat(curr+'/dataset2.mat')['b_var']
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
# noise
Q = np.diag([v_var[0,0], om_var[0,0]]);   R = np.diag([r_var[0,0], b_var[0,0]])
state_num = 3;                  input_dim = 2        
kappa = 5
# Set Max range
r_max = 10

for i in range(r_meas.shape[0]):
    for j in range(r_meas.shape[1]):
        if r_meas[i,j] > r_max:
            r_meas[i,j] = 0.0
# ----------------------- MAIN UKF LOOP ----------------------- #
print('Start UKF State Estimation!!!')
for k in range(K-1):
    k = k+1 
    # UKF prediction
    Ppr_sigma = Ppo[k-1] + np.identity(state_num) * 0.0001
    Sigma_zz = linalg.block_diag(Ppr_sigma, Q)
    # print(Sigma_zz)
    L = linalg.cholesky(Sigma_zz, lower = True)
    # get sigma points
    Z_sp, sp_num, L_num = getSigmaP(Xpo[:,k-1], L, kappa, input_dim)
    # extract sigma points into states and input motion noise
    sp_X = Z_sp[0:state_num, :]
    sp_w = Z_sp[state_num:, :]
    # Make prediction: send sigma points into nonlinear motion model
    # inputs: v, om
    sp_Xpr = np.zeros_like(sp_X)
    for idx in range(sp_num):
        A = np.array([[T * math.cos(sp_X[2, idx]), 0],
                      [T * math.sin(sp_X[2, idx]), 0],
                      [0,                          T] ])
        V = np.array([[v[k-1,0]],[om[k-1,0]]]) + sp_w[:,idx].reshape(-1,1)
        sp_Xpr[:,idx] = sp_X[:,idx] + np.squeeze(A.dot(V))
        
    # get Xpr[k]  
    for idx in range(sp_num):
        alpha = getAlpha(idx, L_num, kappa)
        Xpr[:,k] = Xpr[:,k] + alpha * sp_Xpr[:,idx] 

    # get Ppr[k]
    for idx in range(sp_num):
        alpha = getAlpha(idx, L_num, kappa)
        delat_X = (sp_Xpr[:,idx] - Xpr[:,k]).reshape(-1,1)      # column vector
        Ppr[k] = Ppr[k] + alpha * delat_X.dot(delat_X.transpose())
    # Enforce symmetry
    Ppr[k] = 0.5*(Ppr[k] + Ppr[k].transpose())  

    # If no measurement data, take our posterior estimates as the prior estimates
    if np.all(r_meas[k,:] == 0):
        Xpo[:,k] = Xpr[:,k]
        Ppo[k] = Ppr[k]
    # If there are measurements,
    # Correction: update the states with measurements    
    else:
        landmark_idx = np.nonzero(r_meas[k,:])
        landmark_idx = np.asarray(landmark_idx).reshape(1,-1)
        R_num = np.count_nonzero(r_meas[k,:],axis=0)
        # for R_num measurements at time step k
        R_k = np.kron(np.eye(R_num,dtype=int),R)
        Ppo_sigma = Ppr[k] + np.identity(state_num) * 0.0001
        Sigma_ZZ = linalg.block_diag(Ppo_sigma, R_k)
        L_update = linalg.cholesky(Sigma_ZZ, lower=True)
        # get sigma points
        Z_sp, sp_num, L_num = getSigmaP(Xpr[:,k], L_update, kappa, R_num*2)
        # extract sigma points into state and motion noise
        sp_X = Z_sp[0:state_num, :] 
        sp_w = Z_sp[state_num:, :]
        # Correction: send sigma points into nonlinear measurement model
        # yk_sigma = g(xk_sigma, nk_sigma)
        sp_y = np.zeros((2*R_num, sp_num))
        # sp_y = [r1_sigma0,    r1_sigma1,   ... ,  r1_sigma_sp_num
        #         th_1_sigma0,  th_1_sigma1, ... ,  th_1_sigma_sp_num
        #         :
        #         rn_sigma0,    rn_sigma1,   ... ,  rn_sigma_sp_num  
        #         th_n_sigma0,  th_n_sigma1, ... ,  th_n_sigma_sp_num  ]
        for c_idx in range(sp_num):   # 33
            # row number
            row_num = 0
            for r_idx in range(R_num): 
                # get the landmark x and y 
                landmark_x = l[landmark_idx[0,r_idx], 0]
                landmark_y = l[landmark_idx[0,r_idx], 1]
                theta_sp_k = sp_X[2,c_idx]
                sp_y[row_num,   c_idx] = np.sqrt((landmark_x - sp_X[0,c_idx] - d*math.cos(theta_sp_k))**2 + (landmark_y - sp_X[1,c_idx] - d*math.sin(theta_sp_k))**2) + sp_w[row_num, c_idx]
                
                y_tan = landmark_y - sp_X[1,c_idx] - d*math.sin(theta_sp_k)
                x_tan = landmark_x - sp_X[0,c_idx] - d*math.cos(theta_sp_k) 
                sp_y[row_num+1, c_idx] = np.arctan2(y_tan, x_tan) - theta_sp_k+ sp_w[row_num+1, c_idx]
                row_num = row_num+2
                    
        # get mu_yk
        mu_yk = np.zeros((2*R_num,1))
        for idx in range(sp_num):
            alpha = getAlpha(idx, L_num, kappa)
            mu_yk = mu_yk + alpha * sp_y[:,idx].reshape(-1,1)
        # get Sigma_yy_k and Sigma_xy_k
        Sigma_yy_k = np.zeros((2*R_num,   2*R_num))
        Sigma_xy_k = np.zeros((state_num, 2*R_num))
        for idx in range(sp_num):
            alpha = getAlpha(idx, L_num, kappa)       
            delat_y = sp_y[:,idx].reshape(-1,1) - mu_yk
            delat_x = (sp_X[:,idx] - Xpr[:,k]).reshape(-1,1)
            Sigma_yy_k = Sigma_yy_k + alpha * delat_y.dot(delat_y.transpose())
            Sigma_xy_k = Sigma_xy_k + alpha * delat_x.dot(delat_y.transpose())  

        # Update Xpo[k] and Ppo[k]
        # Calculate Kalman Gain 
        Kk = Sigma_xy_k.dot(linalg.inv(Sigma_yy_k))
        Ppo[k] = Ppr[k] - Kk.dot(Sigma_xy_k.transpose())
        # Enforce symmetry of covariance matrix
        Ppo[k] = 0.5 * (Ppo[k] + Ppo[k].transpose())
        # innovation error term
        y_meas_k = []
        for idx in range(R_num):
            meas_Rnum = [r_meas[k,landmark_idx[0,idx]],b_meas[k,landmark_idx[0,idx]]]
            y_meas_k.append(meas_Rnum)
        # get the measurements in column vector
        y_meas_k = np.asarray(y_meas_k).reshape(-1,1)
        # print(y_meas_k)
        err_yk = y_meas_k - mu_yk
        # wrap to Pi
        for idx in range(err_yk.shape[0]):
            if (idx+1)%2 == 0:     #idx+1: 1,...,14
                err_yk[idx,0] = wrapToPi(err_yk[idx,0])
        Xpo[:,k] = Xpr[:,k] + np.squeeze(Kk.dot(err_yk.reshape(-1,1)))
        
print('State Estimation Finished.')  
plot_2d(t,Xpo,Ppo,vicon_gt)
    
plt.show()
        
    



















