'''
RTS smoother implementation for batch estimation cf
'''
import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy import linalg
import math
from scipy.linalg import block_diag
from pytransform3d.rotations import axis_angle_from_matrix
from rot_util import skew, zeta

np.set_printoptions(precision=4)

class RTS_Smoother:
    '''initialization'''
    def __init__(self, robot, K):
        self.pert_pr_f = np.zeros((K, 9))    # perturb(error states): dx, dy, dz, dvx, dvy, dvz, droll, dpitch, dyaw
        self.pert_po_f = np.zeros((K, 9))
        self.pert_po   = np.zeros((K, 9))
        self.Ppr_f     = np.zeros((K, 9, 9))
        self.Ppo_f     = np.zeros((K, 9, 9))
        self.Ppo       = np.zeros((K, 9, 9))
        # save the forward Jacobian (for backward pass)
        self.F = np.zeros((K, 9, 9))
        self.Kmax = K

        self.robot = robot

    '''compute ev_k'''
    def compute_ev_k(self, X_op_k1, X_op_k, dt, v_k, w_k):
        # compute x_check using robot motion model
        # {v_k, w_k} is the input at timestamp k
        # X_op_k1 is the opertating point at timestamp k-1
        x_check = self.robot.motion_model(X_op_k1, v_k, w_k, dt)
        # compute the error 
        dp = x_check[0:3] - X_op_k[0:3]
        dv = x_check[3:6] - X_op_k[3:6]
        # compute error in angle using quaternion (careful about the manifold) 
        # q = [qw, qx, qy, qz]
        q_op_k = Quaternion([X_op_k[6], X_op_k[7], X_op_k[8], X_op_k[9]])
        q_op_k_inv = q_op_k.inverse
        q_k_check = Quaternion([x_check[6], x_check[7], x_check[8], x_check[9]])
        # SO3 X SO3 -> R^3, Log(q*q^{-1}),  equation (161) in eskf
        # consider the direction here: q_op_k_inv * q_k_check = dq_k --> q_k_check = q_op_k * dq_k
        
        d_rpy = 2 * q_op_k_inv * q_k_check                         # 2 * (q_k_check * q_op_k_inv)[1:4]
        d_theta = np.array([d_rpy[1], d_rpy[2], d_rpy[3]])

        ev_k = np.block([dp, dv, d_theta]).reshape(-1,1)       # error in perturbation (error state)

        return ev_k

    '''compute ey_k'''
    def compute_ey_k(self, y_k, X_op_k):
        # get the meas. from robot meas. model
        y_uwb = self.robot.meas_model(y_k, X_op_k)   # y_k has the anchor id
        # compute error in y (one element)
        err_y = y_k[2] - y_uwb

        return err_y

    '''forward pass'''
    def forward(self, pert_x0, P0, X_op, v_data, w_data, y_data, t):
        # v_data and w_data: K x 3 dimension
        # X_op: K x 10
        # init. state and covariance
        self.pert_pr_f[0,:] = np.squeeze(pert_x0)
        self.Ppr_f[0,:,:] = P0
        # loop over all timestamps
        for k in range(self.Kmax):
            if k == 0:
                X_op_k = X_op[0,:]
            else:
                # operating point
                X_op_k  = X_op[k,:]     # timestamp k
                X_op_k1 = X_op[k-1,:]   # timestamp k-1
                # input data at timestamp k
                v_data_k = v_data[k,:];  w_data_k = w_data[k,:] # input at timestamp k
                dt = t[k] - t[k-1]
                Qv_k = self.robot.nv_prop(dt)
                # compute ev_k
                ev_k = self.compute_ev_k(X_op_k1, X_op_k, dt, v_data_k, w_data_k)
                # print("k = {0}, ev_k = {1}\n".format(k, ev_k))

                # compute F_k
                F_k = self.robot.compute_F(X_op_k1, v_data_k, w_data_k, dt)
                self.F[k-1, :, :] = F_k

            # measurements
            G = np.empty((0,9));  ey_k = np.empty(0);   Rk_y = np.empty((0,0))

            # If data is not sync., check if we have meas. (see batch-3d)
            # compute the meas. Jacobian
            G = self.robot.compute_G(y_data[k,:], X_op_k)
            # receive one UWB meas. at one timestamp
            ey_k = self.compute_ey_k(y_data[k,:], X_op_k)
            # print("k = {0}, ey_k = {1}\n".format(k, ey_k))

            # meas. cov
            Rk_y = self.robot.Rm

            # forward equations
            if G.size == 0:      # no valid meas. 
                if k != 0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k.dot(self.Ppo_f[k-1,:,:]).dot(F_k.T) + Qv_k
                    # equ. 2
                    pert_pr_f_k = F_k.dot(self.pert_po_f[k-1,:].reshape(-1,1)) + ev_k.reshape(-1,1)
                    self.pert_pr_f[k,:] = np.squeeze(pert_pr_f_k) 

                # no meas. to update
                self.Ppo_f[k,:,:] = self.Ppr_f[k,:,:]
                self.pert_po_f[k,:] = self.pert_pr_f[k,:]

            else:
                # update with measurements
                if k != 0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k.dot(self.Ppo_f[k-1,:,:]).dot(F_k.T) + Qv_k
                    # equ. 2
                    pert_pr_f_k = F_k.dot(self.pert_po_f[k-1,:].reshape(-1,1)) + ev_k.reshape(-1,1)
                    self.pert_pr_f[k,:] = np.squeeze(pert_pr_f_k) 

                # equ. 3
                GM = G.dot(self.Ppr_f[k,:,:]).dot(G.T) + Rk_y
                K_k = self.Ppr_f[k,:,:].dot(G.T).dot(linalg.inv(GM))
                # equ. 4
                self.Ppo_f[k,:,:] = (np.eye(9) - K_k.dot(G)).dot(self.Ppr_f[k,:,:])
                # equ. 5
                in_err = ey_k.reshape(-1,1) - G.dot(self.pert_pr_f[k,:].reshape(-1,1))
                pert_hat = self.pert_pr_f[k,:].reshape(-1,1) + K_k.dot(in_err)
                self.pert_po_f[k,:] = np.squeeze(pert_hat)


    '''bacekward pass'''
    def backward(self):
        for k in range(self.Kmax-1, 0, -1):     #k = K-1 ~ 1
            # init
            if k == self.Kmax-1:
                self.pert_po[k,:] = self.pert_po_f[k,:]
                pert_hat = self.pert_po_f[k,:].reshape(-1,1)
                self.Ppo[k,:,:] = self.Ppo_f[k,:,:]
            else:
                # pert_po[k,:] compute from last iteration
                pert_hat = self.pert_po[k,:].reshape(-1,1)

            d_pert = pert_hat - self.pert_pr_f[k,:].reshape(-1,1)
            PAP = self.Ppo_f[k-1,:,:].dot(self.F[k-1,:,:].T).dot(linalg.inv(self.Ppr_f[k,:,:]))
            d_pert_hat = self.pert_po_f[k-1,:].reshape(-1,1) + PAP.dot(d_pert)
            self.pert_po[k-1,:] = np.squeeze(d_pert_hat)
            # compute the final Ppo. (Barfoot book Appendix A.3.2)
            self.Ppo[k-1,:,:] = self.Ppo_f[k-1,:,:] + PAP.dot(self.Ppo[k,:,:] - self.Ppr_f[k,:,:]).dot(PAP.T)
