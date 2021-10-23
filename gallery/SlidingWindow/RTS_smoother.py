'''
RTS smoother implementation for batch estimation
'''
import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag
from util import wrapToPi

class RTS_Smoother_2D:
    '''initialization'''
    def __init__(self, P0, robot, K):
        # in nonlinear batch estimation,
        # we use GN to compute dx and update to the operating points until convergence  
        self.dXpr_f = np.zeros((K, 3))
        self.dXpo_f = np.zeros((K, 3))
        self.dXpo   = np.zeros((K, 3))
        self.Ppr_f  = np.zeros((K, 3, 3))
        self.Ppo_f  = np.zeros((K, 3, 3))
        self.Ppo    = np.zeros((K, 3, 3))          # final posterior covariance

        self.robot = robot
        # initial state and covariance
        self.Ppr_f[0,:,:] = P0
        # save the forward Jacobian (will be used for the backward pass)
        self.F = np.zeros((K, 3, 3))
        self.Kmax = K

    '''compute ev_k'''
    def compute_ev_k(self, x_op_k1, x_op_k, v, om):
        x_pro = self.robot.motion_model(x_op_k1, v, om)  # f(x_op,k-1, v_k, 0)
        ev_k = x_pro - x_op_k.reshape(-1,1)                 # f(x_op,k-1, v_k, 0) - x_op,k
        ev_k[2] = wrapToPi(ev_k[2])
        return ev_k

    '''forward pass'''
    def forward(self, x_op, v, om, r_meas, b_meas):
        # loop over all timestamp
        # note: careful about k = 0
        for k in range(self.Kmax):               # k = 0 ~ K-1
            if k == 0:
                # compute Ppr_0,f and dXpr[0,:]
                r_k = r_meas[0,:]             # y(0)
                b_k = b_meas[0,:]
                x_op_k = x_op[0 : 3]
            else:
                r_k = r_meas[k,:]             # y(0)
                b_k = b_meas[k,:]
                # operting point 
                x_op_k1 = x_op[3*k-3 : 3*k]       # x_op(k-1)
                x_op_k = x_op[3*k : 3*k+3]        # x_op(k)

                v_k = v[k];   om_k = om[k]
                # compute ev_k
                ev_k = self.compute_ev_k(x_op_k1, x_op_k, v_k, om_k)
                # compute F_k-1
                F_k1 = self.robot.compute_F(x_op_k1, v_k)
                # save motion Jacobian
                self.F[k-1,:,:] = F_k1
                # compute Qv_k
                Qv_k = self.robot.nv_prop(x_op_k1)       # Q'_k = w_c(x_op_k1) * self.Qi * w_c(x_op_k1)

            # measurements
            l_k = np.nonzero(r_k);    l_k = np.asarray(l_k).reshape(-1,1)
            M = np.count_nonzero(r_k, axis=0)               # at timestamp k, we have M meas. in total
            # measurements
            # check if we have meas. or not
            ey_k = np.empty(0);       Rk_y  = np.empty((0,0))
            # flag for measurement update
            update = True
            if M: 
                G = np.empty((0,3))                         # empty G to contain all the Gs in timestep k
                for m in range(M): 
                    l_xy = self.robot.landmark[l_k[m,0], :]       # the m-th landmark pos.
                    r_l = r_k[l_k[m,0]]                     # [timestamp, id of landmark]
                    b_l = b_k[l_k[m,0]]
                    y = np.array([r_l, b_l]).reshape(-1,1)
                    ey = y - self.robot.meas_model(x_op_k, l_xy)
                    # compute angle error, wrap to pi
                    ey[1] = wrapToPi(ey[1])
                    '''compute ey_k'''
                    ey_k = np.append(ey_k, np.squeeze(ey))
                    # --------- save the variance of meas. ------------ #
                    Rk_y = block_diag(Rk_y, self.robot.Rm)
                    # compute G
                    G_i = self.robot.compute_G(x_op_k, l_xy)
                    G = np.concatenate((G, G_i), axis=0)
            else:
                # when no measurements, use Xpr as Xpo
                update = False

            # forward equations
            if update:
                if k!=0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k1.dot(self.Ppo_f[k-1,:,:]).dot(F_k1.T) + Qv_k
                    # equ. 2
                    dx_check = F_k1.dot(self.dXpo_f[k-1,:].reshape(-1,1)) + ev_k
                    # wrap to pi
                    dx_check[2,0] = wrapToPi(dx_check[2,0])    # dx_check[2,0] is delta th
                    self.dXpr_f[k,:] = np.squeeze(dx_check)

                # equ. 3
                GM = G.dot(self.Ppr_f[k,:,:]).dot(G.T) + Rk_y 
                K_k = self.Ppr_f[k,:,:].dot(G.T).dot(linalg.inv(GM))
                # equ. 4
                self.Ppo_f[k,:,:] = (np.eye(3) - K_k.dot(G)).dot(self.Ppr_f[k,:,:])
                # equ. 5
                in_err = ey_k.reshape(-1,1) - G.dot(self.dXpr_f[k,:].reshape(-1,1))   # innovation error
                # wrap to pi
                for idx in range(in_err.shape[0]):
                    if (idx%2):
                        in_err[idx,0] = wrapToPi(in_err[idx,0])
                dx_hat = self.dXpr_f[k,:].reshape(-1,1) + K_k.dot(in_err)
                # wrap to pi
                dx_hat[2,0] = wrapToPi(dx_hat[2,0])    # dx_hat[2,0] is delta th
                self.dXpo_f[k,:] = np.squeeze(dx_hat)

            else:
                if k!=0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k1.dot(self.Ppo_f[k-1,:,:]).dot(F_k1.T) + Qv_k
                    # equ. 2
                    dx_check = F_k1.dot(self.dXpo_f[k-1,:].reshape(-1,1)) + ev_k
                    # wrap to pi
                    dx_check[2,0] = wrapToPi(dx_check[2,0])    # dx_check[2,0] is delta th
                    self.dXpr_f[k,:] = np.squeeze(dx_check)

                # no meas. to update
                self.Ppo_f[k,:,:] = self.Ppr_f[k,:,:]
                self.dXpo_f[k,:]  = self.dXpr_f[k,:]

    '''backward pass'''
    def backward(self):
        for k in range(self.Kmax-1, 0, -1):   # k = K-1 ~ 1
            if k == self.Kmax -1:   # the last idx is K-1
                # initialize
                self.dXpo[k,:] = self.dXpo_f[k,:]
                xk_hat = self.dXpo_f[k,:].reshape(-1,1)

                self.Ppo[k,:,:] = self.Ppo_f[k,:,:]
            else:
                # dXpo[k,:] computed from last iteration
                xk_hat = self.dXpo[k,:].reshape(-1,1)

            dx = xk_hat - self.dXpr_f[k,:].reshape(-1,1)
            dx[2,0] = wrapToPi(dx[2,0])

            PAP = self.Ppo_f[k-1,:,:].dot(self.F[k-1,:,:].T).dot(linalg.inv(self.Ppr_f[k,:,:]))

            dx_hat = self.dXpo_f[k-1,:].reshape(-1,1) + PAP.dot(dx)
            dx_hat[2,0] = wrapToPi(dx_hat[2,0])
            self.dXpo[k-1,:] = np.squeeze(dx_hat)

            # compute the final Ppo, Barfoot book Appendix A.3.2
            self.Ppo[k-1,:,:] = self.Ppo_f[k-1,:,:] + PAP.dot(self.Ppo[k,:,:] - self.Ppr_f[k,:,:]).dot(PAP.T)