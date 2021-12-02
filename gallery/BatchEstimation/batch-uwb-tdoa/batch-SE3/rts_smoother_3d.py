'''
RTS smoother implementation for batch estimation using SO3 
'''
import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag
# from pytransform3d.rotations import axis_angle_from_matrix
from so3_util import getTrans, getTrans_in, skew, axisAngle_from_rot

np.set_printoptions(precision=4)

class RTS_Smoother_3D:
    '''initialization'''
    def __init__(self, robot, K):
        self.pert_pr_f = np.zeros((K, 6))
        self.pert_po_f = np.zeros((K, 6))
        self.pert_po   = np.zeros((K, 6))              # posterior of the perturbation
        self.Ppr_f     = np.zeros((K, 6, 6))
        self.Ppo_f     = np.zeros((K, 6, 6))
        self.Ppo       = np.zeros((K, 6, 6))
        # init. X0 = np.zeros((1,6))
        # save the forward Jacobian (for backward pass)
        self.F = np.zeros((K, 6, 6))

        self.robot = robot
        self.Kmax = K

    '''compute ev_k'''
    def compute_ev_k(self, dt, v_k_1, w_k_1, C_op_k1, r_op_k1, C_op_k, r_op_k):
        # compute ln(...)^{v}
        input = np.array([-v_k_1, -w_k_1]).reshape(-1,1)
        rho = input[0:3].reshape(-1,1)
        phi = input[3:6]
        phi_skew = skew(phi)
        zeta = np.block([
            [phi_skew, rho],
            [0,  0,  0,  0]
            ])
        Psi = linalg.expm(dt*zeta)
        T = getTrans(C_op_k1, r_op_k1)
        T_in = getTrans_in(C_op_k, r_op_k)
        tau = Psi.dot(T).dot(T_in)
        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        # Equ. (60) to compute J. Then, J * rho = r
        # could be done by eigenvalue
        # axisAngle = [x, y, z, angle]
        #axisAngle = axis_angle_from_matrix(C_now)      # this package is not stable
        axisAngle = axisAngle_from_rot(C_now)
        axisAngle = np.squeeze(axisAngle)

        if(axisAngle[3] == 0):
            a = axisAngle[0:3]
            J = np.eye(3)
        else:
            a = axisAngle[0:3]
            term1 = math.sin(axisAngle[3]) / axisAngle[3]
            term2 = (1.0 - math.cos(axisAngle[3])) / axisAngle[3]
            a_skew = skew(a)
            a_vec = a.reshape(-1,1)
            J = term1 * np.eye(3) + (1-term1)*a_vec.dot(a_vec.T) + term2 * a_skew

        # J * rho_now = r_now
        r_now = r_now.reshape(-1,1)
        rho_now = np.linalg.solve(J, r_now)      
        phi_now = axisAngle[3] * a

        rho_now = np.squeeze(rho_now)
        phi_now = np.squeeze(phi_now)
        return rho_now, phi_now

    '''compute ey_k'''
    def compute_ey_k(self, y_k, C_op_k, r_op_k):
        # get the meas. from meas. model
        y_uwb = self.robot.meas_model(y_k, C_op_k, r_op_k)
        # compute error in y (one element)
        err_y = y_k[2] - y_uwb
        return err_y


    '''forward pass'''
    def forward(self, X0, P0, C_op, r_op, v_data, w_data, y_data, t):
        # v_data and w_data: k x 3 dimension
        # init. state and covariance
        self.pert_pr_f[0,:] = np.squeeze(X0)
        self.Ppr_f[0,:,:] = P0
        # loop over all timesteps
        for k in range(self.Kmax):
            if k == 0:
                C_op_k = C_op[0,:,:];     r_op_k = r_op[0,:]
            else:
                # operating point
                C_op_k = C_op[k,:,:];     r_op_k  = r_op[k,:]      # timestep k
                C_op_k1 = C_op[k-1,:,:];  r_op_k1 = r_op[k-1,:]    # timestep k-1
                # input data
                v_data_k = v_data[k,:];   w_data_k = w_data[k,:]   # input at timestep k
                dt = t[k] - t[k-1]
                Qv_k = self.robot.nv_prop(dt)
                # compute ev_k 
                rho, phi = self.compute_ev_k(dt, v_data_k, w_data_k, C_op_k1, r_op_k1, C_op_k, r_op_k)
                # compute F_k
                F_k = self.robot.compute_F(C_op_k1, r_op_k1, C_op_k, r_op_k)
                self.F[k-1,:,:] = F_k

            # measurements
            G = np.empty((0,6));    ey_k = np.empty(0);    Rk_y = np.empty((0,0))
            # If data is not sync., check if we have meas
            # compute the meas. Jacobian 
            G = self.robot.compute_G(y_data[k,:], C_op_k, r_op_k)
            # receive one UWB meas. at one timestamp
            ey_k = self.compute_ey_k(y_data[k,:], C_op_k, r_op_k)
            # meas. cov
            Rk_y = self.robot.Rm

            # forward equations
            if G.size == 0:  # no valid meas. 
                if k!=0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k.dot(self.Ppo_f[k-1,:,:]).dot(F_k.T) + Qv_k
                    # equ. 2
                    pert_pr_f_k = F_k.dot(self.pert_po_f[k-1,:].reshape(-1,1)) + np.block([rho, phi]).reshape(-1,1)
                    self.pert_pr_f[k,:] = np.squeeze(pert_pr_f_k) 

                # no meas. to update
                self.Ppo_f[k,:,:] = self.Ppr_f[k,:,:]
                self.pert_po_f[k,:] = self.pert_pr_f[k,:]
            else:     
                # update with measurements
                if k!=0:
                    # equ. 1
                    self.Ppr_f[k,:,:] = F_k.dot(self.Ppo_f[k-1,:,:]).dot(F_k.T) + Qv_k
                    # equ. 2
                    pert_pr_f_k = F_k.dot(self.pert_po_f[k-1,:].reshape(-1,1)) + np.block([rho, phi]).reshape(-1,1)
                    self.pert_pr_f[k,:] = np.squeeze(pert_pr_f_k) 

                # equ. 3
                GM = G.dot(self.Ppr_f[k,:,:]).dot(G.T) + Rk_y
                K_k = self.Ppr_f[k,:,:].dot(G.T).dot(linalg.inv(GM))
                # equ. 4
                self.Ppo_f[k,:,:] = (np.eye(6) - K_k.dot(G)).dot(self.Ppr_f[k,:,:])
                # equ. 5
                in_err = ey_k.reshape(-1,1) - G.dot(self.pert_pr_f[k,:].reshape(-1,1))
                pert_hat = self.pert_pr_f[k,:].reshape(-1,1) + K_k.dot(in_err)
                self.pert_po_f[k,:] = np.squeeze(pert_hat)


    '''backward pass'''
    def backward(self):
        for k in range(self.Kmax-1, 0, -1):    # k = K-1 ~ 1
            # init
            if k == self.Kmax-1:
                self.pert_po[k,:] = self.pert_po_f[k,:]
                pert_hat = self.pert_po_f[k,:].reshape(-1,1)
                self.Ppo[k,:,:] = self.Ppo_f[k,:,:]
            else:
                # pert_po[k,:] computed from last iteration
                pert_hat = self.pert_po[k,:].reshape(-1,1)

            d_pert = pert_hat - self.pert_pr_f[k,:].reshape(-1,1)
            PAP = self.Ppo_f[k-1,:,:].dot(self.F[k-1,:,:].T).dot(linalg.inv(self.Ppr_f[k,:,:]))
            d_pert_hat = self.pert_po_f[k-1,:].reshape(-1,1) + PAP.dot(d_pert)
            self.pert_po[k-1,:] = np.squeeze(d_pert_hat)
            # compute the final Ppo. (Barfoot book Appendix A.3.2)
            self.Ppo[k-1,:,:] = self.Ppo_f[k-1,:,:] + PAP.dot(self.Ppo[k,:,:] - self.Ppr_f[k,:,:]).dot(PAP.T)

