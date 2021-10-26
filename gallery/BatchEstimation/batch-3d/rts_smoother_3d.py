'''
RTS smoother implementation for batch estimation using SO3 
'''
import numpy as np
from numpy.core.shape_base import block
from scipy import linalg
import math
from scipy.linalg import block_diag
from pytransform3d.rotations import axis_angle_from_matrix
from so3_util import getTrans, getTrans_in, skew, circle

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
    def compute_ev_k(self, dt, v_k_1, w_k_1, C_op_k, r_op_k, C_op_k_1, r_op_k_1):
        input = np.array([-v_k_1, -w_k_1]).reshape(-1,1)
        rho = input[0:3].reshape(-1,1)
        phi = input[3:6]
        phi_skew = skew(phi)
        zeta = np.block([
            [phi_skew, rho],
            [0,  0,  0,  0]
            ])
        Psi = linalg.expm(dt*zeta)
        T = getTrans(C_op_k, r_op_k)
        T_in = getTrans_in(C_op_k_1, r_op_k_1)
        tau = Psi.dot(T).dot(T_in)
        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        # could be done by eigenvalue
        axisAngle = axis_angle_from_matrix(C_now)
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


    def compute_ey_k(self, y_k_j, P_opt):
        P_opt = np.squeeze(P_opt)
        vec = np.array([
            self.robot.fu*P_opt[0],
            self.robot.fv*P_opt[1],
            self.robot.fu*(P_opt[0] - self.robot.b),
            self.robot.fv*P_opt[1]
        ]).reshape(-1,1)

        vec_c = np.array([self.robot.cu, 
                          self.robot.cv, 
                          self.robot.cu, 
                          self.robot.cv]).reshape(-1,1)
        y_o = (1.0/P_opt[2]) * vec + vec_c
        err_y = y_k_j.reshape(-1,1) - y_o
        return err_y

    '''forward pass'''
    def forward(self, X0, P0, C_op, r_op, v_data, w_data, y_data, t):
        # init. state and covariance
        self.pert_pr_f[0,:] = np.squeeze(X0)
        self.Ppr_f[0,:,:] = P0
        for k in range(self.Kmax-1):            # k = 0 ~ K-2
            # init
            G = np.empty((0,6));  ey_k = np.empty(0);  Rk_y = np.empty((0,0))

            dt = t[k+1] - t[k]
            Qv_k = self.robot.nv_prop(dt)

            C_op_k = C_op[k,:,:];     r_op_k = r_op[k,:]
            C_op_k_1 = C_op[k+1,:,:]; r_op_k_1 = r_op[k+1,:]

            F_k = self.robot.compute_F(C_op_k_1, r_op_k_1, C_op_k, r_op_k)
            self.F[k,:,:] = F_k
            # compute ev_k
            rho, phi = self.compute_ev_k(dt, v_data[:,k+1], w_data[:,k+1], C_op_k, r_op_k, C_op_k_1, r_op_k_1)

            for j in range(20):
                cond = np.sum(y_data[:,k,j] == -1)    # if meas. invalid, all elements are -1, cond = 4
                if (cond == 0):
                    P_opt = self.robot.compute_point(C_op_k, r_op_k, j)
                    G_i = self.robot.compute_G(C_op_k, r_op_k, j, P_opt)

                    e_y_i = self.compute_ey_k(y_data[:,k,j], P_opt)

                    ey_k = np.append(ey_k, np.squeeze(e_y_i))

                    Rk_y = block_diag(Rk_y, self.robot.Rm)

                    G = np.concatenate((G, G_i), axis=0)

            if G.size == 0:  # no valid meas.
                self.Ppo_f[k,:,:] = self.Ppr_f[k,:,:]
                self.pert_po_f[k,:] = self.pert_pr_f[k,:]

                pert_pr_f_k_1 = F_k.dot(self.pert_po_f[k,:].reshape(-1,1)) + np.block([rho, phi]).reshape(-1,1)
                self.pert_pr_f[k+1,:] = np.squeeze(pert_pr_f_k_1) 
                self.Ppr_f[k+1,:,:] = F_k.dot(self.Ppo_f[k,:,:]).dot(F_k.T) + Qv_k
            else:
                GM = G.dot(self.Ppr_f[k,:,:]).dot(G.T) + Rk_y
                K_k = self.Ppr_f[k,:,:].dot(G.T).dot(linalg.inv(GM))
                self.Ppo_f[k,:,:] = (np.eye(6) - K_k.dot(G)).dot(self.Ppr_f[k,:,:])
                
                in_err = ey_k.reshape(-1,1) - G.dot(self.pert_pr_f[k,:].reshape(-1,1))
                pert_hat = self.pert_pr_f[k,:].reshape(-1,1) + K_k.dot(in_err)
                self.pert_po_f[k,:] = np.squeeze(pert_hat)

                pert_pr_f_k_1 = F_k.dot(self.pert_po_f[k,:].reshape(-1,1)) + np.block([rho, phi]).reshape(-1,1)
                self.pert_pr_f[k+1,:] = np.squeeze(pert_pr_f_k_1) 
                self.Ppr_f[k+1,:,:] = F_k.dot(self.Ppo_f[k,:,:]).dot(F_k.T) + Qv_k

        # the last one
        # init
        G = np.empty((0,6));  ey_k = np.empty(0);  Rk_y = np.empty((0,0))
        k_end = self.Kmax-1
        for j in range(20):
            cond = np.sum(y_data[:,k_end,j] == -1)   # if meas. invalid, all elements are -1, cond = 4
            if (cond == 0):
                P_opt = self.robot.compute_point(C_op_k, r_op_k, j)
                G_i = self.robot.compute_G(C_op_k, r_op_k, j, P_opt)

                e_y_i = self.compute_ey_k(y_data[:,k_end,j], P_opt)

                ey_k = np.append(ey_k, np.squeeze(e_y_i))

                Rk_y = block_diag(Rk_y, self.robot.Rm)

                G = np.concatenate((G, G_i), axis=0)


        if G.size == 0:  # no valid meas.
            self.Ppo_f[k_end,:,:] = self.Ppr_f[k_end,:,:]
            self.pert_po_f[k_end,:] = self.pert_pr_f[k_end,:]
        else:
            GM = G.dot(self.Ppr_f[k_end,:,:]).dot(G.T) + Rk_y
            K_k = self.Ppr_f[k_end,:,:].dot(G.T).dot(linalg.inv(GM))
            self.Ppo_f[k_end,:,:] = (np.eye(6) - K_k.dot(G)).dot(self.Ppr_f[k_end,:,:])
            
            in_err = ey_k.reshape(-1,1) - G.dot(self.pert_pr_f[k_end,:].reshape(-1,1))
            pert_hat = self.pert_pr_f[k_end,:].reshape(-1,1) + K_k.dot(in_err)
            self.pert_po_f[k_end,:] = np.squeeze(pert_hat)

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





