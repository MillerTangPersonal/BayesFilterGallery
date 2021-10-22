'''
A class for the batch estimation in 2D
'''
from util import wrapToPi
import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag

class Batch_2D:
    '''initialization'''
    def __init__(self, P0, robot, K):
        self.P0_check = P0     # init covariance
        self.robot = robot     # robot dynamics
        self.Kmax = K          # for construct A, b matrix

    '''compute ev_k'''
    def compute_ev_k(self, x_op_k1, x_op_k, v, om):
        x_pro = self.robot.motion_model(x_op_k1, v, om)  # f(x_op,k-1, v_k, 0)
        ev_k = x_pro - x_op_k.reshape(-1,1)           # f(x_op,k-1, v_k, 0) - x_op,k
        ev_k[2] = wrapToPi(ev_k[2])
        return ev_k

    '''construct the Ax=b '''
    def construct_A_b(self, x_op, x_check, v, om, r_meas, b_meas, dt):
        # known variables:
        # x_check, self.P0_check, x_op, y_k (measurements), v_k, om_k (inputs), dt
        # compute at operating points
        # e_{v,k}, e_{y,k}, -Fk, -Gk, Qk', Rk'

        # init. to save the error e_{v,k} and e_{y,k}
        e_v = np.empty(0);      e_y = np.empty(0)
        # save the covariance matrix
        W_v = np.empty((0,0));  W_y = np.empty((0,0))

        # construct H matrix
        H_U = np.eye(3*(self.Kmax))              # the upper matrix has a fixed dimension
        H_L = np.empty((0,0))                        # the lower matrix changes in dimension.

        e_v_0 = x_op[0:3].reshape(-1,1) - x_check.reshape(-1,1)
        e_v = np.append(e_v, np.squeeze(e_v_0))
        W_v = block_diag(W_v, self.P0_check)

        # compute -F_0 ~ -F_k-1, Q1' ~ QK', e_v_0 ~ e_v_K 
        for k in range(1, self.Kmax):
            # operting point 
            x_op_k1 = x_op[3*k-3 : 3*k]       # x_op(k-1)
            x_op_k = x_op[3*k : 3*k+3]        # x_op(k)
            # input at timestamp k
            v_k = v[k];  om_k = om[k]
            
            # compute Qk' 
            Q_k = self.robot.nv_prop(x_op_k1)
            Q_k = Q_k + np.eye(3) * 1e-10              # avoid numerical prob. 

            # compute e_k_v
            e_v_k = self.compute_ev_k(x_op_k1, x_op_k, v_k, om_k)

            e_v = np.append(e_v, np.squeeze(e_v_k))
            W_v = block_diag(W_v, Q_k)

            # --------------- construct H_U matrix ------------------------ #
            H_U[(3*k):(3+3*k), (3*k-3):(3*k)] = -1.0 * self.robot.compute_F(x_op_k1, v_k)  # get F_{0,1,2}
           
        # compute G0 ~ Gk, R0 ~ Rk, e_y_1 ~ e_y_k
        for k in range(self.Kmax):
            # operating point
            x_op_k = x_op[3*k : 3*k+3]
            # measurements at timestamp k
            r_k = r_meas[k,:];        b_k = b_meas[k,:]
            l_k = np.nonzero(r_k);    l_k = np.asarray(l_k).reshape(-1,1)
            M = np.count_nonzero(r_k, axis=0)        # at timestamp k, we have M meas. in total
            # check if we have meas. or not

            if M:
                G = np.empty((0,3))
                e_y_k = np.empty(0); 
                for m in range(M):
                    l_xy = self.robot.landmark[l_k[m,0], :]
                    r_l = r_k[l_k[m,0]]
                    b_l = b_k[l_k[m,0]]
                    y = np.array([r_l, b_l]).reshape(-1,1)
                    ey_m = y - self.robot.meas_model(x_op_k, l_xy)
                    # wrap to Pi
                    ey_m[1] = wrapToPi(ey_m[1])

                    e_y_k = np.append(e_y_k, np.squeeze(ey_m))
                    W_y = block_diag(W_y, self.robot.Rm)
                    
                    # compute G
                    G_i = self.robot.compute_G(x_op_k, l_xy)
                    G = np.concatenate((G, G_i), axis=0)

                e_y = np.append(e_y, np.squeeze(e_y_k))
                H_L = block_diag(H_L, G)
            else:
                # ??
                # when there is no measurements, append zero G, e_y, and W_y to match the dimension 
                G = np.zeros((1,3))
                W_y = block_diag(W_y, 0)
                e_y_k = np.array([0]); 
                e_y = np.append(e_y, np.squeeze(e_y_k))
                H_L = block_diag(H_L, G)

        e_v = e_v.reshape(-1,1); e_y = e_y.reshape(-1,1)

        # stack error in motion and meas. 
        ex = np.concatenate((e_v,e_y),axis=0)    

        # construct covariance matrix
        W = block_diag(W_v, W_y)

        W = W + np.eye(len(W)) * 1e-10          # avoid numerical errors

        H = np.concatenate((H_U, H_L), axis=0)  # stack H_U, H_L vertically
        W_inv = linalg.inv(W)

        A = (H.T).dot(W_inv).dot(H)
        b = (H.T).dot(W_inv).dot(ex)

        return A, b