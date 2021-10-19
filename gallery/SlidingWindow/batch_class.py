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
    def __init__(self, P0, Q, R, l, d, K):
        self.P0_check = P0     # init covariance
        self.Qi = Q            # input noise
        self.Rm = R            # meas. noise
        self.landmark = l      # landmark positions
        self.d = d             # external calibration

        self.Kmax = K          # for construct A, b matrix

    '''motion model'''
    def motion_model(self, x, dt, v, om):
        x = np.squeeze(x)
        A = np.array([[dt * math.cos(x[2]), 0],
                      [dt * math.sin(x[2]), 0],
                      [0,                   dt]])
        V = np.array([v, om]).reshape(-1,1)
        x_new = x + np.squeeze(A.dot(V))
        x_new[2] = wrapToPi(x_new[2])
        return x_new.reshape(-1,1)

    '''propagate input noise'''
    def nv_prop(self, x, dt, Q_var):
        # x = x_op(k-1)
        x = np.squeeze(x)
        A = np.array([[dt * math.cos(x[2]), 0],
                      [dt * math.sin(x[2]), 0],
                      [0                 , dt]])
        W_prop = A.dot(Q_var).dot(A.T) + np.eye(3)*1e-10     # avoid numerical errors
        return W_prop

    '''meas. model'''
    def meas_model(self, x, l_xy, d):
        # x = [x_k, y_k, theta_k] = x_op(k)
        # l_xy = [x_l, y_l]
        x = np.squeeze(x)
        x_m = l_xy[0]-x[0]-d*math.cos(x[2])
        y_m = l_xy[1]-x[1]-d*math.sin(x[2])

        r_m = np.sqrt( x_m**2 + y_m**2 )
        phi_m = np.arctan2(y_m, x_m) - x[2]    # in radian
        
        # safety code: wrap to PI
        phi_m = wrapToPi(phi_m)

        meas = np.array([r_m, phi_m])
        return meas.reshape(-1,1)

    '''compute motion model Jacobian'''
    def compute_F(self, x_op, dt, v):
        # for F_k-1, x_op = x_op(k-1)
        # v = v[k]
        F = np.array([[1, 0, -dt * math.sin(x_op[2]) * v],
                      [0, 1,  dt * math.cos(x_op[2]) * v],
                      [0, 0,  1]],dtype=float)
        return F

    '''compute meas. model Jacobian'''
    def compute_G(self, x_op, l_xy, d):
        # x_op = [x, y, theta] = x_op(k)
        # denominator 1
        x_op = np.squeeze(x_op)
        D1 = math.sqrt( (l_xy[0] - x_op[0] - d*math.cos(x_op[2]))**2 + (l_xy[1] - x_op[1] - d*math.sin(x_op[2]))**2 )
        # denominator 2
        D2 = ( l_xy[0]-x_op[0]-d*math.cos(x_op[2]) )**2 + ( l_xy[1]-x_op[1]-d*math.sin(x_op[2]) )**2

        g11 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g12 = -(l_xy[1] - x_op[1] - d*math.sin(x_op[2]))
        g13 = (l_xy[0] - x_op[0]) * d * math.sin(x_op[2]) - (l_xy[1] - x_op[1]) * d * math.cos(x_op[2])

        g21 = l_xy[1] - x_op[1] - d*math.sin(x_op[2])
        g22 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g23 = d**2 - d*math.cos(x_op[2])*(l_xy[0] - x_op[0]) - d*math.sin(x_op[2])*(l_xy[1] - x_op[1])

        G = np.array([[g11/D1, g12/D1, g13/D1],
                      [g21/D2, g22/D2, g23/D2 - 1.0]], dtype=float)

        return np.squeeze(G)

    '''compute ev_k'''
    def compute_ev_k(self, x_op_k1, x_op_k, v, om, T):
        x_pro = self.motion_model(x_op_k1, T, v, om)  # f(x_op,k-1, v_k, 0)
        ev_k = x_pro - x_op_k.reshape(-1,1)           # f(x_op,k-1, v_k, 0) - x_op,k
        ev_k[2] = wrapToPi(ev_k[2])
        return ev_k

    '''construct the Ax=b relation'''
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
            Q_k = self.nv_prop(x_op_k1, dt, self.Qi)
            # compute e_k_v
            e_v_k = self.compute_ev_k(x_op_k1, x_op_k, v_k, om_k, dt)

            e_v = np.append(e_v, np.squeeze(e_v_k))
            W_v = block_diag(W_v, Q_k)

            # --------------- construct H_U matrix ------------------------ #
            H_U[(3*k):(3+3*k), (3*k-3):(3*k)] = -1.0 * self.compute_F(x_op_k1, dt, v_k)  # get F_{0,1,2}


            
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
                    l_xy = self.landmark[l_k[m,0], :]
                    r_l = r_k[l_k[m,0]]
                    b_l = b_k[l_k[m,0]]
                    y = np.array([r_l, b_l]).reshape(-1,1)
                    ey_m = y - self.meas_model(x_op_k, l_xy, self.d)
                    # wrap to Pi
                    ey_m[1] = wrapToPi(ey_m[1])

                    e_y_k = np.append(e_y_k, np.squeeze(ey_m))
                    W_y = block_diag(W_y, self.Rm)
                    
                    # compute G
                    G_i = self.compute_G(x_op_k, l_xy, self.d)
                    G = np.concatenate((G, G_i), axis=0)

                e_y = np.append(e_y, np.squeeze(e_y_k))
                H_L = block_diag(H_L, G)

            else:
                # ???
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