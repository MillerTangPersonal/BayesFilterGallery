'''
RTS smoother implementation for batch estimation
'''
from util import wrapToPi
import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag

class RTS_Smoother_2D:
    '''initialization'''
    def __init__(self, P0, Q, R, l, d, K):
        # in nonlinear batch estimation,
        # we use GN to compute dx and update to the operating points until convergence  
        self.dXpr_f = np.zeros((K, 3))
        self.dXpo_f = np.zeros((K, 3))
        self.dXpo = np.zeros((K, 3))

        self.Ppr = np.zeros((K, 3, 3))
        self.Ppo = np.zeros((K, 3, 3))

        # self.dXpr_f = np.zeros((1,3))  # initial state error is set to be zero.  

        # initial state and covariance
        self.Ppr[0] = P0
        self.Ppo[0] = P0
        self.Qi = Q        # input noise
        self.Rm = R        # meas. noise
        self.landmark = l  # landmark positions
        self.d = d
        # save the forward Jacobian (will be used for the backward pass)
        self.F = np.zeros((K, 3, 3))
        self.Kmax = K

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
        W_prop = A.dot(Q_var).dot(A.T)
        return W_prop

    '''meas. model'''
    def meas_model(self, x, l_xy, d):
        # x = [x_k, y_k, theta_k] = x_op(k)
        # l_xy = [x_l, y_l]
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
        F = np.array([[1, 0, -dt * math.sin(x_op[2]) * v],
                      [0, 1,  dt * math.cos(x_op[2]) * v],
                      [0, 0,  1]],dtype=float)
        return F

    '''compute meas. model Jacobian'''
    def compute_G(self, x_op, l_xy, d):
        # x_op = [x, y, theta] = x_op(k)
        # denominator 1
        D1 = np.sqrt( (l_xy[0] - x_op[0] - d*math.cos(x_op[2]))**2 + (l_xy[1] - x_op[1] - d*math.sin(x_op[2]))**2 )
        # denominator 2
        D2 = ( l_xy[0]-x_op[0]-d*math.cos(x_op[2]) )**2 + ( l_xy[1]-x_op[1]-d*math.sin(x_op[2]) )**2

        g11 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g12 = -(l_xy[1] - x_op[1] - d*math.sin(x_op[2]))
        g13 = (l_xy[0] - x_op[0]) * d * math.sin(x_op[2]) - (l_xy[1] - x_op[1]) * d * math.cos(x_op[2])

        g21 = l_xy[1] - x_op[1] - d*math.sin(x_op[2])
        g22 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g23 = d**2 - d*math.cos(x_op[2])*(l_xy[0] - x_op[0]) - d*math.sin(x_op[2])*(l_xy[1] - x_op[1])

        G = np.array([[g11/D1, g12/D1, g13/D1],
                      [g21/D2, g22/D2, g23/D2 -1]])

        return np.squeeze(G)

    '''compute ev_k'''
    def compute_ev_k(self, x_op_k1, x_op_k, v, om, T):
        x_pro = self.motion_model(x_op_k1, T, v, om)  # f(x_op,k-1, v_k, 0)
        ev_k = x_pro - x_op_k.reshape(-1,1)           # f(x_op,k-1, v_k, 0) - x_op,k
        ev_k[2] = wrapToPi(ev_k[2])
        return ev_k

    '''forward pass'''
    def forward(self, x_op_k1, x_op_k, v_k, om_k, r_k, b_k, dt, k):
        # compute ev_k
        ev_k = self.compute_ev_k(x_op_k1, x_op_k, v_k, om_k, dt)
        # compute F_k-1
        F_k1 = self.compute_F(x_op_k1, dt, v_k)

        # save motion Jacobian
        self.F[k-1,:,:] = F_k1

        # compute Wv_k
        Wv_k = self.nv_prop(x_op_k1, dt, self.Qi)       # Q'_k = w_c(x_op_k1) * self.Qi * w_c(x_op_k1)

        l_k = np.nonzero(r_k);    l_k = np.asarray(l_k).reshape(-1,1)
        M = np.count_nonzero(r_k, axis=0)               # at timestamp k, we have M meas. in total

        # measurements
        # check if we have meas. or not
        ey_k = np.empty(0);       W_y  = np.empty((0,0))
        # flag for measurement update
        update = True
        if M: 
            G = np.empty((0,3))                         # empty G to contain all the Gs in timestep k
            for m in range(M): 
                l_xy = self.landmark[l_k[m,0], :]       # the m-th landmark pos.
                r_l = r_k[l_k[m,0]]                     # [timestamp, id of landmark]
                b_l = b_k[l_k[m,0]]
                y = np.array([r_l, b_l]).reshape(-1,1)
                ey = y - self.meas_model(x_op_k, l_xy, self.d)
                # compute angle error, wrap to pi
                ey[1] = wrapToPi(ey[1])
                '''compute ey_k'''
                ey_k = np.append(ey_k, np.squeeze(ey))
                # --------- save the variance of meas. ------------ #
                W_y = block_diag(W_y, self.Rm)
                # compute G
                G_i = self.compute_G(x_op_k, l_xy, self.d)
                G = np.concatenate((G, G_i), axis=0)
        else:
            # when no measurements, use Xpr as Xpo
            update = False
        
        # forward equations
        if update:
            # equ. 1
            self.Ppr[k,:,:] = F_k1.dot(self.Ppo[k-1,:,:]).dot(F_k1.T) + Wv_k
            # equ. 2
            dx_check = F_k1.dot(self.dXpo_f[k-1,:].reshape(-1,1)) + ev_k
            # wrap to pi
            dx_check[2,0] = wrapToPi(dx_check[2,0])    # dx_check[2,0] is delta th
            self.dXpr_f[k,:] = np.squeeze(dx_check)
            # equ. 3
            GM = G.dot(self.Ppr[k,:,:]).dot(G.T) + W_y
            K_k = self.Ppr[k,:,:].dot(G.T).dot(linalg.inv(GM))
            # equ. 4
            self.Ppo[k,:,:] = (np.eye(3) - K_k.dot(G)).dot(self.Ppr[k,:,:])
            # equ. 5
            in_err = ey_k.reshape(-1,1) - G.dot(self.dXpr_f[k,:].reshape(-1,1))   # innovation error
            dx_hat = self.dXpr_f[k,:].reshape(-1,1) + K_k.dot(in_err)
            # wrap to pi
            dx_hat[2,0] = wrapToPi(dx_hat[2,0])    # dx_hat[2,0] is delta th
            self.dXpo_f[k,:] = np.squeeze(dx_hat)

        else:
            # equ. 1
            self.Ppr[k,:,:] = F_k1.dot(self.Ppo[k-1,:,:]).dot(F_k1.T) + Wv_k
            # equ. 2
            dx_check = F_k1.dot(self.dXpo_f[k-1,:].reshape(-1,1)) + ev_k
            # wrap to pi
            dx_check[2,0] = wrapToPi(dx_check[2,0])    # dx_check[2,0] is delta th
            self.dXpr_f[k,:] = np.squeeze(dx_check)
            # no meas. to update
            self.Ppo[k,:,:] = self.Ppr[k,:,:]
            self.dXpo_f[k,:] = self.dXpr_f[k,:]


    '''backward pass'''
    def backward(self, k):
        if k == self.Kmax -1:   # the last idx is K-1
            # initialize
            self.dXpo[k,:] = self.dXpo_f[k,:]
            xk_hat = self.dXpo_f[k,:].reshape(-1,1)
        else:
            # dXpo[k,:] computed from last iteration
            xk_hat = self.dXpo[k,:].reshape(-1,1)

        dx = xk_hat - self.dXpr_f[k,:].reshape(-1,1)

        # print("\n")
        # print("dXpo(k):", self.dXpo[k,:])
        # print("dXpr_f(k): ", self.dXpr_f[k,:])
        # print("\n")
        # print("[dx_2: {0}, backward k: {1}]".format(dx[2,0],k))
        dx[2,0] = wrapToPi(dx[2,0])

        PAP = self.Ppo[k-1,:,:].dot(self.F[k-1,:,:].T).dot(linalg.inv(self.Ppr[k,:,:])).dot(dx)
        dx_hat = self.dXpo_f[k-1,:].reshape(-1,1) + PAP
        self.dXpo[k-1,:] = np.squeeze(dx_hat)


