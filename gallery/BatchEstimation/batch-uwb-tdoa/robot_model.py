'''
3D drone model
'''
import numpy as np
from scipy import linalg
from so3_util import axisAngle_to_Rot, getTrans, getTrans_in, skew, circle
np.set_printoptions(precision=4)

class DroneModel:
    '''initialization'''
    def __init__(self, Q, R, C_u_v, rho_v_u_v, An):
        self.Qi = Q                   # imu input variance ()
        self.Rm = R                   # uwb measurement variance
        self.C_u_v = C_u_v            # rotation from vehicle (imu) frame to UWB frame
        self.rho_v_u_v = rho_v_u_v    # translation from vehicle (imu) frame to UWB frame
        self.An = An                  # anchor positions

    '''propagate input noise'''
    def nv_prop(self, dt):
        # Q' = self.Qi * dt^2
        Q_prop = self.Qi * dt**2

        return Q_prop

    '''motion model'''
    def motion_model(self, C_prev, r_prev, v_vk_vk_i, w_vk_vk_i, dt):
        d_v = dt * v_vk_vk_i.reshape(-1,1)  
        r_new = r_prev.reshape(-1,1) + (C_prev.T).dot(d_v)
        Psi_vec = w_vk_vk_i * dt
        # compute Psi
        Psi = axisAngle_to_Rot(Psi_vec)
        C_new = Psi.dot(C_prev)
        r_new = np.squeeze(r_new)
        return C_new, r_new

    '''compute motion model Jacobian in Batch estimation'''
    def compute_F(self,  C_op_k1, r_op_k1, C_op_k, r_op_k):
        # C_op_k1, r_op_k1 are C and r at timestep: k-1
        T = getTrans(C_op_k, r_op_k)
        T_in = getTrans_in(C_op_k1, r_op_k1)
        tau = T.dot(T_in)
        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        r_skew = skew(r_now)
        F = np.block([[C_now,            r_skew.dot(C_now)],
                      [np.zeros((3,3)),  C_now]])
        return F

    '''measurement model'''
    def meas_model(self, y_k, C_op_k, r_op_k):
        # get anchor positions
        an_i = self.An[int(y_k[0]),:] 
        an_j = self.An[int(y_k[1]),:]
        # compute the T_op from inertial to vehicle
        T_iv = getTrans_in(C_op_k, r_op_k)
        rho = np.squeeze(self.rho_v_u_v)
        r_l_arm = np.block([rho, 1]).reshape(-1,1)
        dj = linalg.norm(T_iv.dot(r_l_arm)[0:3] - an_j.reshape(-1,1))
        di = linalg.norm(T_iv.dot(r_l_arm)[0:3] - an_i.reshape(-1,1))
        # TDOA measurement
        y  = dj - di
        return y 

    '''compute meas. motion Jacobian'''
    def compute_G(self, y_k, C_op_k, r_op_k):
        # get anchor positions
        an_i = self.An[int(y_k[0]),:] 
        an_j = self.An[int(y_k[1]),:]
        # compute the T_op from inertial to vehicle
        T_iv = getTrans_in(C_op_k, r_op_k)
        rho = np.squeeze(self.rho_v_u_v)
        r_l_arm = np.block([rho, 1]).reshape(-1,1)
        dj = linalg.norm(T_iv.dot(r_l_arm)[0:3] - an_j.reshape(-1,1))
        di = linalg.norm(T_iv.dot(r_l_arm)[0:3] - an_i.reshape(-1,1))
        # Gk = A.dot(B)  
        # A: 1 x 4,  B: 4 x 6
        A = (T_iv.dot(r_l_arm) - np.block([np.squeeze(an_j), 1]).reshape(-1,1)).T / dj - \
            (T_iv.dot(r_l_arm) - np.block([np.squeeze(an_i), 1]).reshape(-1,1)).T / di

        p_hom = np.block([np.squeeze(rho), 1])
        p_circle = circle(p_hom)
        B = -1.0 * T_iv.dot(p_circle) 

        Gk = A.dot(B)

        return Gk

    '''compute the point'''
    def compute_point(self, C_op_k, r_op_k, idx):
        pass










