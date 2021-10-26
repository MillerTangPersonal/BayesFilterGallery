'''
3D robot model
'''
import numpy as np
from scipy import linalg
from so3_util import axisAngle_to_Rot
from so3_util import getTrans, getTrans_in, skew, circle
np.set_printoptions(precision=4)

class Robot3D:
    '''initialization'''
    def __init__(self, Q, R, C_c_v, rho_v_c_v, fu, fv, cu, cv, b, rho_i_pj_i):
        self.Qi = Q                    # imu input variance
        self.Rm = R                    # stereo camera measurement variance
        self.C_c_v = C_c_v             # rotation from vehicle (IMU) frame to camera frame
        self.rho_v_c_v = rho_v_c_v     # translation from vehicle (IMU) frame to camera frame
        self.fu = fu                   # stereo camera's horizontal focal length
        self.fv = fv                   # stereo camera's vertical focal length
        self.cu = cu                   # stereo camera's horizontal optical center
        self.cv = cv                   # stereo camera's vertical optical center
        self.b  = b                    # stereo camera's baseline
        self.rho_i_pj_i = rho_i_pj_i   # landmark of visual features

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


    '''compute motion model Jacobian'''
    def compute_F(self, C_op_k_1, r_op_k_1, C_op_k, r_op_k):
        T = getTrans(C_op_k_1, r_op_k_1)
        T_in = getTrans_in(C_op_k, r_op_k)
        tau = T.dot(T_in)
        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        r_skew = skew(r_now)
        F = np.block([[C_now,            r_skew.dot(C_now)],
                      [np.zeros((3,3)),  C_now]])
        return F

    '''measurement model'''
    def meas_model(self,):
        pass

    '''compute meas. motion Jacobian'''
    def compute_G(self, C_op_k, r_op_k, idx, P_opt):
        rho_i_pj_i = self.rho_i_pj_i[:,idx]
        Pj_c = np.squeeze(P_opt)
        Pj = np.block([rho_i_pj_i, 1.0]).reshape(-1,1)
        D = np.block([[np.eye(3)],
                      [0, 0, 0]])
        T = getTrans(C_op_k, r_op_k)
        pose = T.dot(Pj)
        four_by_six = circle(pose)
        first_lin = self.C_c_v.dot(D.T).dot(four_by_six)
        second_lin = np.block([
            [self.fu/Pj_c[2],   0.0,               -(self.fu*Pj_c[0]) / (Pj_c[2]**2)],
            [0.0,               self.fv/Pj_c[2],   -(self.fv*Pj_c[1]) / (Pj_c[2]**2)],
            [self.fu/Pj_c[2],   0.0,               -(self.fu*(Pj_c[0]-self.b)) / (Pj_c[2]**2)],
            [0.0,               self.fv/Pj_c[2],   -(self.fv*Pj_c[1]) / (Pj_c[2]**2)]
        ])
        G = second_lin.dot(first_lin)

        return G

    '''compute the point'''
    def compute_point(self, C_op_k, r_op_k, idx):
        rho_i_pj_i = self.rho_i_pj_i[:,idx]
        rho = C_op_k.dot(rho_i_pj_i.reshape(-1,1) - r_op_k.reshape(-1,1))
        point = self.C_c_v.dot(rho - self.rho_v_c_v.reshape(-1,1))
        return point


