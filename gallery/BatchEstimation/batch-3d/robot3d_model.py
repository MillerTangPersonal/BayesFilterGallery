'''
3D robot model
'''
from so3_util import skew
import numpy as np
from scipy import linalg
from so3_util import axisAngle_to_Rot
import math

class Robot3D:
    '''initialization'''
    def __init__(self, Q, R, C_c_v, rho_v_c_v, fu, fv, cu, cv, b):
        self.Qi = Q                    # imu input variance
        self.Rm = R                    # stereo camera measurement variance
        self.C_c_v = C_c_v             # rotation from vehicle (IMU) frame to camera frame
        self.rho_v_c_v = rho_v_c_v     # translation from vehicle (IMU) frame to camera frame
        self.fu = fu                   # stereo camera's horizontal focal length
        self.fv = fv                   # stereo camera's vertical focal length
        self.cu = cu                   # stereo camera's horizontal optical center
        self.cv = cv                   # stereo camera's vertical optical center
        self.b  = b                    # stereo camera's baseline

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
    def compute_F(self,):
        pass

    '''measurement model'''
    def meas_model(self,):
        pass

    '''compute meas. motion Jacobian'''
    def compute_G(self, ):
        pass





