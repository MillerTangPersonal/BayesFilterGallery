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





