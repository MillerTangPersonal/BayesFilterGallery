'''
3D drone model
'''
import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy import linalg
from rot_util import skew, zeta
np.set_printoptions(precision=4)

class DroneModel:
    '''initialization'''
    def __init__(self, std_acc, std_gyro, R, C_u_v, rho_v_u_v, An):
        self.std_acc = std_acc        # std of acc (imu)
        self.std_gyro = std_gyro      # std of gyro (imu)
        self.Rm = R                   # uwb measurement variance
        self.C_u_v = C_u_v            # rotation from vehicle (imu) frame to UWB frame (not used here)
        self.rho_v_u_v = rho_v_u_v    # translation from vehicle (imu) frame to UWB frame
        self.An = An                  # anchor positions
        self.gravity = np.array([0, 0, 9.81]).reshape(-1,1)   # gravity vector

    '''propagate input noise'''
    def nv_prop(self, dt):
        # construct noise
        Vi = (self.std_acc**2)*(dt**2)*np.eye(3)
        Thetai = (self.std_gyro**2)*(dt**2)*np.eye(3)
        Qi = np.block([
            [Vi,               np.zeros((3,3)) ],
            [np.zeros((3,3)),  Thetai          ]
        ])
        Fi = np.block([
            [np.zeros((3,3)),   np.zeros((3,3))],
            [np.eye(3),         np.zeros((3,3))],
            [np.zeros((3,3)),   np.eye(3)      ]
        ])
        Qv_k = Fi.dot(Qi).dot(Fi.T)
        return Qv_k

    '''motion model'''
    def motion_model(self, X_k1, acc_k, gyro_k, dt):
        # convert quaternion to rotation matrix
        q_k1 = Quaternion([X_k1[6], X_k1[7], X_k1[8], X_k1[9]])
        R_k1 = q_k1.rotation_matrix
        # position 
        p_check = X_k1[0:3] + X_k1[3:6]*dt + 0.5 * np.squeeze(R_k1.dot(acc_k.reshape(-1,1)) - self.gravity) * dt**2
        # velocity
        v_check = X_k1[3:6] + np.squeeze(R_k1.dot(acc_k.reshape(-1,1)) - self.gravity) * dt
        # quaternion
        dw = gyro_k * dt
        dqk = Quaternion(zeta(dw)) # convert incremental rotation vector to quaternion [check: where is this equation]
        quat = q_k1 * dqk          # compute quaternion multiplication with package
        q_check = np.array([quat.w, quat.x, quat.y, quat.z])
        X_check = np.block([p_check, v_check, q_check])   # X_check \in R^10
        return X_check 

    '''compute motion model Jacobian in Batch estimation'''
    def compute_F(self,  X_k1, acc_k, gyro_k, dt):
        # convert quaternion to rotation matrix
        q_k1 = Quaternion([X_k1[6], X_k1[7], X_k1[8], X_k1[9]])
        R_k1 = q_k1.rotation_matrix
        dw = gyro_k * dt
        # Jacobian matrix of the motion
        Fx = np.block([
                [np.eye(3),         dt*np.eye(3),      -0.5*dt**2*R_k1.dot(skew(acc_k))],
                [np.zeros((3,3)),   np.eye(3),         -dt*R_k1.dot(skew(acc_k))       ],
                [np.zeros((3,3)),   np.zeros((3,3)),   linalg.expm(skew(dw)).T         ]            
            ])

        return Fx


    '''measurement model'''
    def meas_model(self, y_k, X_k):
        # get anchor positions
        an_i = self.An[int(y_k[0]), :]
        an_j = self.An[int(y_k[1]), :]
        # compute the uwb antenna position
        q_k = Quaternion([X_k[6], X_k[7], X_k[8], X_k[9]])
        C_iv = q_k.rotation_matrix
        p_uwb = C_iv.dot(self.rho_v_u_v) + X_k[0:3].reshape(-1,1)
        # measurement model: L2 norm between anchor and uwb tag position 
        d_i = linalg.norm(an_i - np.squeeze(p_uwb)) 
        d_j = linalg.norm(an_j - np.squeeze(p_uwb))
        y_meas = d_j - d_i
        return y_meas


    '''compute meas. motion Jacobian'''
    def compute_G(self, y_k, X_k):
        # get anchor positions
        an_i = self.An[int(y_k[0]), :]
        an_j = self.An[int(y_k[1]), :]
        # compute the uwb antenna position
        q_k = Quaternion([X_k[6], X_k[7], X_k[8], X_k[9]])
        C_iv = q_k.rotation_matrix
        p_uwb = C_iv.dot(self.rho_v_u_v) + X_k[0:3].reshape(-1,1)
        d_i = linalg.norm(np.squeeze(p_uwb) - an_i)
        d_j = linalg.norm(np.squeeze(p_uwb) - an_j)
        g_p = ((np.squeeze(p_uwb) - an_j)/d_j).reshape(1,-1) - ((np.squeeze(p_uwb) - an_i)/d_i).reshape(1,-1)
        g_v = np.zeros((1,3))
        # q_k = [q_w, q_x, q_y, q_z] = [q_w, q_v]
        q_w = q_k[0];  q_v = np.array([ q_k[1], q_k[2], q_k[3] ])
 
        # d_RVq = 2[q_w t_uv + q_v x t_uv, q_v^T t_uv I(3) + q_v t_uv^T - t_uv q_v - q_w[t_uv]x]
        t_uv = self.rho_v_u_v  # (for convenience)
        d_vec = q_w*t_uv + skew(q_v).dot(t_uv).reshape(-1,1)   # 3 x 1 vector
        d_mat = q_v.reshape(1,-1).dot(t_uv) * np.eye(3) + q_v.reshape(-1,1).dot(np.transpose(t_uv)) - t_uv.dot(q_v.reshape(1,-1)) - q_w * skew(t_uv)

        d_RVq = 2*np.concatenate((d_vec, d_mat), axis=1)

        g_q = ((np.squeeze(p_uwb) - an_j)/d_j).reshape(1,-1).dot(d_RVq) - ((np.squeeze(p_uwb) - an_i)/d_i).reshape(1,-1).dot(d_RVq)
        G_x = np.concatenate((g_p, g_v, g_q), axis=1)

        Q_dtheta = 0.5*np.array([
            [-q_k[1], -q_k[2], -q_k[3]],
            [ q_w,    -q_k[3],  q_k[2]],
            [ q_k[3],  q_w,    -q_k[1]],
            [-q_k[2],  q_k[1],  q_w]
        ])
        G_dx = linalg.block_diag(np.eye(6), Q_dtheta)
        G = G_x.dot(G_dx)    # shape: 1 x 9
        return G







