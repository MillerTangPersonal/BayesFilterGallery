'''
ESKF class without lever-arm
'''
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion

# ------------------ parameters ------------------ #
# Process noise
w_accxyz = 2.0;      w_gyro_rpy = 0.1    # rad/sec
w_vel = 0;           w_pos = 0;          w_att = 0;        

w_uwb_bias = 1.0;    # process noise for UWB bias

# Constants
GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD  = math.pi/180.0
e3 = np.array([0, 0, 1]).reshape(-1,1)   

class ESKF:
    '''initialization'''
    def __init__(self, X0, q0, P0, K):
        # Standard devirations of UWB meas. (tuning parameter)
        self.std_uwb_tdoa = np.sqrt(0.05)
        self.f = np.zeros((K, 3))
        self.omega = np.zeros((K,3))
        self.q_list = np.zeros((K,4))    # quaternion list
        self.R_list = np.zeros((K,3,3))  # Rotation matrix list (from body frame to inertial frame) 

        # nominal-state: X = [x, y, z, vx, vy, vz, uwb_bias0, ..., uwb_bias7]
        # error-state:   delta_x = [dx, dy, dz, dvx, dvy, dvz, dr1, dp2, dyaw3, bias0, ..., bias7]
        self.Xpr = np.zeros((K,14))
        self.Xpo = np.zeros((K,14))
        self.Ppr = np.zeros((K, 17, 17))  # 9+8
        self.Ppo = np.zeros((K, 17, 17))  # 9+8

        self.Ppr[0] = P0
        self.Ppo[0] = P0
        self.Xpr[0] = X0.T
        self.Xpo[0] = X0.T
        self.q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
        # current rotation matrix list (from body frame to inertial frame) 
        self.R = q0.rotation_matrix
        # # Process noise
        # self.Fi = np.block([
        #     [np.zeros((3,3)),   np.zeros((3,3))],
        #     [np.eye(3),         np.zeros((3,3))],
        #     [np.zeros((3,3)),   np.eye(3)      ]
        # ])

        self.Fi = np.block([
            [np.zeros((3,14))],
            [np.eye(14)]
        ])
        # Fi is 10 x 14

    '''ESKF prediction using IMU'''
    def predict(self, imu, dt, imu_check, k):
        # construct noise
        Vi = (w_accxyz**2)*(dt**2) * np.eye(3)
        Thetai = (w_gyro_rpy**2)*(dt**2) * np.eye(3)
        # uwb bias
        W_bias = (w_uwb_bias**2)*(dt**2) * np.eye(8)

        Qi = np.block([
            [Vi,               np.zeros((3,3)) , np.zeros((3,8))],
            [np.zeros((3,3)),  Thetai          , np.zeros((3,8))],
            [np.zeros((8,3)),  np.zeros((8,3)),  W_bias]
        ])
        # Qi is 7x7
        if imu_check:
            # We have a new IMU measurement
            # nominal state propagation
            # update the prior Xpr based on accelerometer and gyroscope data
            omega_k = imu[3:]                 # * DEG_TO_RAD        # in simulation, gyro is rad/sec
            self.omega[k] = omega_k
            Vpo = self.Xpo[k-1,3:6]
            # Acc: G --> m/s^2
            f_k = imu[0:3]                    # * GRAVITY_MAGNITUDE  # in simulation, acc is in m/s^2
            self.f[k] = f_k
            dw = omega_k * dt                      # Attitude error
            # nominal state motion model
            # position prediction 
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(self.R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2
            # velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt
            # if CF is on the ground
            if self.Xpr[k, 2] < 0:  
                self.Xpr[k, 2:6] = np.zeros((1,4))    
            # quaternion prediction
            qk_1 = Quaternion(self.q_list[k-1,:])
            dqk  = Quaternion(self.zeta(dw))           # convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                          # compute quaternion multiplication with package
            self.q_list[k,:] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])  # save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list

            # UWB bias prediction 
            self.Xpr[k,6:14] = self.Xpo[k-1, 6:14] 

            # error state covariance matrix 
            # use the rotation matrix from timestep k-1
            self.R = qk_1.rotation_matrix          
            # Jacobian matrix
            Fx = np.block([
                [np.eye(3),         dt*np.eye(3),      -0.5*dt**2*self.R.dot(self.cross(f_k)), np.zeros((3,8))],
                [np.zeros((3,3)),   np.eye(3),         -dt*self.R.dot(self.cross(f_k))       , np.zeros((3,8))],
                [np.zeros((3,3)),   np.zeros((3,3)),   linalg.expm(self.cross(dw)).T         , np.zeros((3,8))],
                [np.zeros((8,3)),   np.zeros((8,3)),   np.zeros((8,3))                       , np.eye(8)]            
            ])
            # Fx: 10 x 10

            # Process noise matrix Fi, Qi are defined above
            self.Ppr[k] = Fx.dot(self.Ppo[k-1]).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T) 
            # Enforce symmetry
            self.Ppr[k] = 0.5*(self.Ppr[k] + self.Ppr[k].T)  

        else:
            # if we don't have IMU data
            self.Ppr[k] = self.Ppo[k-1] + self.Fi.dot(Qi).dot(self.Fi.T)
            # Enforce symmetry
            self.Ppr[k] = 0.5*(self.Ppr[k] + self.Ppr[k].T)  
            
            self.omega[k] = self.omega[k-1]
            self.f[k] = self.f[k-1]
            dw = self.omega[k] * dt                      # Attitude error
            # nominal state motion model
            # position prediction 
            Vpo = self.Xpo[k-1,3:6]
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2
            # velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt
            # quaternion update
            qk_1 = Quaternion(self.q_list[k-1,:])
            dqk  = Quaternion(self.zeta(dw))       # convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                 # compute quaternion multiplication with package
            self.q_list[k] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])    # save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list

            # UWB bias prediction 
            self.Xpr[k,6:14] = self.Xpo[k-1, 6:14] 

        # End of Prediction

        # Initially take our posterior estimates as the prior estimates
        # These are updated if we have sensor measurements (UWB)
        self.Xpo[k] = self.Xpr[k]
        self.Ppo[k] = self.Ppr[k]

    '''ESKF correction using UWB'''
    def UWB_correct(self, uwb, anchor_position, k):
        an_A = anchor_position[int(uwb[0]),:]   # idA
        an_B = anchor_position[int(uwb[1]),:]   # idB
        dx0 = self.Xpr[k,0] - an_A[0];  dx1 = self.Xpr[k,0] - an_B[0]
        dy0 = self.Xpr[k,1] - an_A[1];  dy1 = self.Xpr[k,1] - an_B[1]
        dz0 = self.Xpr[k,2] - an_A[2];  dz1 = self.Xpr[k,2] - an_B[2]

        d_A = linalg.norm(an_A - np.squeeze(self.Xpr[k,0:3])) 
        d_B = linalg.norm(an_B - np.squeeze(self.Xpr[k,0:3]))
        predicted = d_B - d_A

        gb = np.zeros((1,8))
        if uwb[0]==7 and uwb[1]==0:
            gb[0,0] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,6]    # compensate for the bias
        elif uwb[0]==0 and uwb[1]==1:
            gb[0,1] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,7]    # compensate for the bias
        elif uwb[0]==1 and uwb[1]==2:
            gb[0,2] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,8]    # compensate for the bias 
        elif uwb[0]==2 and uwb[1]==3:
            gb[0,3] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,9]    # compensate for the bias
        elif uwb[0]==3 and uwb[1]==4:
            gb[0,4] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,10]    # compensate for the bias
        elif uwb[0]==4 and uwb[1]==5:
            gb[0,5] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,11]    # compensate for the bias
        elif uwb[0]==5 and uwb[1]==6:
            gb[0,6] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,12]    # compensate for the bias
        elif uwb[0]==6 and uwb[1]==7:
            gb[0,7] = 1
            err_uwb = uwb[2] - predicted - self.Xpr[k,13]    # compensate for the bias
        else:
            err_uwb = uwb[2] - predicted

        # compute the gradient of measurement model
        # G is 1 x 18 
        G = np.block([dx1/d_B - dx0/d_A,   dy1/d_B - dy0/d_A, dz1/d_B - dz0/d_A, np.zeros((1,6)), gb]).reshape(1,-1)

        # uwb covariance
        Q = self.std_uwb_tdoa**2
        M = np.squeeze(G.dot(self.Ppr[k]).dot(G.T) + Q)     # scalar 
        d_m = math.sqrt(err_uwb**2/M)

        # -------------------- Statistical Validation -------------------- #
        if d_m < 5:
            # Kk is 9 x 1
            Kk = (self.Ppr[k].dot(G.T) / M).reshape(-1,1)           # in scalar case
            # update the posterios covariance matrix for error states
            self.Ppo[k]= (np.eye(17) - Kk.dot(G.reshape(1,-1))).dot(self.Ppr[k])
            # enforce symmetry
            self.Ppo[k] = 0.5 * (self.Ppo[k] + self.Ppo[k].T)
            derror = Kk.dot(err_uwb)             
            # update nominal states 
            self.Xpo[k,0:6] = self.Xpr[k,0:6] +  np.squeeze(derror[0:6])
            # update UWB bias (the last state is the bias)
            self.Xpo[k,6:14] = self.Xpr[k,6:14] + np.squeeze(derror[9:])

            dq_k = Quaternion(self.zeta(np.squeeze(derror[6:9])))    
            #update quaternion: q_list
            qk_pr = Quaternion(self.q_list[k])
            qk_po = qk_pr * dq_k
            self.q_list[k] = np.array([qk_po.w, qk_po.x, qk_po.y, qk_po.z])
        else:
            # keep the previous state
            self.Xpo[k] = self.Xpr[k]
            self.Ppo[k] = self.Ppr[k]
            # keep the previous quaterion  q_list[k]


    '''help function'''
    def cross(self, v):    # input: 3x1 vector, output: 3x3 matrix
        v = np.squeeze(v)
        vx = np.array([
            [ 0,    -v[2], v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0], 0 ] 
        ])
        return vx
        
    '''help function'''
    def zeta(self, phi):
        phi_norm = np.linalg.norm(phi)
        if phi_norm == 0:
            dq = np.array([1, 0, 0, 0])
        else:
            dq_xyz = (phi*(math.sin(0.5*phi_norm)))/phi_norm
            dq = np.array([math.cos(0.5*phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
        return dq