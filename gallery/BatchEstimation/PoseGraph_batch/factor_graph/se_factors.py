# import jaxlie as jxl
import numpy as np
from scipy import linalg
import math
from se_utils import skew, getTrans, getTrans_in, circle, axisAngle_from_rot

class Factor:
    def __init__(self):
        self.id = -1
        self.n_residuals = 0
        self.n_parameters = 0

    # implemented by child class
    def get_error(self, param_blocks, residuals):
        return -1

    # implemented by child class
    def get_jacobian(self, param_blocks, jacobian_blocks):
        return []

    def echo_info():
        print("Default factor!")

class SE3PriorFactor(Factor):
    '''
        T_prior: Pose3 object
        var: associated uncertainty
    '''
    def __init__(self, T_prior, var):
        self.n_residuals = 6
        self.n_parameters = 6
        self.data_prior = T_prior.data
        self.var = var

    def get_error(self, param_blocks, residuals):
        # log(T_prior @ T_op0_inverse)
        T_op0 = param_blocks[0].data
        error_T = self.data_prior @ np.linalg.inv(T_op0)

        C_data = error_T[0:3, 0:3]
        r_data = error_T[0:3, 3]
        # Equ. (60) to compute J. Then, J * rho = r
        # could be done by eigenvalue
        axisAngle = np.squeeze(axisAngle_from_rot(C_data))
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
        # J * rho = r_now
        r_data = r_data.reshape(-1,1)
        rho_data = np.squeeze(np.linalg.solve(J, r_data))   # [3,]
        phi_data = np.squeeze(axisAngle[3] * a)             # [3,]

        residuals[:] = np.block([rho_data, phi_data])       # shape (6,)
        return True

    def get_jacobian(self, param_blocks, J):
        assert np.shape(J[0]) == (6,6)
        J[0][:,:] = np.identity(self.n_residuals, dtype = float)
        return True

    def get_covariance(self, covariance):
        assert np.shape(covariance) == (6,6)
        covariance[:,:] = np.diag(np.reciprocal(self.var))
        return True

class SE3BetweenFactorTwist(Factor):
    def __init__(self, lin_vel, ang_vel, lin_var, ang_var, dt):
        self.dt = dt
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        self.lin_var = lin_var
        self.ang_var = ang_var
        self.n_residuals = 6
        self.n_parameters = 6

    def get_error(self, param_blocks, residuals):
        # log(T_prior_k @ T_opk_inverse)
        T_op_k1 = param_blocks[0].data      # operating point at time k-1
        T_op_k  = param_blocks[1].data      # operating point at time k

        input = np.block([-self.lin_vel, -self.ang_vel])
        rho = input[0:3].reshape(-1,1)
        phi = input[3:6]
        phi_skew = skew(phi)
        zeta = np.block([
            [phi_skew, rho],
            [0,  0,  0,  0]])
        Psi = linalg.expm(self.dt*zeta)
        # T_op = [C_op, -C_op @ r_op]
        # so that r_op = -1.0 * C_op.T @ (-C_op @ r_op)

        # can be replaced by T_op_k1 and T_op_k
        C_op_k1 = T_op_k1[0:3, 0:3];  #r_op_k1 = -1.0 * C_op_k1.T @ T_op_k1[3, 0:3]
        C_op_k  = T_op_k[0:3, 0:3];   #r_op_k  = -1.0 * C_op_k.T @ T_op_k[3, 0:3]
        
        
        r_op_k1 = -1.0 * C_op_k1.T @ T_op_k1[0:3, 3]
        r_op_k  = -1.0 * C_op_k.T @ T_op_k[0:3, 3]
        
        T_k1 = getTrans(C_op_k1, r_op_k1)
        T_k_in = getTrans_in(C_op_k, r_op_k)
        tau = Psi @ T_k1 @ T_k_in

        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        # Equ. (60) to compute J. Then, J * rho = r
        # could be done by eigenvalue
        axisAngle = axisAngle_from_rot(C_now)
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
        rho_now = np.squeeze(np.linalg.solve(J, r_now))
        phi_now = np.squeeze(axisAngle[3] * a)
        residuals[:] = np.block([rho_now, phi_now])

        return True

    def get_jacobian(self, param_blocks, J):
        # compute Adjoint with T[[C, r],[0.T,1]]
        # Ad = [ [C, r_skew @ C] [0, C] ]
        T_op_k1 = param_blocks[0].data         # operating point at time k-1
        T_op_k  = param_blocks[1].data         # operating point at time k
        # TODO: improve this part with jxl.adjoint()
        C_op_k1 = T_op_k1[0:3, 0:3];  r_op_k1 = T_op_k1[3, 0:3]
        C_op_k  = T_op_k[0:3, 0:3];   r_op_k  = T_op_k[3, 0:3]
        #
        T = getTrans(C_op_k, r_op_k)
        T_in = getTrans_in(C_op_k1, r_op_k1)
        tau = T.dot(T_in)
        C_now = tau[0:3, 0:3]
        r_now = tau[0:3, 3]
        r_skew = skew(r_now)
        F_ad = np.block([[C_now,            r_skew.dot(C_now)],
                         [np.zeros((3,3)),  C_now]])
        # jacobian [-F, 1]
        J[0][:,:] = -1.0 * F_ad
        J[1][:,:] = np.identity(self.n_residuals, dtype=float)
        return True

    def get_covariance(self, covariance):
        var = np.concatenate((self.lin_var, self.ang_var), axis=0) * self.dt;
        covariance[:,:] = np.diag(np.reciprocal(var))
        return True


class StereoFactor(Factor):
    def __init__(self, cam_params, C_c_v, rho_v_c_v, landmark, pixels, var):
        self.cu = cam_params[0]
        self.cv = cam_params[1]
        self.fu = cam_params[2]
        self.fv = cam_params[3]
        self.b  = cam_params[4]
        self.C_c_v = C_c_v
        self.rho_v_c_v = rho_v_c_v
        self.landmark = landmark
        self.pixels = pixels
        self.var = var
        self.n_residuals = 4
        self.n_parameters = 6

    def get_error(self, param_blocks, residuals):
        T_op = param_blocks[0].data
        C_op = T_op[0:3,0:3]                       # C_vk,i
        r_op = -1.0 * C_op.T @ T_op[0:3,3]         # r_i^vk,i
        # compute the point: (landmark = rho_i_pj_i)
        rho = C_op.dot(self.landmark.reshape(-1,1) - r_op.reshape(-1,1))
        point = self.C_c_v.dot(rho - self.rho_v_c_v.reshape(-1,1))
        # measurement model
        P_opt = np.squeeze(point)
        vec = np.array([
            self.fu*P_opt[0],
            self.fv*P_opt[1],
            self.fu*(P_opt[0] - self.b),
            self.fv*P_opt[1]
        ]).reshape(-1,1)

        vec_c = np.array([self.cu,
                          self.cv,
                          self.cu,
                          self.cv]).reshape(-1,1)
        y_o = (1.0/P_opt[2]) * vec + vec_c
        # pixel = y_k_j
        residuals[:] = self.pixels - np.squeeze(y_o)     # shape of (4,)

        return True

    def get_jacobian(self, param_blocks, J):
        T_op = param_blocks[0].data
        C_op = T_op[0:3,0:3]                       # C_vk,i
        r_op = -1.0 * C_op.T @ T_op[0:3,3]         # r_i^vk,i
        # compute the point: (landmark = rho_i_pj_i)
        rho = C_op.dot(self.landmark.reshape(-1,1) - r_op.reshape(-1,1))
        point = self.C_c_v.dot(rho - self.rho_v_c_v.reshape(-1,1))
        #
        Pj_c = np.squeeze(point)
        Pj = np.block([self.landmark, 1.0]).reshape(-1,1)
        D = np.block([[np.eye(3)],
                [0, 0, 0]])

        # can be replaced by  param_blocks[0].data
        T = getTrans(C_op, r_op)

        pose = T @ Pj
        four_by_six = circle(pose)
        first_lin = self.C_c_v.dot(D.T).dot(four_by_six)

        second_lin = np.block([
            [self.fu/Pj_c[2],   0.0,               -(self.fu*Pj_c[0]) / (Pj_c[2]**2)],
            [0.0,               self.fv/Pj_c[2],   -(self.fv*Pj_c[1]) / (Pj_c[2]**2)],
            [self.fu/Pj_c[2],   0.0,               -(self.fu*(Pj_c[0]-self.b)) / (Pj_c[2]**2)],
            [0.0,               self.fv/Pj_c[2],   -(self.fv*Pj_c[1]) / (Pj_c[2]**2)]
        ])
        J[0][:,:] = second_lin @ first_lin

        return True

    def get_covariance(self, covariance):
        assert np.shape(covariance) == (self.n_residuals, self.n_residuals)
        covariance[:,:] = np.diag(np.reciprocal(self.var))
        return True
