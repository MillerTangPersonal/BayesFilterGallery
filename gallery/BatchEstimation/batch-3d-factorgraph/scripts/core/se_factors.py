from se_data_types import *
from se_vertex import *
from se_utils import *

class Factor:
    def __init__(self):
        self.id = -1
        self.n_residuals = 0
        self.n_parameters = 0
    
    # must be implemented by child class
    def get_error(self, param_blocks, residuals):
        return -1
    
    # must be implemented by child class
    def get_jacobian(self, param_blocks, jacobian_blocks):
        return []
    
    def echo_info():
        print("Default factor!")

class SE3PriorFactor(Factor):
    '''
        T_prior: Pose3 object
        var: associated uncertainity
    '''
    def __init__(self, T_prior, var):
        self.n_residuals = 6
        self.n_parameters = 6
        self.data_prior = T_prior.data
        self.var = var
    
    def get_error(self, param_blocks, residuals):
        T_prior_est = param_blocks[0].data
        residuals[:] = jxl.SE3.log(T_prior_est.inverse() @ self.data_prior)
        return True
    
    def get_jacobian(self, param_blocks, J):
        assert np.shape(J[0]) == (6,6)
        J[0][:,:] = -np.identity(self.n_residuals, dtype=float)
        return True
    
    def get_covariance(self, covariance):
        assert np.shape(covariance) == (6,6)
        covariance[:,:] = np.diag(np.reciprocal(self.var))
        return True

class SE3BetweenFactorTwist(Factor):
    def __init__(self, lin_vel, ang_vel, lin_var, ang_var,
        dt):
        self.dt = dt
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        self.lin_var = lin_var
        self.ang_var = ang_var
        self.n_residuals = 6
        self.n_parameters = 6

    def get_error(self, param_blocks, residuals):
        T_prev = param_blocks[0].data
        T_curr = param_blocks[1].data
        tau_est = jxl.SE3.log(T_prev.inverse() @ T_curr)
        tau_meas = np.concatenate((self.lin_vel, self.ang_vel), axis=0) * self.dt;
        residuals[:] = tau_meas - tau_est;
        return True
    
    def get_jacobian(self, param_blocks, J):
        T_prev = param_blocks[0].data
        T_curr = param_blocks[1].data

        J[0][:,:] = np.identity(self.n_residuals, dtype=float)
        J[1][:,:] = -1 * (T_prev.inverse() @ T_curr).adjoint()
        
        return True

    def get_covariance(self, covariance):
        var = np.concatenate((self.lin_var, self.ang_var), axis=0) * self.dt;
        covariance[:,:] = np.diag(np.reciprocal(var))
        return True

class StereoFactor(Factor):
    def __init__(self, cam_params, extrinsic, landmark, pixels, var):
        self.cu = cam_params[0]
        self.cv = cam_params[1]
        self.fu = cam_params[2]
        self.fv = cam_params[3]
        self.b = cam_params[4]
        self.cTv = extrinsic
        self.landmark = landmark
        self.pixels = pixels
        self.var = var
        self.n_residuals = 4
        self.n_parameters = 6
    
    def get_error(self, param_blocks, residuals):
        T = param_blocks[0].data
        # landmark in vehicle frame
        # print(self.landmark)
        # print(T.as_matrix())
        lv = T.inverse() @ self.landmark
        # print(lv)
        # landmark in camera frame
        lc = self.cTv @ lv
        #
        x_, y_, z_ = lc
        reproj = np.zeros((4,), dtype=float)
        reproj[0] = self.fu * x_/z_ + self.cu
        reproj[1] = self.fv * y_/z_ + self.cv
        reproj[2] = self.fu * (x_ - self.b)/z_ + self.cu
        reproj[3] = self.fv * y_/z_ + self.cv
        residuals[:] = self.pixels - reproj
        return True
    
    def get_jacobian(self, param_blocks, jacobian_blocks):
        # for jacobian in jacobian_blocks:
        #     assert np.shape(jacobian) == (self.n_residuals, self.n_parameters)
        T = param_blocks[0].data
        # landmark in vehicle frame
        lv = T.inverse() @ self.landmark
        # landmark in camera frame
        lc = self.cTv @ lv
        x_, y_, z_ = lc

        J_camera = np.zeros((4,3), dtype=float)
        J_camera[0,0] = self.fu/z_
        J_camera[0,2] = -self.fu * x_/(z_**2)
        J_camera[1,1] = self.fv/z_
        J_camera[1,2] = -self.fv * y_/(z_**2)
        J_camera[2,0] = self.fu/z_
        J_camera[2,2] = -self.fu * (x_ - self.b)/(z_**2)
        J_camera[3,1] = self.fv/z_
        J_camera[3,2] = -self.fv * y_/(z_**2)

        D = np.zeros((3,4), dtype=float)
        D[0:3, 0:3] = np.identity(3, dtype=float)
        cdot = np.zeros((4,6), dtype=float)
        cdot[0:3, 0:3] = np.identity(3, dtype=float)
        cdot[0:3, 3:6] = skew(-lv)
        J_pose = np.dot(np.dot(D, self.cTv.as_matrix()), cdot)
        jacobian_blocks[0][:,:] = np.dot(J_camera, J_pose)
        return True
    
    def get_covariance(self, covariance):
        assert np.shape(covariance) == (self.n_residuals, self.n_residuals)
        covariance[:,:] = np.diag(np.reciprocal(self.var))
        return True
