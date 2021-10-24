'''
RTS smoother implementation for batch estimation using SO3 
'''
import numpy as np
from scipy import linalg
import math
from scipy.linalg import block_diag


class RTS_Smoother_3D:
    '''initialization'''
    def __init__(self, P0, robot, K):
        self.dXpr_f = np.zeros((K, 6))
        self.dXpo_f = np.zeros((K, 6))
        self.dXpo   = np.zeros((K, 6))
        self.Ppr_f  = np.zeros((K, 6, 6))
        self.Ppo_f  = np.zeros((K, 6, 6))
        self.Ppo    = np.zeros((K, 6, 6))
        # init. covariance
        self.Ppr_f[0,:,:] = P0
        # save the forward Jacobian (for backward pass)
        self.F = np.zeros((K, 6, 6))

        self.robot = robot
        self.Kmax = K

    '''compute ev_k'''
    def compute_ev_k(self,):
        pass
    
    '''forward pass'''
    def forward(self,):
        pass

    '''backward pass'''
    def backward(self,):
        pass






