'''
2D ground robot dynamics
'''
import numpy as np
from scipy import linalg
import math
from util import wrapToPi

class GroundRobot:
    '''initialization'''
    def __init__(self, Q, R, d, l, T):
        self.Qi = Q                          # input noise
        self.Rm = R                          # meas. noise
        self.dt = T                          # discrete time interval
        self.d  = d                          # sensor calibration
        self.landmark = l                    # landmark positions {x,y}

    '''motion model'''
    def motion_model(self, x, v, om):
        # x(k) = motion_model(x(k-1), input(k))
        x = np.squeeze(x)
        A = self.dt * np.array([[math.cos(x[2]),   0.0],
                                [math.sin(x[2]),   0.0],
                                [0.0,              1.0]], dtype=float)
        V = np.array([v, om]).reshape(-1,1)    # input
        x_new = x + np.squeeze(A.dot(V))
        x_new[2] = wrapToPi(x_new[2])

        return x_new.reshape(-1,1)

    '''propagate input noise'''
    def nv_prop(self, x):
        # Q'(k) = (T*A(x_k-1)) * Qi * (T*A(x_k-1)).T
        x = np.squeeze(x)
        A = self.dt * np.array([[math.cos(x[2]),  0.0],
                                [math.sin(x[2]),  0.0],
                                [0.0,             1.0]], dtype=float)
        W_prop  = A.dot(self.Qi).dot(A.T)
        return W_prop

    '''compute motion model Jacobian'''
    def compute_F(self, x_op, v):
        # for F_k-1, x_op = x_op(k-1) and v = v[k]
        F = np.array([[1.0,  0.0,  - self.dt * math.sin(x_op[2]) * v],
                      [0.0,  1.0,    self.dt * math.cos(x_op[2]) * v],
                      [0.0,  0.0,    1.0]], dtype=float)
        return F

    '''measurement model'''
    def meas_model(self, x, l_xy):
        # x = [x_k, y_k, theta_k] = x_op(k)
        # l_xy = [l_x, l_y]
        x = np.squeeze(x)
        x_m = l_xy[0] - x[0] - self.d * math.cos(x[2])
        y_m = l_xy[1] - x[1] - self.d * math.sin(x[2])

        r_m = np.sqrt(x_m**2 + y_m**2)
        phi_m= np.arctan2(y_m, x_m) - x[2]   # in radian
        # safety code: wrap to Pi
        phi_m = wrapToPi(phi_m)

        meas = np.array([r_m, phi_m]).reshape(-1,1)
        return meas

    '''compute meas. motion Jacobian'''
    def compute_G(self, x_op, l_xy):
        # x_op = [x, y, theta] = x_op(k)
        
        d = self.d; 
        # denominator 1
        x_op = np.squeeze(x_op)
        D1 = math.sqrt( (l_xy[0] - x_op[0] - d*math.cos(x_op[2]))**2 + (l_xy[1] - x_op[1] - d*math.sin(x_op[2]))**2 )
        # denominator 2
        D2 = ( l_xy[0]-x_op[0]-d*math.cos(x_op[2]) )**2 + ( l_xy[1]-x_op[1]-d*math.sin(x_op[2]) )**2

        g11 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g12 = -(l_xy[1] - x_op[1] - d*math.sin(x_op[2]))
        g13 = (l_xy[0] - x_op[0]) * d * math.sin(x_op[2]) - (l_xy[1] - x_op[1]) * d * math.cos(x_op[2])

        g21 = l_xy[1] - x_op[1] - d*math.sin(x_op[2])
        g22 = -(l_xy[0] - x_op[0] - d*math.cos(x_op[2]))
        g23 = d**2 - d*math.cos(x_op[2])*(l_xy[0] - x_op[0]) - d*math.sin(x_op[2])*(l_xy[1] - x_op[1])

        G = np.array([[g11/D1, g12/D1, g13/D1],
                      [g21/D2, g22/D2, g23/D2 - 1.0]], dtype=float)

        return np.squeeze(G)







