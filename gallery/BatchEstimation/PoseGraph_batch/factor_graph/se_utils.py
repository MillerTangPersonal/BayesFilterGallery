import numpy as np
import math

'''skew operation'''
def skew(v):
    v = np.squeeze(v)
    return np.array([[0.0, -v[2],  v[1]], 
                     [v[2],  0.0, -v[0]], 
                     [-v[1], v[0],  0.0]], dtype=float)
                    
'''compute transformation matrix from C and r'''
def getTrans(C, r):
    r = r.reshape(-1,1)
    T = np.block([[C,    -C.dot(r)],
                  [0,  0,  0,  1.0]])
    return T

'''compute the inv. of transformation matrix'''
def getTrans_in(C, r):
    r = r.reshape(-1,1)
    T_in = np.block([[C.T,          r],
                     [0,  0,  0,  1.0]])
    return T_in

'''cricle operator in SO3'''
def circle(pose):
    pose = np.squeeze(pose)
    rho = pose[0:3]
    eta = pose[3]
    rho_skew = skew(rho)
    cir = np.block([ [eta*np.eye(3), -1.0 * rho_skew],
                     [0, 0, 0, 0, 0, 0]  ])
    return cir

'''compute rotation matrix from axis-angle vector'''
def axisAngle_to_Rot(theta):
    # compute the norm
    theta = np.squeeze(theta)
    nm = np.linalg.norm(theta)
    if nm == 0:
        Psi = np.eye(3)
    else:
        rot_vec = (theta/nm).reshape(-1,1);     # column vector
        rot_skew = skew(rot_vec)
        Psi = math.cos(nm)*np.eye(3) + ((1-math.cos(nm))*rot_vec.dot(rot_vec.T)) - (math.sin(nm)*rot_skew)
        
    return Psi
