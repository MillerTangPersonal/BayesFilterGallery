'''Util functions for SO3 operation'''
import numpy as np
import math
from scipy import linalg

'''compute rotation matrix from axis-angle vector'''
def axisAngle_to_Rot(theta):
    # compute the norm
    theta = np.squeeze(theta)
    nm = linalg.norm(theta)
    if nm == 0:
        Psi = np.eye(3)
    else:
        rot_vec = (theta/nm).reshape(-1,1);     # column vector
        rot_skew = skew(rot_vec)
        Psi = math.cos(nm)*np.eye(3) + ((1-math.cos(nm))*rot_vec.dot(rot_vec.T)) - (math.sin(nm)*rot_skew)
        
    return Psi

'''skew operation'''
def skew(r):
    r = np.squeeze(r)
    r_skew = np.array([[0.0,   -r[2],   r[1]],
                       [r[2],      0,  -r[0]],
                       [-r[1],  r[0],    0.0]], dtype=float)
    return r_skew

'''compute transformation matrix from C and r (from inertial to vehicle --> system state)'''
def getTrans(C, r):
    r = r.reshape(-1,1)
    T = np.block([[C,    -C.dot(r)],
                  [0,  0,  0,  1.0]])
    return T

'''compute the inv. of transformation matrix (from vehicle to inertial --> useful elements)'''
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
    
''' compute angle-axis from rotation matrix (modified from pytransform3d.rotations.axis_angle_from_matrix)'''
def axisAngle_from_rot(R):
    """Compute axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    check : bool, optional (default: True)
        Check if rotation matrix is valid

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """

    cos_angle = (np.trace(R) - 1.0) / 2.0
    angle = np.arccos(min(max(-1.0, cos_angle), 1.0))

    '''the original package is not numerically stable in here'''
    #  if angle == 0.0: 
    if angle < 1.0e-06:  # R == np.eye(3)
        return np.array([1.0, 0.0, 0.0, 0.0])

    a = np.empty(4)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(angle - np.pi) < 1e-4:  # np.trace(R) close to -1
        # The threshold is a result from this discussion:
        # https://github.com/rock-learning/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.
        a[:3] = np.sqrt(0.5 * (np.diag(R) + 1.0)) * np.sign(axis_unnormalized)
    else:
        a[:3] = axis_unnormalized
        # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
        # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
        # but the following is much more precise for angles close to 0 or pi:
    a[:3] /= np.linalg.norm(a[:3])

    a[3] = angle
    return a



