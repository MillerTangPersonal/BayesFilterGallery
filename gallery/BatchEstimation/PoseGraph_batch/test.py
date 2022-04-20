import numpy as np
import math
from jaxlie import SE3

twist = np.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
T_w_b = SE3.exp(twist)

# We can print the (quaternion) rotation term; this is an `SO3` object:
print(T_w_b.rotation())

# Or print the translation; this is a simple array with shape (3,):
print(T_w_b.translation())

# Or the underlying parameters; this is a length-7 (quaternion, translation) array:
print(T_w_b.wxyz_xyz)  # SE3-specific field.
print(T_w_b.parameters())  # Helper shared by all groups.

# There are also other helpers to generate transforms, eg from matrices:
T_w_b = SE3.from_matrix(T_w_b.as_matrix())

# Or from explicit rotation and translation terms:
T_w_b = SE3.from_rotation_and_translation(
    rotation=T_w_b.rotation(),
    translation=T_w_b.translation(),
)

# Or with the dataclass constructor + the underlying length-7 parameterization:
T_w_b = SE3(wxyz_xyz=T_w_b.wxyz_xyz)


#############################
# (2) Applying transforms.
#############################

# Transform points with the `@` operator:
p_b = np.random.randn(3)
p_w = T_w_b @ p_b
print(p_w)

# or `.apply()`:
p_w = T_w_b.apply(p_b)
print(p_w)

# or the homogeneous matrix form:
p_w = (T_w_b.as_matrix() @ np.append(p_b, 1.0))[:-1]
print(p_w)


#############################
# (3) Composing transforms.
#############################

# Compose transforms with the `@` operator:
T_b_a = SE3.identity()
T_w_a = T_w_b @ T_b_a
print(T_w_a)

# or `.multiply()`:
T_w_a = T_w_b.multiply(T_b_a)
print(T_w_a)


#############################
# (4) Misc.
#############################

# Compute inverses:
T_b_w = T_w_b.inverse()
identity = T_w_b @ T_b_w
print(identity)

# Compute adjoints:
adjoint_T_w_b = T_w_b.adjoint()
print(adjoint_T_w_b)

# Recover our twist, equivalent to `vee(logm(T_w_b.as_matrix()))`:
twist_recovered = T_w_b.log()
print(twist_recovered)

# ---------------------------------- #
def skew(v): 
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]], dtype=float)

def axis_angle_from_matrix(R, strict_check=True, check=True):
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

    if angle == 0.0:  # R == np.eye(3)
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

# -------- tested here ------- # 
# T_w_b.log() equals to the following ways to compute log()_vee
tau = T_w_b.as_matrix()._value

C_now = tau[0:3, 0:3]
r_now = tau[0:3, 3]
# Equ. (60) to compute J. Then, J * rho = r
# could be done by eigenvalue
axisAngle = axis_angle_from_matrix(C_now)
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
rho_now = np.linalg.solve(J, r_now)      
phi_now = axisAngle[3] * a

rho_now = np.squeeze(rho_now)
phi_now = np.squeeze(phi_now)
print(rho_now)
print(phi_now)