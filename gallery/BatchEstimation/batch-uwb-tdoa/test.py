import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy import linalg
import math
from scipy.linalg import block_diag

np.set_printoptions(precision=5)
# yaw=0.2, pitch=0.1, roll=0.05 (degree)

# yaw = 0.00349,  pitch=0.001745, roll = 0.00087 (radian)
q = Quaternion([0.9999, 0.0004, 0.0008, 0.0017])
# yaw =0, pitch =0, roll=0
q1 = Quaternion([1,0,0,0])

q1_inv = q1.inverse

dq = q1_inv * q         # q = q1 * dq ---> dq = q1_inv * q
dq2 = 2*dq
d_log = Quaternion.log_map(q1_inv, q)   # [???, need to time 2 to get the error in angles], see equation (112)
d_log2 = 2 * Quaternion.log_map(q1_inv, q)
print("Test the SO3 X SO3 -> R3: \n")
print("dq gives {0}, {1}, {2}\n".format(dq[1],dq[2], dq[3]))
print("log_map gives {0}, {1}, {2}\n".format(d_log[1],d_log[2], d_log[3]))
print("2*dq gives {0}, {1}, {2}\n".format(dq2[1],dq2[2], dq2[3]))
print("2*log_map gives {0}, {1}, {2}\n".format(d_log2[1],d_log2[2], d_log2[3]))

print("true values: roll: {0}, pitch:{1}, yaw:{2} in radian\n".format(0.00087, 0.001745,  0.00349))


