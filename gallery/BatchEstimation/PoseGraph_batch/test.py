import numpy as np
import math
from jaxlie import SE3

a = np.array([[1,2,3]])
b = np.array([3, 4, 5])

c = np.block([a,b])
print(a.reshape(-1))
print(a.shape)
