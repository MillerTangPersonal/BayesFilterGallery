#import jaxlie as jxl
import numpy as np

# enumeration for supported types
class BASE_TYPE:
    R0 = 0
    R2 = 1
    SO2 = 2
    SE2 = 3
    R3 = 4
    SO3 = 5
    SE3 = 6

def type_as_string(idx):
    if idx == BASE_TYPE.R0:
        return "R0"
    elif idx == BASE_TYPE.R2:
        return "R2"
    elif idx == BASE_TYPE.SO2:
        return "SO2"
    elif idx == BASE_TYPE.SE2:
        return "SE2"
    elif idx == BASE_TYPE.R3:
        return "R3"
    elif idx == BASE_TYPE.SO3:
        return "SO3"
    elif idx == BASE_TYPE.SE3:
        return "SE3"
    else:
        return "UNKNOWN"

class Scalar:
    def __init__(self, val):
        self.data = val
        self.type = BASE_TYPE.R0
    
    def value(self):
        return self.data

# TODO: Add to each class check for type
class Vector2:
    def __init__(self, _data):
        self.data = _data
        self.type = BASE_TYPE.R2

    def x(self):
        return self.data[0]
    
    def y(self):
        return self.data[1]

# typedef for Point2
Point2 = Vector2

# typedef for SO2
class Rot2:
    def __init__(self, _data):
        self.type = BASE_TYPE.SO2
        self.data = _data

# typedef for SE2
class Pose2:
    def __init__(self, _data):
        self.type = BASE_TYPE.SE2
        self.data = _data
# 
class Vector3:
    def __init__(self, _data):
        self.type = BASE_TYPE.R3
        self.data = _data

    def x(self):
        return self.data[0]
    
    def y(self):
        return self.data[1]
    
    def z(self):
        return self.data[2]

# typedef for Point3
Point3 = Vector3

# typedef for SO3
class Rot3:
    def __init__(self, _data):
        self.type = BASE_TYPE.SO3
        self.data = _data

# typedef for SE3
class Pose3:
    def __init__(self, _data):
        self.type = BASE_TYPE.SE3
        self.data = _data
    
    def translation(self):
        return self.data.translation()

    def rotation(self):
        return self.data.rotation()

class UWBRange:
    def __init__(self, stamp, mob, anc, mob_pos, anc_pos, rng, cov):
        self.stamp = stamp
        self.anc = anc
        self.mob = mob
        self.anc_pos = np.reshape(anc_pos, -1)
        self.mob_pos = np.reshape(mob_pos, -1)
        self.dist = rng
        self.cov = cov