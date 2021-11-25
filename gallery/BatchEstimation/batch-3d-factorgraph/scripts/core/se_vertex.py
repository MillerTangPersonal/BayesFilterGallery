import jaxlie as jxl
import numpy as np

from se_data_types import *

class Vertex(object):
    def create(_id, _obj):
        if _obj.type == BASE_TYPE.R0:
            return VertexScalar(_id, _obj)
        elif _obj.type == BASE_TYPE.R2:
            return VertexPoint2(_id, _obj)
        elif _obj.type == BASE_TYPE.SO2:
            return VertexSO2(_id, _obj)
        elif _obj.type == BASE_TYPE.SE2:
            return VertexSE2(_id, _obj)
        elif _obj.type == BASE_TYPE.R3:
            return VertexPoint3(_id, _obj)
        elif _obj.type == BASE_TYPE.SO3:
            return VertexSO3(_id, _obj)
        elif _obj.type == BASE_TYPE.SE3:
            return VertexSE3(_id, _obj)
        else:
            assert 0, "Uknown type request for vertex creation"

        create = staticmethod(create)

class VertexScalar(Vertex):
    def __init__(self, _id, _scalar):
        self.id = _id
        self.type = BASE_TYPE.R0
        self.n_parameters = 1
        self.data = _scalar.data
        self.var = np.zeros((self.n_parameters,), dtype=float)
    
    def update(self, _delta):
        assert len(_delta) == self.n_parameters, "VertexScalar retraction Scalar inconsistent"
        self.data += _delta[0]

    def value(self):
        return self.data

class VertexPoint2(Vertex):
    def __init__(self, _id, _point):
        self.id = _id
        self.type = BASE_TYPE.R2
        self.n_parameters = 2
        self.data = _point.data
        self.var = np.zeros((self.n_parameters,), dtype=float)
    
    def norm(self):
        return np.linalg.norm(self.data)

    def update(self, _delta):
        assert len(_delta) == self.n_parameters, "VertexPoint2 retraction Point2 inconsistent"
        self.data[:] += self.retract(_delta)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexPoint2 covariance update inconsistent"
        self.var = cov

    def retract(self, _delta):
        return _delta
    
    def x(self):
        return self.data[0]

    def y(self):
        return self.data[1]

    def echo_info(self):
        print("type:[%s] vals:[%.2f, %.2f]"%(type_as_string(self.type),
            self.data[0], self.data[1]))

class VertexSO2(Vertex):
    def __init__(self, _id, _rot):
        self.id = _id
        self.type = BASE_TYPE.SO2
        self.n_parameters = 1
        self.data = _rot.data
        self.var = np.zeros((self.n_parameters,), dtype=float)
    
    def as_matrix(self):
        return self.data.as_matrix()

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSO2 retraction SO2 inconsistent"
        self.data = self.data @ self.retract(_twist)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSO2 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        return jxl.SO2.exp(_twist)

    def inverse(self):
        return self.data.inverse()
    
    def inverseretract(self):
        return self.data.log()

class VertexSE2(Vertex):
    def __init__(self, _id, _pose):
        self.id = _id;
        self.type = BASE_TYPE.SE2
        self.n_parameters = 3
        self.data = _pose.data
        self.var = np.zeros((self.n_parameters,), dtype=float)

    def translation(self):
        return self.data.translation()
    
    def rotation(self):
        return self.data.rotation().as_matrix()

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSE2 retraction SE2 inconsistent"
        self.data = self.data @ self.retract(_twist)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSE2 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        return jxl.SE2.exp(_twist)

    def inverse(self):
        return self.data.inverse()
    
    def inverseretract(self):
        return self.data.log()

class VertexPoint3(Vertex):
    def __init__(self, _id, _point):
        self.id = _id
        self.type = BASE_TYPE.R3
        self.n_parameters = 3
        self.data = _point.data
        self.var = np.zeros((self.n_parameters,), dtype=float)

    def norm(self):
        return np.linalg.norm(self.data)

    def update(self, _delta):
        assert len(_delta) == self.n_parameters, "VertexPoint3 retraction Point3 inconsistent"
        self.data[:] += self.retract(_delta)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexPoint3 covariance update inconsistent"
        self.var = cov

    def retract(self, _delta):
        return _delta
    
    def x(self):
        return self.data[0]

    def y(self):
        return self.data[1]

    def z(self):
        return self.data[2]

    def echo_info(self):
        print("type:[%s] vals:[%.2f, %.2f, %.2f]"%(type_as_string(self.type),
            self.data[0], self.data[1], self.data[2]))

class VertexSO3(Vertex):
    def __init__(self, _id, _rot):
        self.id = _id
        self.type = BASE_TYPE.SO3
        self.n_parameters = 3
        self.data = _rot.data
        self.var = np.zeros((self.n_parameters,), dtype=float)
    
    def as_matrix(self):
        return self.data.as_matrix()

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSO3 retraction SO3 inconsistent"
        self.data = self.data @ self.retract(_twist)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSO3 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        return jxl.SO3.exp(_twist)

    def inverse(self):
        return self.data.inverse()
    
    def inverseretract(self):
        return self.data.log()

class VertexSE3(Vertex):
    def __init__(self, _id, _pose):
        self.id = _id;
        self.type = BASE_TYPE.SE3
        self.n_parameters = 6
        self.data = _pose.data
        self.var = np.zeros((self.n_parameters,), dtype=float)
    
    def translation(self):
        return self.data.translation()
    
    def rotation_as_matrix(self):
        return self.data.rotation().as_matrix()

    def rotation_as_quaternion(self):
        return self.data.rotation().as_quaternion_xyzw()

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSE3 retraction SE3 inconsistent"
        self.data = self.data @ self.retract(_twist)
    
    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSE3 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        return jxl.SE3.exp(_twist)

    def inverse(self):
        return self.data.inverse()
    
    def inverseretract(self):
        return self.data.log()