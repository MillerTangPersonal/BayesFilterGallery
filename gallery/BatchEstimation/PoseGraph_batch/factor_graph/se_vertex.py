# import jaxlie as jxl
import numpy as np
from scipy import linalg
from se_data_types import BASE_TYPE, type_as_string
from se_utils import skew, axisAngle_from_rot

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
            assert 0, "Unkown type request for vertex creation"

        create = staticmethod(create)

# define each type of vertex

class VertexScalar(Vertex):
    def __init__(self, _id, _scalar):
        self.id = _id
        self.type = BASE_TYPE.R0
        self.n_parameters = 1
        self.data = _scalar.data
        self.var = np.zeros((self.n_parameters,), dtype = float)

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
        self.var = np.zeros((self.n_parameters,), dtype = float)

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
    pass

class VertexSE2(Vertex):
    pass

class VertexPoint3(Vertex):
    def __init__(self, _id, _point):
        self.id = _id
        self.type = BASE_TYPE.R3
        self.n_parameters = 3
        self.data = _point.data
        self.var = np.zeros((self.n_parameters,), dtype = float)

    def norm(self):
        return np.linalg.norm(self.data)

    def update(self, _delta):
        assert len(_delta) == self.n_parameters, "VertexPoint3 retraction Point3 inconsistent"
        self.data[:] += self.retract(_delta)

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexPoints covariance update inconsistent"
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

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSO3 retraction SO3 inconsistent"
        # self.data = self.data @ self.retract(_twist)
        raise Exception("Called SO3 update")
        pass

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSO3 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        # return jxl.SO3.exp(_twist)
        raise Exception("Called SO3 retract")
        pass

    def inverse(self):
        # return self.data.inverse()
        raise Exception("Called SO3 inverse")
        pass

    def inverseretract(self):
        # return self.data.log()
        raise Exception("Called SO3 inverseretract")
        pass


class VertexSE3(Vertex):
    def __init__(self, _id, _pose):
        self.id = _id
        self.type = BASE_TYPE.SE3
        self.n_parameters = 6
        self.data = _pose.data
        self.var = np.zeros((self.n_parameters), dtype = float)

    def translation(self):
        return self.data[0:3, 3]

    def rotation_as_matrix(self):
        return self.data[0:3,0:3]

    # def rotation_as_quaternion(self):
    #     return self.data.rotation().as_quaternion_xyzw()

    def update(self, _twist):
        assert len(_twist) == self.n_parameters, "VertexSE3 retraction SE3 inconsistent"
        self.data =  self.retract(_twist) @ self.data

    def set_cov(self, cov):
        assert len(cov) == self.n_parameters, "VertexSE3 covariance update inconsistent"
        self.var = cov

    def retract(self, _twist):
        rho = _twist[0:3].reshape(-1,1)
        phi = _twist[3:6]
        phi_skew = skew(phi)
        zeta = np.block([
            [phi_skew, rho],
            [0,  0,  0,  0]
            ])
        Psi = linalg.expm(zeta)
        return Psi

    def inverse(self):
        return np.linalg.inv(self.data)

    def inverseretract(self):
        C_data = self.data.rotation()
        r_data = self.data.translation()
        # Equ. (60) to compute J. Then, J * rho = r
        # could be done by eigenvalue
        axisAngle = np.squeeze(axisAngle_from_rot(C_data))
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
        # J * rho = r_now
        r_data = r_data.reshape(-1,1)
        rho_data = np.squeeze(np.linalg.solve(J, r_data)) # [3,]
        phi_data = np.squeeze(axisAngle[3] * a) # [3,]
        return np.block([rho_data, phi_data])    # shape(6)
