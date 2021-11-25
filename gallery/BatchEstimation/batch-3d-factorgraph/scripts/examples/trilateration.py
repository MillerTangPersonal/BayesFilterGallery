import numpy as np
import sys
sys.path.append("../core/")

from se_data_types import *
from se_vertex import *
from se_factors import *
from se_factor_graph import *

ANCHOR_MAP = np.array([[4., 4., 10.],
             [-4., 4., 10.],
             [-4., -4., 10.],
             [4., -4., 10.]])

class RangeFactor(Factor):
    def __init__(self, anc_pos, meas_range):
        self.n_parameters = 3
        self.n_residuals = 1
        self.anc_pos = anc_pos
        self.meas_range = meas_range
    
    def get_error(self, param_blocks, residuals):
        residuals[0] = self.meas_range - np.linalg.norm(self.anc_pos - param_blocks[0].data)
        return True

    def get_jacobian(self, param_blocks, jacobian_blocks):
        # TODO: check shape of jacobian
        diff = self.anc_pos - param_blocks[0].data
        pred_range = np.linalg.norm(diff)
        jacobian_blocks[0][0,0] = diff[0]/pred_range;
        jacobian_blocks[0][0,1] = diff[1]/pred_range;
        jacobian_blocks[0][0,2] = diff[2]/pred_range;
        return True
    
    def get_covariance(self, covariance):
        covariance[0] = 0.1;
        return True
    
    def echo_info(self):
        print("Anchor:[%.2f, %.2f, %.2f] range:[%.2f]"%(self.anc_pos[0], self.anc_pos[1],
            self.anc_pos[2], self.meas_range))

def sim_data(gt_pos, count):
    range_data = []
    anc_id = 0
    for idx in range(count):
        range_data.append([anc_id, np.linalg.norm(gt_pos - ANCHOR_MAP[anc_id, :])])
        anc_id = (anc_id + 1) % 4
    return range_data

if __name__ == "__main__":
    gt_pos = np.array([2.0, 2.0, 0.0])
    est_pos = np.array([0., 0., 0.])
    data = sim_data(gt_pos, 12)

    vertex_id = 0
    v1 = Vertex.create(vertex_id, Point3(est_pos))
    print("Initial values")
    v1.echo_info()

    options = SolverOptions("LM", iterations=5, linear_solver="CHOLESKY", lam=0)
    graph = FactorGraph(options);
    graph.add_vertex(v1)

    for rx in data:
        factor = RangeFactor(ANCHOR_MAP[rx[0]], rx[1])
        graph.add_factor(vertex_id, factor)
    
    graph.echo_info()
    graph.solve()

    v1.echo_info()