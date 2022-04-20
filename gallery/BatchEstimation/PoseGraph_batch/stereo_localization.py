import sys
sys.path.append("./factor_graph")
from factor_graph.se_data_types import *
from factor_graph.se_vertex import *
from factor_graph.se_factors import *
from factor_graph.se_factor_graph import *
from factor_graph.se_utils import axisAngle_to_Rot
from vis_util import visual_traj

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


data = sio.loadmat('./data/dataset3.mat')

t_start = 1214
t_end = 1714

time_stamps = np.reshape(data["t"], -1)
# motion model
lin_vel = data["v_vk_vk_i"]
ang_vel = data["w_vk_vk_i"]
lin_var = data["v_var"]
ang_var = data["w_var"]

vertices = []
vertex_id_counter = 0

# ------------- create a prior ------------- # 
roll, pitch, yaw = data["theta_vk_i"][:, t_start];
px, py, pz = data["r_i_vk_i"][:, t_start]; 

# test with jaxlie
# TODO: need to be tested carefully, rotation is not the same
C_gt_test = jxl.SO3.from_rpy_radians(roll, pitch, yaw)
C_gt0 = axisAngle_to_Rot(np.array([roll, pitch, yaw]))
print(C_gt_test.as_matrix())
print('\n')
print(C_gt0)
prior_gt = Pose3()

# ------------------------------------------ #
options = SolverOptions("GN", iterations=3, cal_cov=True)
graph = FactorGraph(options);

prior_vertex = Vertex.create(vertex_id_counter, prior_gt)
prior_factor = SE3PriorFactor(prior_gt, 0.1*np.ones((6,), dtype=float))

# prior vertex
graph.add_vertex(prior_vertex)
# prior factor
graph.add_factor(vertex_id_counter, prior_factor)
# store for future analysis
vertices.append(prior_vertex)

vertex_id_counter += 1
# add motion model
t_prev = time_stamps[t_start]
prev_vertex = prior_vertex

for idx in range(t_start+1, t_end):
    # generate prior estimate
    v = np.reshape(lin_vel[:, idx-1], -1)
    w = np.reshape(ang_vel[:, idx-1], -1)
    v_var = np.reshape(lin_var, -1)
    w_var = np.reshape(ang_var, -1)

    t_curr = time_stamps[idx]
    dt = t_curr - t_prev

    # TODO: careful about the convention
    pass


# parameters related to observation model
cam_params = np.array([data['cu'], data['cv'], data['fu'], data['fv'], data['b']])
landmarks = data["rho_i_pj_i"]
meas_data = data["y_k_j"]
meas_var = np.reshape(data["y_var"], -1)

C_c_v = data["C_c_v"]
rho_v_c_v = data["rho_v_c_v"]

vertex_id_counter = 0
for idx in range(t_start, t_end):
    meas_datum = meas_data[:, idx, :]
    for meas_idx in range(np.shape(meas_datum)[1]):
        if not np.sum(meas_datum[:, meas_idx]) == -4:
            stereo_factor = StereoFactor(cam_params, # intrinsic parameters 
                C_c_v,      # extrinsic rotation
                rho_v_c_v,  # extrinsic translation
                landmarks[:, meas_idx], # landmark positions
                meas_datum[:,meas_idx], # measured pixel values
                meas_var) # measurement variance
            graph.add_factor(vertex_id_counter, stereo_factor)
    vertex_id_counter += 1

graph.echo_info()
graph.solve(); 

# estimated traj. and ground_truth 
traj = np.zeros((len(vertices), 6), dtype=float)
for idx, idy in enumerate(range(t_start, t_end)):
    gt = data['r_i_vk_i'][:, idy]
    est = vertices[idx].translation()
    # print("Vertex:[%d] gt:[%.3f, %.3f, %.3f] est:[%.3f, %.3f, %.3f]"%(idx,
    #     gt[0], gt[1], gt[2], est[0], est[1], est[2]));
    traj[idx, :] = [gt[0], gt[1], gt[2], est[0], est[1], est[2]]; 

print("Error: x: mu:[%.3f] std:[%.3f] y: mu:[%.3f] std:[%.3f] z: mu:[%.3f] std:[%.3f]"%(
    np.mean(traj[:,0] - traj[:,3]), np.std(traj[:,0] - traj[:,3]),
    np.mean(traj[:,1] - traj[:,4]), np.std(traj[:,1] - traj[:,4]),
    np.mean(traj[:,2] - traj[:,5]), np.std(traj[:,2] - traj[:,5])))

visual_traj(traj, landmarks)

plt.show()


