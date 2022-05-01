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
from scipy import linalg
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
axisAngle = data["theta_vk_i"][:, t_start] # C_v0_i
r_gt0     = data["r_i_vk_i"][:, t_start]   # r_i^{v0i}

C_gt0 = axisAngle_to_Rot(axisAngle)        # C_v0_i
prior_gt = Pose3(getTrans(C_gt0, r_gt0))   # T_v0_i

# datastruture: SE3 is a numpy array [4x4]

options = SolverOptions("GN", iterations=5, cal_cov=False)
graph = FactorGraph(options)

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
    v = lin_vel[:, idx-1]
    w = ang_vel[:, idx-1]
    v_var = np.reshape(lin_var, -1)   # 1x3
    w_var = np.reshape(ang_var, -1)

    t_curr = time_stamps[idx]
    dt = t_curr - t_prev
    # Pose graph motion model
    input = np.block([-1.0 * v, -1.0 * w])    # shape of (6,)
    rho = input[0:3].reshape(-1,1)
    phi = input[3:6]
    phi_skew = skew(phi)
    zeta = np.block([
        [phi_skew, rho],
        [0,  0,  0,  0]])
    Psi = linalg.expm(dt*zeta)
    # --------------------------------------- #
    curr_est = Pose3(Psi @ prev_vertex.data)
    curr_vertex = Vertex.create(vertex_id_counter, curr_est)
    bet_fac = SE3BetweenFactorTwist(v, w, v_var, w_var, dt)
    # create a factor between current vertex and previous vertex
    graph.add_vertex(curr_vertex)
    # add the factor to problem
    graph.add_factor([prev_vertex.id, curr_vertex.id], bet_fac)
    # store vertices
    vertices.append(curr_vertex)
    # update t, prev_vertex, id_counter
    t_prev = t_curr
    prev_vertex = curr_vertex
    vertex_id_counter += 1


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
graph.solve()

# estimated traj. and ground_truth
traj = np.zeros((len(vertices), 6), dtype=float)
for idx, idy in enumerate(range(t_start, t_end)):
    gt = data['r_i_vk_i'][:, idy]
    #
    est_C = vertices[idx].rotation_as_matrix()
    est_r = -1.0 * est_C.T @ vertices[idx].translation()
    # print("Vertex:[%d] gt:[%.3f, %.3f, %.3f] est:[%.3f, %.3f, %.3f]"%(idx,
    #     gt[0], gt[1], gt[2], est[0], est[1], est[2]));
    traj[idx, :] = [gt[0], gt[1], gt[2], est_r[0], est_r[1], est_r[2]]

print("Error: x: mu:[%.3f] std:[%.3f] y: mu:[%.3f] std:[%.3f] z: mu:[%.3f] std:[%.3f]"%(
    np.mean(traj[:,0] - traj[:,3]), np.std(traj[:,0] - traj[:,3]),
    np.mean(traj[:,1] - traj[:,4]), np.std(traj[:,1] - traj[:,4]),
    np.mean(traj[:,2] - traj[:,5]), np.std(traj[:,2] - traj[:,5])))

visual_traj(traj, landmarks)

plt.show()
