'''
    stereo camera based localization using pose graph sliding window filter
'''
import sys

from numpy import dtype
sys.path.append("./factor_graph")
from factor_graph.se_data_types import *
from factor_graph.se_vertex import *
from factor_graph.se_factors import *
from factor_graph.se_factor_graph import *
from factor_graph.se_utils import axisAngle_to_Rot, getTrans
from vis_util import visual_traj, visual_est

import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt

data = sio.loadmat('./data/dataset3.mat')


time_stamps = np.reshape(data["t"], -1)
# motion model
lin_vel = data["v_vk_vk_i"]
ang_vel = data["w_vk_vk_i"]
lin_var = data["v_var"]
ang_var = data["w_var"]


t_init = 1214
# t_end   = 1714
kappa   = 50

# ------------- create a prior ------------- #
axisAngle = data["theta_vk_i"][:, t_init]   # C_v0_i
r_gt0     = data["r_i_vk_i"][:, t_init]     # r_i^{v0i}

C_gt0 = axisAngle_to_Rot(axisAngle)          # C_v0_i
prior_gt = Pose3(getTrans(C_gt0, r_gt0))     # T_v0_i

# data struture: SE3 is a numpy array [4x4]
# solver = "GN" or "LM"
# linear_solver = "QR" or "Cholesky"
options = SolverOptions(solver="GN", iterations=1, linear_solver = "Cholesky", cal_cov=True)   
graph = FactorGraph(options)

# init vertices
vertices = []
vertex_id_counter = 0

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
t_prev = time_stamps[t_init]
prev_vertex = prior_vertex

t_start = t_init

# for ---
    # t_start = t_start + 1
    # t_end   = t_start + kappa
    
    # add prior factor into the graph

    # init vertices
    # vertices = []
    # vertex_id_counter = 0

    # graph = FactorGraph(options)



# prior
for idx in range(t_start+1, t_end):
    # generate prior estimate
    # follow prof. Barfoot's convention
    v = -1.0 * lin_vel[:, idx-1] 
    w = -1.0 * ang_vel[:, idx-1]
    v_var = np.reshape(lin_var, -1)   # 1x3
    w_var = np.reshape(ang_var, -1)

    t_curr = time_stamps[idx]
    dt = t_curr - t_prev
    # Pose graph motion model
    input = np.block([v, w])    # shape of (6,)
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

# measurements
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

# solve factor graph
graph.echo_info()
graph.solve()
    

