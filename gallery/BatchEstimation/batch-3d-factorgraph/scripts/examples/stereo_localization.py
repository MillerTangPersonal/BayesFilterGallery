import sys
sys.path.append("../core")

from se_data_types import *
from se_vertex import *
from se_factors import *
from se_factor_graph import *
import numpy as np
import math
import scipy.io as sio

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import rc


data = sio.loadmat('./data/dataset3.mat')

t_start = 1214
# t_end = 1216
t_end = 1714
# t_end = 1713

time_stamps = np.reshape(data["t"], -1)
# motion model
lin_vel = data["v_vk_vk_i"]
ang_vel = data["w_vk_vk_i"]
lin_var = data["v_var"]
ang_var = data["w_var"]
# camera to vehicle offset, required for observation model
cTv = jxl.SE3.from_rotation_and_translation(jxl.SO3.from_matrix(data["C_c_v"]),
    np.reshape(-1*np.dot(data["C_c_v"], data["rho_v_c_v"]), (3,)))

vertices = []
vertex_id_counter = 0

# create a prior
roll, pitch, yaw = data["theta_vk_i"][:, t_start];
px, py, pz = data["r_i_vk_i"][:, t_start]; 
prior_gt = Pose3(
    jxl.SE3.from_rotation_and_translation(
        jxl.SO3.from_rpy_radians(roll, pitch, yaw),
        np.array([px, py, pz])));

print(prior_gt.rotation().as_matrix())

options = SolverOptions("GN", iterations=3, cal_cov=False)
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
# # add motion model
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

    # input
    tau = np.concatenate((v,w), axis=0);
    curr_est = Pose3(prev_vertex.data @ jxl.SE3.exp(tau*dt))
    curr_vertex = Vertex.create(vertex_id_counter, curr_est)
    bet_fac = SE3BetweenFactorTwist(v, w, v_var, w_var, dt)
    # create a factor between current vertex and previous vertex
    graph.add_vertex(curr_vertex)
    # add the factor to problem
    graph.add_factor([prev_vertex.id, curr_vertex.id], bet_fac)
    # store vertices
    vertices.append(curr_vertex)
    t_prev = t_curr
    prev_vertex = curr_vertex
    vertex_id_counter += 1

# parameters related to obsevation model
cam_params = np.array([data['cu'], data['cv'], data['fu'], data['fv'], data['b']])
landmarks = data["rho_i_pj_i"]
meas_data = data["y_k_j"]
meas_var = np.reshape(data["y_var"], -1)

vertex_id_counter = 0
for idx in range(t_start, t_end):
    meas_datum = meas_data[:, idx, :]
    for meas_idx in range(np.shape(meas_datum)[1]):
        if not np.sum(meas_datum[:, meas_idx]) == -4:
            stereo_factor = StereoFactor(cam_params, # intrinsic parameters 
                cTv, # extrinsic parameters
                landmarks[:, meas_idx], # landmark positions
                meas_datum[:,meas_idx], # measured pixel values
                meas_var) # measurement variance
            graph.add_factor(vertex_id_counter, stereo_factor)
    vertex_id_counter += 1

graph.echo_info()
graph.solve(); 

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

######
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

fig = plt.figure(0)
ax = fig.gca(projection='3d')
# Plot estimate position
ax.plot(traj[:,0], traj[:,1], traj[:,2], color='b', 
    label='Ground Truth')
ax.plot(traj[:, 3], traj[:, 4], traj[:, 5], color='g', 
    label='Estimate')
ax.scatter(landmarks[0,:], landmarks[1,:], landmarks[2,:], 
    marker='o', color='r',label='Landmarks')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Stereo based localization')
ax.legend()

plt.show()
