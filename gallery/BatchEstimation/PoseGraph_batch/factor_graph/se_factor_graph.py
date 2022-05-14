# import jaxlie as jxl
import numpy as np
import time

from se_data_types import *
from se_vertex import *
from se_factors import *

SOLVERS = ["GN", "LM"]
DEBUG = False

class SolverOptions:
    def __init__(self, solver, linear_solver = "QR", iterations = 5, cal_cov = False, lam = 1.0):
        assert solver in SOLVERS, "Solver:" + solver + " not supported. Options:" + ''.join(SOLVERS)
        self.solver = solver
        self.iterations = iterations
        self.cal_cov = cal_cov
        self.lam = lam                       # for LevenbergMarquardt
        self.linear_solver = linear_solver
        print("Solver options:")
        print("Nonlinear solver:[%s] iterations:[%d] lam:[%.3f] linear solver:[%s]"%(self.solver,
            self.iterations, self.lam, self.linear_solver))

class FactorGraph:
    def __init__(self, _options):
        self.n_parameters = 0
        self.n_vertices = 0
        self.n_factors = 0
        self.n_residuals = 0
        self.factor_id = 0
        self.options = _options

        self.vertices = {}                  # dictionary
        self.factor_list = []
        self.factor_vertex_map = {}

    def add_vertex(self, vertex):
        if not vertex.id in self.vertices:
            # print("Adding new vertex: id:[%d] type:[%s]"%(vertex.id,
                # type_as_string(vertex.type)))
            self.vertices[vertex.id] = vertex
            self.n_parameters += vertex.n_parameters
            self.n_vertices += 1
        else:
            print("Vertex already exists.")

    # TODO: add another method that takes in vertices
    def add_factor(self, vertex_ids, factor):
        if not isinstance(vertex_ids, list):
            vertex_ids = [vertex_ids]

        for idx in vertex_ids:
            if idx not in self.vertices:
                print("Vertex with id:[%d] does not exist. Not adding factor", vertex_ids);
                return

        # print("Adding factor ")
        factor.id = self.factor_id
        self.factor_vertex_map[factor.id] = vertex_ids
        self.factor_list.append(factor)
        self.factor_id += 1
        self.n_factors += 1
        self.n_residuals += factor.n_residuals


    def remove_vertex(self, vertex):
        if vertex.id in self.vertices:
            # pop vertices using key od dict.
            self.vertices.pop(vertex.id)
            self.n_parameters -= vertex.n_parameters
            self.n_vertices -= 1
        else:
            print("Vertex does not exist.")

    def remove_factor(self, vertex):
        # TODO: not considering loop-closure for now
        # remove all the factor connected to vertex.id
        vertex_id = [vertex.id]          

        vertex_list1 = [vertex.id - 1, vertex.id]
        vertex_list2 = [vertex.id,     vertex.id + 1]

        # find factor id: get key from values
        v_id1 = list(self.factor_vertex_map.values()).index(vertex_id)
        factor_id1 = list(self.factor_vertex_map.keys())[v_id1]
        self.factor_vertex_map.pop(factor_id1)

        v_id2 = list(self.factor_vertex_map.values()).index(vertex_list1)
        factor_id2 = list(self.factor_vertex_map.keys())[v_id2]
        self.factor_vertex_map.pop(factor_id2)

        v_id3 = list(self.factor_vertex_map.values()).index(vertex_list2)
        factor_id3 = list(self.factor_vertex_map.keys())[v_id3]
        self.factor_vertex_map.pop(factor_id3)

        # remove factors from list using factor id
        for factor in self.factor_list:
            if factor.id in [factor_id1, factor_id2, factor_id3]:
                self.factor_list.remove(factor)      
                self.n_factors -= 1
                self.n_residuals -= factor.n_residuals


    def echo_info(self):
        print("Problem has [%d] vertices with [%d] parameters and [%d] factors."%(self.n_vertices,
            self.n_parameters, self.n_factors))

    def linearize(self, J, W_inv, residuals):
        vi_map = {}
        idy = 0
        for it in range(self.n_vertices):
            offset = self.vertices[it].n_parameters
            vi_map[it] = [idy, idy + offset]
            idy += offset

        # for each factor, compute jacobians and residuals
        row_counter = 0
        for f_id, factor in enumerate(self.factor_list):
            #print("Processing factor:[%d] [%d]"%(factor.id, factor.n_residuals))
            # factor.echo_info();
            row_i, row_j = row_counter, row_counter + factor.n_residuals
            # each factor may depend on multiple parameters/vertices
            param_blocks = []
            jacobian_blocks = []
            for v_id in self.factor_vertex_map[factor.id]:
                col_i, col_j = vi_map[v_id]
                param_blocks.append(self.vertices[v_id])
                jacobian_blocks.append(J[row_i:row_j, col_i:col_j])

            factor.get_error(param_blocks, residuals[row_i:row_j])
            factor.get_jacobian(param_blocks, jacobian_blocks)
            factor.get_covariance(W_inv[row_i:row_j, row_i:row_j])
            row_counter += factor.n_residuals

    def loss(self):
        residuals = np.zeros((self.n_residuals,), dtype=float)
        vi_map = {}
        idy = 0
        for it in range(self.n_vertices):
            offset = self.vertices[it].n_parameters
            vi_map[it] = [idy, idy + offset]
            idy += offset
        # for each factor, compute jacobians and residuals
        row_counter = 0
        for f_id, factor in enumerate(self.factor_list):
            #print("Processing factor:[%d] [%d]"%(factor.id, factor.n_residuals))
            # factor.echo_info();
            row_i, row_j = row_counter, row_counter + factor.n_residuals
            # each factor may depend on multiple parameters/vertices
            param_blocks = []
            for v_id in self.factor_vertex_map[factor.id]:
                col_i, col_j = vi_map[v_id]
                param_blocks.append(self.vertices[v_id])

            factor.get_error(param_blocks, residuals[row_i:row_j])
            row_counter += factor.n_residuals
        return np.sum(residuals**2)

    def GaussNewtonSolver(self):
        for iteration in range(self.options.iterations):
            print("Gauss Newton, Iteration:[%d]"%iteration)
            # jacobian
            J = np.zeros((self.n_residuals, self.n_parameters), dtype=float)
            # inverse covariance/information
            W_inv = np.identity(self.n_residuals, dtype=float)
            # vector of residuals
            residuals = np.zeros((self.n_residuals,), dtype=float)
            # -------- linearize around current estimate
            start = time.process_time()
            self.linearize(J, W_inv, residuals)
            print("Linearize took:", time.process_time() - start)
            # import matplotlib.pyplot as plt
            # plt.imshow(J, interpolation="nearest")
            # plt.show()
            A_ = J.T @ W_inv @ J
            b_ = J.T @ W_inv @ residuals
            # do cholesky decomposition
            L_ = np.linalg.cholesky(A_)
            # do forward pass
            d_ = np.linalg.solve(L_, b_)
            # do backward pass
            delta = np.linalg.solve(L_.T, d_)

            # solve for covariance
            if self.options.cal_cov:
                
                if DEBUG:
                    tic = time.time()                                             #### Delete this line after testing
                    A_inv_ = np.linalg.inv(A_)                                    #### Delete this line after testing
                    print("\nDirect inverse of A Took: ", time.time() - tic, 's') #### Delete this line after testing
                
                
                P_list_ = [] # note: the P matrix stored in this list is in a reverse order, i.e., P_K, P_K-1, ..., P_1
                n_rows, n_cols = L_.shape # extract the shape of the L_ matrix
                # the final P block only relates to the bottom right block in the L_ matrix
                offset = self.vertices[self.n_vertices-1].n_parameters       # offset related to the final vertex
                L_k_inv_ = L_[n_rows-offset:n_rows, n_cols-offset:n_cols]    # extract the bottom-right block from L_
                L_k_inv_ = np.linalg.inv(L_k_inv_)                           # inverse of bottom right block in L_ matrix

                Ppo = L_k_inv_.T @ L_k_inv_
                P_list_.append(Ppo)                                          # append to the storage list, P_k's are stored in a reversed order!
                # set the cov
                self.vertices[self.n_vertices-1].set_cov(Ppo)

                for it in reversed(range(0, self.n_vertices-1)): # this will start at self.n_vertices-2, and ends at 0
                    offset = self.vertices[it].n_parameters # offset related to the it^th vertex
                    L_k_inv_ = np.linalg.inv(L_[offset*it:offset*it+offset, offset*it:offset*it+offset])
                    L_k_k_1_ = L_[offset*it+offset : offset*(it+2), offset*it : offset*it+offset] # the block below L_k_inv_

                    Ppo = L_k_inv_.T @ (np.identity(offset)+ L_k_k_1_.T @ P_list_[-1] @ L_k_k_1_) @ L_k_inv_
                    P_list_.append(Ppo)
                    # set the cov
                    self.vertices[it].set_cov(Ppo)
                
                if DEBUG:
                    ## Checking accuracy against direct inverse of A
                    dif = 0.0
                    for it in range(0, self.n_vertices):
                        offset = self.vertices[it].n_parameters
                        len_P_list = len(P_list_)
                        dif += np.abs(np.sum( A_inv_[it*offset : it*offset+offset, it*offset : it*offset+offset] - P_list_[len_P_list-it-1]))
                    print("\nThe sum of absolute difference between direct inverse method and TDB method: ", dif,"\n")
                
            
            idy = 0
            start = time.process_time()
            for it in range(self.n_vertices):
                offset = self.vertices[it].n_parameters
                # print("Vertex:", it," delta:", delta[idy : idy+offset])
                self.vertices[it].update(delta[idy : idy + offset])
                # if self.options.cal_cov and len(P_) > 0:
                #     print("Not implemented yet")
                #     self.vertices[it].update_cov(delta[i_:j_])

                idy += offset
            print("Update took:", time.process_time() - start)
            loss = self.loss()
            print("Loss:", loss)


    def LevenbergMarquardt(self):
        for iteration in range(self.options.iterations):

            # TODO: Adjust lambda
            print("Levenberg-Marquardt, Iteration:[%d]"%iteration)
            # jacobian
            J = np.zeros((self.n_residuals, self.n_parameters), dtype=float)
            # inverse covariance
            W_inv = np.identity(self.n_residuals, dtype=float)
            # vector of residuals
            residuals = np.zeros((self.n_residuals,), dtype=float)
            # estimated covariance
            P_ = []
            # linearize around current estimate
            start = time.process_time()
            self.linearize(J, W_inv, residuals)
            print("Linearize took:", time.process_time() - start)

            A_ = J.T @ W_inv @ J
            # LM
            A_ += (self.options.lam * np.identity(A_.shape[0]))
            # residuals
            b_ = J.T @ W_inv @ residuals
            # 
            if self.options.linear_solver == "QR":
                # QR decomposition
                Q_, R_ = np.linalg.qr(A_)
                # forward pass
                d_ = Q_.T @ b_
                # backward pass
                delta = np.linalg.solve(R_, d_)
                # TODO: solve for covariance
                if self.options.cal_cov:
                    u_ = np.linalg.solve(R_, np.identity(self.n_residuals, dtype=float))
                    P_ = np.linalg.solve(R_.T, u_)
            elif self.options.linear_solver == "Cholesky":
                # choleskey decomposition
                L_ = np.linalg.cholesky(A_)
                # forward pass
                d_ = np.linalg.solve(L_, b_)
                # backward pass
                delta = np.linalg.solve(L_.T, d_)
                # solve for covariance
                if self.options.cal_cov:
                    u_ = np.linalg.solve(L_, np.identity(self.n_residuals, dtype=float))
                    P_ = np.linalg.solve(L_.T, u_)                    

            idy = 0
            start = time.process_time()
            for it in range(self.n_vertices):
                offset = self.vertices[it].n_parameters
                # print("Vertex:", it," delta:", delta[idy : idy+offset])
                self.vertices[it].update(delta[idy : idy + offset])
                if self.options.cal_cov and len(P_) > 0:
                    print("Not implemented yet")
                    #self.vertices[it].update_cov(delta[i_:j_])
                idy += offset
            print("Update took:", time.process_time() - start)
            loss = self.loss()
            print("Loss:", loss)
 

    def solve(self):
        if self.options.solver == "GN":
            self.GaussNewtonSolver()
        elif self.options.solver == "LM":
            self.LevenbergMarquardt()
