import jaxlie as jxl
import numpy as np

from se_data_types import *
from se_vertex import *
from se_factors import *
import time

SOLVERS = ["GN", "LM"]
np.set_printoptions(linewidth=10000)
np.set_printoptions(precision=1)
class SolverOptions:
    def __init__(self, solver, linear_solver="QR", iterations=5, cal_cov=False, lam=1.0):
        assert solver in SOLVERS, "Solver:" + solver + " not supported. Options:"  + ''.join(SOLVERS) 
        self.solver = solver
        self.iterations = iterations
        self.cal_cov = cal_cov
        self.lam = lam
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

        self.vertices = {}
        self.factor_list = []
        self.factor_vertex_map = {}

    def add_vertex(self, vertex):
        if not vertex.id in self.vertices:
            # print("Adding new vertex: id:[%d] type:[%s]"%(vertex.id,
                # type_as_string(vertex.type)))
            self.vertices[vertex.id] = vertex
            self.n_parameters += vertex.n_parameters;
            self.n_vertices += 1
        else:
            print("Vertex already exists.")

    # TODO: add another method that takes in vertices
    def add_factor(self, vertex_ids, factor):
        if not isinstance(vertex_ids, list):
            vertex_ids = [vertex_ids]

        for idx in vertex_ids:
            if idx not in self.vertices:
                print("Vertex with id:[%d] does not exist. Not adding factor", param_id);
                return

        # print("Adding factor ")
        factor.id = self.factor_id;
        self.factor_vertex_map[factor.id] = vertex_ids
        self.factor_list.append(factor)
        self.factor_id += 1
        self.n_factors += 1
        self.n_residuals += factor.n_residuals 
    
    def echo_info(self):
        print("Problem has [%d] vertices with [%d] parameters and [%d] factors."%(self.n_vertices,
            self.n_parameters, self.n_factors))

    def linearize(self, J, W, residuals):
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
            factor.get_covariance(W[row_i:row_j, row_i:row_j])
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
            print("Gauss Newton Iteration:[%d]"%iteration)
            # jacobian
            J = np.zeros((self.n_residuals, self.n_parameters), dtype=float)
            # inverse covariance/information
            W = np.identity(self.n_residuals, dtype=float)
            # vector of residuals
            residuals = np.zeros((self.n_residuals,), dtype=float)
            # estimated covariance
            P_ = []
            # linearize arround current estimate
            start = time.process_time()
            self.linearize(J, W, residuals)
            print("Linearize took:", time.process_time() - start)
            print(J)
            import matplotlib.pyplot as plt
            plt.imshow(J, interpolation="nearest")
            plt.show()
            A_ = np.dot(np.dot(np.transpose(J), W), J)
            b_ = -np.dot(np.dot(np.transpose(J), W), residuals)
            # do cholesky decompositoin
            L_ = np.linalg.cholesky(A_)
            # do forward pass
            d_ = np.linalg.solve(L_, b_)
            # do backward pass
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

 
    def LevenbergMarquardt(self):
        for iteration in range(self.options.iterations):

            # if iteration > 3:
            #     self.options.lam = 0.1;

            # TODO: Adjust lambda
            print("Levenberg-Marquardt Iteration:[%d]"%iteration)
            # jacobian
            J = np.zeros((self.n_residuals, self.n_parameters), dtype=float)
            # inverse covariance/information
            W = np.identity(self.n_residuals, dtype=float)
            # vector of residuals
            residuals = np.zeros((self.n_residuals,), dtype=float)
            # estimated covariance
            P_ = []
            # linearize arround current estimate
            start = time.process_time()
            self.linearize(J, W, residuals)
            print("Linearize took:", time.process_time() - start)

            A_ = np.dot(np.dot(np.transpose(J), W), J)
            # LM 
            A_ += (self.options.lam * np.identity(A_.shape[0]))
            # residuals
            b_ = -np.dot(np.dot(np.transpose(J), W), residuals)
            #
            if self.options.linear_solver == "QR":
                # do cholesky decompositoin
                Q_, R_ = np.linalg.qr(A_)
                # do forward pass
                d_ = np.dot(Q_.T, b_)
                # do backward pass
                delta = np.linalg.solve(R_, d_)
                # solve for covariance
                # TODO: Fix this (this is not correct)
                if self.options.cal_cov:
                    u_ = np.linalg.solve(R_, np.identity(self.n_residuals, dtype=float))
                    P_ = np.linalg.solve(R_.T, u_)
            else:
                # do cholesky decompositoin
                L_ = np.linalg.cholesky(A_)
                # do forward pass
                d_ = np.linalg.solve(L_, b_)
                # do backward pass
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