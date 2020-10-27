from fenics import *

import argparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import h5py

from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve, eigsh
from scipy.stats import norm, truncnorm

import statfenics as sf


# setup
np.random.seed(27)
logger = logging.getLogger(__name__)
set_log_level(40)
parameters["reorder_dofs_serial"] = False


# can't use statfenics interpolation matrix due to mixed FunctionSpace
# this only extracts the wave components (not the second order derivatative)
def build_interpolation_matrix(x_obs, V):
    """ Build interpolation matrix from `x_obs` on function space V.
    From the fenics forums.
    """
    nx, dim = x_obs.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    dolfin_element = V.dolfin_element()
    dofmap = V.dofmap()
    bbt = mesh.bounding_box_tree()

    sdim = dolfin_element.space_dimension()

    v = np.zeros(sdim)
    rows = np.zeros(nx * sdim, dtype='int')
    cols = np.zeros(nx * sdim, dtype='int')
    vals = np.zeros(nx * sdim)

    # Loop over all interpolation points
    for k in range(nx):
        x = x_obs[k, :]
        if dim == 1:
            p = Point(x[0])
        elif dim == 2:
            p = Point(x[0], x[1])
        elif dim == 3:
            p = Point(x[0], x[1], x[2])
        else:
            logger.error("no support for higher dims")
            raise ValueError

        # find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)

        # vertex coordinates for the cell
        xvert = coords[cells[cell_id, :], :]

        # evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x, xvert, cell_id)

        # set the sparse metadata
        jj = np.arange(sdim * k, sdim * (k + 1))
        rows[jj] = k
        cols[jj] = dofmap.cell_dofs(cell_id)
        vals[jj] = v[::2]

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    M = csr_matrix((vals, ij), shape=(nx, V.dim()))
    return(M)


# Sub domain for periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 32 * np.pi


class KSSolve():
    def __init__(self, dt, nu=1.):
        self.dt = dt
        self.mesh = IntervalMesh(512, 0., 32 * np.pi)
        P1 = FiniteElement('P', interval, 1)
        element = MixedElement([P1, P1])
        V_base = self.V_base = FunctionSpace(self.mesh, P1,
                                             constrained_domain=PeriodicBoundary())
        V = self.V = FunctionSpace(self.mesh, element,
                                   constrained_domain=PeriodicBoundary())

        # solution function
        self.u = Function(V)
        u1, u2 = split(self.u)

        # variational functions
        self.du = TrialFunction(V)
        v1, v2 = TestFunctions(V)

        # initial value
        u_init = Expression(("sin(x[0] / 16)",
                             "-sin(x[0] / 16) / pow(16, 2)"), degree=8)
        self.u_prev = interpolate(u_init, V)
        u1_prev, u2_prev = split(self.u_prev)

        # variational definition
        F1 = ((u1 - u1_prev) / dt * v1 * dx
            - nu * u1.dx(0) * v1.dx(0) * dx
            - u2.dx(0) * v1.dx(0) * dx
            + u1 * u1.dx(0) * v1 * dx)
        F2 = -u1.dx(0) * v2.dx(0) * dx - u2 * v2 * dx
        self.F = F1 + F2
        self.J = derivative(self.F, self.u, self.du)
        self.J_prev = derivative(self.F, self.u_prev, self.du)

    def get_u(self):
        return(self.u.vector()[:])

    def timestep(self):
        solve(self.F == 0, self.u, J=self.J)
        self.u_prev.assign(self.u)


class StatKS(KSSolve):
    def __init__(self, dt, nu=1.):
        super().__init__(dt, nu)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.M = sf.dolfin_to_csr(assemble(inner(u, v) * dx))

        # permutation matrix to get the appropriate covariance structure
        # P: pair ordering (u1_dofs, u2_dofs) -> (fenics dof ordering)
        u1_dof_indices = self.V.sub(0).dofmap().dofs()
        u2_dof_indices = self.V.sub(1).dofmap().dofs()
        nx = 512
        P = np.zeros((2 * nx, 2 * nx))
        indices = np.concatenate((u1_dof_indices, u2_dof_indices))
        for i, j in enumerate(indices):
            P[i, j] = 1.

        self.P = P.T
        self.priors = {"scale_G_mean": 1, "scale_G_sd": 1,
                       "length_G_mean": 1, "length_G_sd": 1,
                       "sigma_y_mean": 0, "sigma_y_sd": 1}

        self.mean = np.copy(self.u_prev.vector()[:])
        self.cov = np.zeros_like(P)
        self.params = [np.random.uniform(0, 0.01),
                       np.random.uniform(0, 0.01),
                       np.random.uniform(0, 0.01)]
        self.nugget = 1e-10
        self.n_eigen = 50

    def timestep(self):
        solve(self.F == 0, self.u, J=self.J)
        self.mean = np.copy(self.u.vector()[:])

        # prediction covariance
        self.J_scipy = sf.dolfin_to_csr(assemble(self.J))
        self.J_prev_scipy = sf.dolfin_to_csr(assemble(self.J_prev))
        values, vectors = eigsh(self.cov, k=self.n_eigen)
        temp = spsolve(self.J_scipy, (self.J_prev_scipy @ vectors))
        self.cov = temp @ diags(values) @ temp.T
        self.u_prev.assign(self.u)

    def log_marginal_posterior(self, params, M, V, H, J_curr, mean_obs, cov_obs, x_obs, y_obs):
        scale_G, length_G, sigma_y = params
        priors = self.priors

        u1_dof_indices = self.V.sub(0).dofmap().dofs()
        x_u1 = self.V.tabulate_dof_coordinates()[u1_dof_indices]
        K = sf.cov_square_exp(x_u1, scale_G, length_G)
        K = np.kron([[1, 0], [0, 0]], K)
        G = self.dt * self.M @ self.P @ K @ self.P.T @ self.M.T

        values, vectors = eigsh(G, k=self.n_eigen)
        temp = spsolve(J_curr, vectors)
        temp_obs = H @ temp
        G_obs = temp_obs @ np.diag(values) @ temp_obs.T

        S = cov_obs + G_obs + (self.nugget + sigma_y**2) * np.eye(len(x_obs))
        S_chol = cholesky(S, lower=True)
        S_inv = cho_solve((S_chol, True), np.eye(len(x_obs)))
        alpha = S_inv @ (y_obs - mean_obs)
        alpha_alpha_T = np.outer(alpha, alpha)

        dist = pdist(x_u1, metric="sqeuclidean")
        dist = squareform(dist)

        S_dee_scale_G = 2 * G_obs / scale_G
        S_dee_sigma_y = 2 * sigma_y * np.eye(len(x_obs))

        K = sf.cov_square_exp(x_u1, scale_G, length_G)
        G_base_dee_length = np.kron([[1, 0], [0, 0]], dist / length_G**3 * K)
        G_dee_length = self.dt * self.M @ self.P @ G_base_dee_length @ self.P.T @ self.M.T
        values, vectors = eigsh(G_dee_length, k=self.n_eigen)
        temp = spsolve(J_curr, self.P @ vectors)
        temp_obs = H @ temp

        S_dee_length_G = temp_obs @ np.diag(values) @ temp_obs.T

        log_det = 2 * np.sum(np.log(np.diagonal(S_chol)))
        S_chol_inv_diff = solve_triangular(S_chol, y_obs - mean_obs, lower=True)

        lower, upper = 0., np.inf
        lp = -(- log_det / 2
               - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
               + truncnorm.logpdf(scale_G,
                                  (lower - priors["scale_G_mean"]) / priors["scale_G_sd"],
                                  (upper - priors["scale_G_mean"]) / priors["scale_G_sd"],
                                  priors["scale_G_mean"],
                                  priors["scale_G_sd"])
               + truncnorm.logpdf(length_G,
                                  (lower - priors["length_G_mean"]) / priors["length_G_sd"],
                                  (upper - priors["length_G_mean"]) / priors["length_G_sd"],
                                  priors["length_G_mean"],
                                  priors["length_G_sd"])
               + truncnorm.logpdf(sigma_y,
                                  (lower - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                  (upper - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                  priors["sigma_y_mean"],
                                  priors["sigma_y_sd"]))

        grad = -np.array([
            ((alpha_alpha_T - S_inv) * S_dee_scale_G.T).sum() / 2
            - (scale_G - priors["scale_G_mean"]) / priors["scale_G_sd"]**2,

            ((alpha_alpha_T - S_inv) * S_dee_length_G.T).sum() / 2
            - (length_G - priors["length_G_mean"]) / priors["length_G_sd"]**2,

            ((alpha_alpha_T - S_inv) * S_dee_sigma_y.T).sum() / 2
            - (sigma_y - priors["sigma_y_mean"]) / priors["sigma_y_sd"]**2
        ])
        return(lp, grad)

    def optimize_lmp(self, current_values, *args):
        bounds = 3 * [(1e-12, None)]
        current_values = [
            c + np.random.uniform(0, 0.01) if c < 1e-10 else c for c in current_values
        ]
        self.params = minimize(fun=self.log_marginal_posterior,
                               x0=current_values, args=(*args, ),
                               method="L-BFGS-B", jac=True, bounds=bounds).x

    def analysis(self, H, x_obs, y_obs):
        priors = self.priors
        mean_obs = H @ self.mean
        cov_obs = H @ self.cov @ H.T

        self.optimize_lmp(self.params, self.M, self.V_base, H, self.J_scipy,
                          mean_obs, cov_obs, x_obs, y_obs)
        scale_G, length_G, sigma_y = self.params
        print(f"estimated parameters: {self.params}")

        # covariance on the underlying mesh
        # assembled through kronecker structure
        u1_dof_indices = self.V.sub(0).dofmap().dofs()
        x_u1 = self.V.tabulate_dof_coordinates()[u1_dof_indices]
        # covariance on the first component only
        K = sf.cov_square_exp(x_u1, scale_G, length_G)
        K = np.kron([[1, 0], [0, 0]], K)
        # permute for proper ordering, then scale (as in burgers example)
        G = self.dt * self.M @ self.P @ K @ self.P.T @ self.M.T

        values, vectors = eigsh(G, k=self.n_eigen)
        temp = spsolve(self.J_scipy, vectors)
        self.cov += temp @ np.diag(values) @ temp.T

        cov_obs = H @ self.cov @ H.T
        S = cov_obs + (self.nugget + sigma_y**2) * np.eye(len(x_obs))
        S_chol = cho_factor(S, lower=True)

        # update mean and covariance
        self.mean += self.cov @ H.T @ (cho_solve(S_chol, y_obs - mean_obs))
        self.cov -= self.cov @ H.T @ (cho_solve(S_chol, H @ self.cov))

        # set values
        self.u.vector()[:] = self.mean
        self.u_prev.assign(self.u)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # timestepping
    dt = 0.02
    spinup = 2000
    nt = 5000

    # mismatch from underdamped model
    # just want the prior mean
    data = KSSolve(dt, 0.95)
    prior = KSSolve(dt, 1.)
    post = StatKS(dt, 1.)

    # assimilate ~50 observations each timestep
    x = data.mesh.coordinates()[:-1]
    x_obs = x[::10]
    H = build_interpolation_matrix(x_obs, data.V)
    # only care about the wave (not the derivatives)
    H_out = build_interpolation_matrix(x, data.V)

    t = 0
    thin = 10
    nt_out = len([i for i in range(nt) if i % thin == 0])
    times = [(i + 1) * dt for i in range(nt) if i % thin == 0]

    # data outputs
    output = h5py.File(args.output, "w")
    output.create_dataset("x", data=x.flatten())
    output.create_dataset("x_obs", data=x_obs.flatten())

    kwargs = {"dtype": np.float64}
    prior_output = output.create_dataset("prior", (nt_out, 512), **kwargs)
    mean_output = output.create_dataset("mean", (nt_out, 512), **kwargs)
    cov_output = output.create_dataset("cov", (nt_out, 512, 512), **kwargs)
    y_output = output.create_dataset("y", (nt_out, 52), **kwargs)
    parameters_output = output.create_dataset("parameters", (nt_out, 3), **kwargs)
    t_output = output.create_dataset("t", data = times, **kwargs)

    # spin up to interesting dynamics
    # (skip transient behaviour)
    for i in range(spinup):
        data.timestep()

    # set IC's from spin up
    post.u_prev.assign(data.u)
    prior.u_prev.assign(data.u)

    i_save = 0
    for i in range(nt):
        t += dt
        start_time = time.time()
        data.timestep()
        y_obs = np.copy(data.get_u()[::20]
                        + np.random.normal(size=(52, ), scale=0.05))
        prior.timestep()
        post.timestep()
        post.analysis(H, x_obs, y_obs)

        if i % thin == 0:
            prior_output[i_save, :] = H_out @ prior.get_u()
            mean_output[i_save, :] = H_out @ post.mean[:]
            cov_output[i_save, :, :] = H_out @ post.cov[:, :] @ H_out.T
            y_output[i_save, :] = y_obs[:]
            parameters_output[i_save, :] = post.params[:]
            i_save += 1

        end_time = time.time()
        print(f"iteration {i + 1} / {nt} took {end_time - start_time}")

    output.close()
