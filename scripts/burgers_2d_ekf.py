import argparse
import logging

from dolfin import *
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, spsolve
from scipy.spatial.distance import pdist, squareform
from scipy.stats import truncnorm


np.random.seed(27)
parameters["reorder_dofs_serial"] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_level(50)


def build_observation_operator(x_obs, V):
    """ Build interpolation matrix from `x_obs` on function space V. This
    assumes that the observations are from the first sub-function of V.

    From the fenics forums.
    """
    nx, dim = x_obs.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    bbt = mesh.bounding_box_tree()

    # dofs from first subspace
    dolfin_element = V.sub(0).dolfin_element()
    dofmap = V.sub(0).dofmap()
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
        vals[jj] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    M = csr_matrix((vals, ij), shape=(nx, V.dim()))
    return(M)


def cov_square_exp(x_grid, scale, length):
    """ Compute the square exponential covariance matrix

    Parameters
    ----------
    x_grid : ndarray
        Grid on which to evaluate the covariance.
    scale : float
        Scale parameter.
    length : float
        Length parameter.
    """
    if len(x_grid.shape) == 1:
        x = np.expand_dims(x_grid, 1)
    else:
        x = x_grid

    K = np.exp(-0.5 * pdist(x / length, metric="sqeuclidean"))
    K = squareform(K)
    np.fill_diagonal(K, 1)
    K = scale**2 * K
    return(K)


def dolfin_to_csr(A):
    """ Convert assembled matrix to scipy CSR.
    """
    mat = as_backend_type(A).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return(csr)


class PeriodicBoundary(SubDomain):
    def __init__(self, L):
        SubDomain.__init__(self)
        self.L = L

    # Domain is left and bottom boundary, and not the two corners (0, 1) and (1, 0)
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], self.L)) or
                        (near(x[0], self.L) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.L) and near(x[1], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1] - self.L
        elif near(x[0], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1]
        else:   # near(x[1], self.L)
            y[0] = x[0]
            y[1] = x[1] - self.L


class BurgersSolve():
    def __init__(self, dt=0.01, nx=64, L=2., nu=0.01):
        self.mesh = mesh = RectangleMesh(Point(0., 0.), Point(L, L), nx, nx)
        element = FiniteElement("CG", triangle, 1)
        self.V = V = FunctionSpace(mesh,
                                   MixedElement([element, element]),
                                   constrained_domain=PeriodicBoundary(L))

        # functions
        self.w = Function(V)
        self.w_prev = Function(V)
        (u, v) = split(self.w)
        (r, s) = split(TestFunction(V))
        (u_prev, v_prev) = split(self.w_prev)

        # set initial values on the mesh
        ic = Expression(("sin(pi * (x[0] + x[1]))",
                         "sin(pi * (x[0] + x[1]))"), degree=8)
        self.w.interpolate(ic)
        self.w_prev.interpolate(ic)

        dt = Constant(dt)
        nu = Constant(nu)

        w_half = (self.w + self.w_prev) / 2
        u_half = (u + u_prev) / 2
        v_half = (v + v_prev) / 2
        self.F = ((u - u_prev) * r * dx
                  + dt * inner(w_half, grad(u_half)) * r * dx
                  + dt * nu * inner(grad(u_half), grad(r)) * dx
                  + (v - v_prev) * s * dx
                  + dt * inner(w_half, grad(v_half)) * s * dx
                  + dt * nu * inner(grad(v_half), grad(s)) * dx)
        self.J = derivative(self.F, self.w)

    def timestep(self):
        solve(self.F == 0, self.w, J=self.J)
        self.w_prev.assign(self.w)


class BurgersExtended():
    def __init__(self, dt=0.01, nx=64, L=2., nu=0.01):
        self.nx = nx
        self.L = L

        self.mesh = RectangleMesh(Point(0., 0.), Point(L, L), nx, nx)
        element = FiniteElement("CG", triangle, 1)
        self.V = FunctionSpace(self.mesh,
                               MixedElement([element, element]),
                               constrained_domain=PeriodicBoundary(L))

        psi = TrialFunction(self.V)
        phi = TestFunction(self.V)
        self.M = dolfin_to_csr(assemble(inner(psi, phi) * dx))

        # functions
        self.w = Function(self.V)
        self.w_prev = Function(self.V)
        (u, v) = split(self.w)
        (r, s) = split(TestFunction(self.V))
        (u_prev, v_prev) = split(self.w_prev)

        # set initial values on the mesh
        ic = Expression(("sin(pi * (x[0] + x[1]))",
                         "sin(pi * (x[0] + x[1]))"), degree=8)
        self.w.interpolate(ic)
        self.w_prev.interpolate(ic)

        self.dt = dt
        nu = Constant(nu)

        w_half = (self.w + self.w_prev) / 2
        u_half = (u + u_prev) / 2
        v_half = (v + v_prev) / 2
        self.F = ((u - u_prev) * r * dx
                  + dt * inner(w_half, grad(u_half)) * r * dx
                  + dt * nu * inner(grad(u_half), grad(r)) * dx
                  + (v - v_prev) * s * dx
                  + dt * inner(w_half, grad(v_half)) * s * dx
                  + dt * nu * inner(grad(v_half), grad(s)) * dx)
        self.J = derivative(self.F, self.w)
        self.J_prev = derivative(self.F, self.w_prev)

        # statFEM conditioning stuff
        self.n_dof = len(self.w.vector()[:])
        self.mean = np.zeros((self.n_dof, ))
        self.mean_prev = np.zeros((self.n_dof, ))
        self.cov = np.zeros((self.n_dof, self.n_dof))
        self.u_mean = np.zeros((len(self.V.sub(0).dofmap().dofs()), ))

        self.priors = {"scale_G_mean": 1, "scale_G_sd": 1,
                       "sigma_y_mean": 0, "sigma_y_sd": 1}

    def set_covariance_parameters(self, params=None):
        if params is not None:
            self.scale_G, self.sigma_y = params
            self.estimate_params = False
        else:
            self.scale_G, self.sigma_y = np.random.uniform(0., 0.01, size=(2, ))
            self.estimate_params = True

    def set_lex_ordering(self):
        """
        Find the lexicographical ordering on the mesh for various purposes.
        This gets the permutation matrix to map from lexicographical ordering
        to fenics dof ordering
        """
        # get dofs and coordinates
        all_dof_coords = self.V.tabulate_dof_coordinates().reshape((-1, 2))
        self.u_dof_indices = np.array(self.V.sub(0).dofmap().dofs())
        v_dof_indices = np.array(self.V.sub(1).dofmap().dofs())

        self.u_dof_coords = all_dof_coords[self.u_dof_indices, :]
        self.v_dof_coords = all_dof_coords[v_dof_indices, :]

        u_sorted_indices = np.lexsort((self.u_dof_coords[:, 0],
                                       self.u_dof_coords[:, 1]))
        self.u_dof_coords_sorted = self.u_dof_coords[u_sorted_indices, :]

        v_sorted_indices = np.lexsort((self.v_dof_coords[:, 0],
                                       self.v_dof_coords[:, 1]))

        total_lex_indices = np.concatenate((self.u_dof_indices[u_sorted_indices],
                                            v_dof_indices[v_sorted_indices]))

        assert self.n_dof == all_dof_coords.shape[0]
        n_dof = self.n_dof
        P = lil_matrix((n_dof, n_dof))
        for i, j in enumerate(total_lex_indices):
            P[i, j] = 1.

        # P: lex. ordering -> fenics ordering
        self.P = P.T

    def set_base_covariance(self, length):
        K = cov_square_exp(self.u_dof_coords_sorted, 1., length)
        K[np.diag_indices_from(K)] += 1e-8
        logger.info("K built, starting EVD")
        n = K.shape[0]

        # matrix-vector product
        def mv(v):
            return(np.concatenate((K @ v[0:n], np.zeros((n, )))))

        A = LinearOperator((2 * n, 2 * n), mv)
        self.G_base_vals, self.G_base_vecs = eigsh(A, k=128)

        # now set the correct ordering and scale by the mass matrix
        self.G_base_vecs = self.M @ self.P @ self.G_base_vecs
        logger.info(f"EVD done, eigenvalue range = {self.G_base_vals[-1], self.G_base_vals[0]}")

    def log_marginal_posterior(self, params, y, mean_obs, cov_obs, G_hat_obs):
        scale_G, sigma_y = params
        priors = self.priors

        S = cov_obs + self.dt * scale_G**2 * G_hat_obs + (1e-8 + sigma_y**2) * np.eye(len(y))
        S_chol = cholesky(S, lower=True)
        S_inv = cho_solve((S_chol, True), np.eye(len(y)))
        alpha = S_inv @ (y - mean_obs)
        alpha_alpha_T = np.outer(alpha, alpha)

        log_det = 2 * np.sum(np.log(S_chol.diagonal()))
        S_chol_inv_diff = solve_triangular(S_chol, y - mean_obs, lower=True)

        S_dee_scale_G = 2 * self.dt * scale_G * G_hat_obs
        S_dee_sigma_y = 2 * sigma_y * np.eye(len(y))

        lower, upper = 0., np.inf
        lp = -(- log_det / 2
               - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
               + truncnorm.logpdf(scale_G,
                                  (lower - priors["scale_G_mean"]) / priors["scale_G_sd"],
                                  (upper - priors["scale_G_mean"]) / priors["scale_G_sd"],
                                  priors["scale_G_mean"],
                                  priors["scale_G_sd"])
               + truncnorm.logpdf(sigma_y,
                                  (lower - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                  (upper - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                  priors["sigma_y_mean"],
                                  priors["sigma_y_sd"]))

        grad = -np.array([
            ((alpha_alpha_T - S_inv) * S_dee_scale_G.T).sum() / 2
            - (scale_G - priors["scale_G_mean"]) / priors["scale_G_sd"]**2,
            ((alpha_alpha_T - S_inv) * S_dee_sigma_y.T).sum() / 2
            - (sigma_y - priors["sigma_y_mean"]) / priors["sigma_y_sd"]**2
        ])

        return(lp, grad)

    def optimize_lmp(self, current_values, *args):
        bounds = 2 * [(1e-12, None)]
        inits = [
            c + np.random.uniform(0, 0.01) if c < 1e-10 else c for c in current_values
        ]
        n_optim_runs = 100
        for i in range(n_optim_runs):
            try:
                out = minimize(fun=self.log_marginal_posterior, x0=inits,
                               args=(*args, ), method="L-BFGS-B", jac=True, bounds=bounds).x
            except np.linalg.LinAlgError:
                logger.info("cholesky failed -- restarting with jittered inits")
                inits = [i + np.random.uniform(0, 0.01) for i in inits]
        return(out)

    def timestep(self, y, H):
        solve(self.F == 0, self.w, J=self.J)

        self.mean[:] = np.copy(self.w.vector()[:])
        mean_obs = H @ self.mean
        J = dolfin_to_csr(assemble(self.J))
        J_prev = dolfin_to_csr(assemble(self.J_prev))

        # prediction covariance
        self.cov_vals, self.cov_vecs = eigsh(self.cov, k=128)
        logger.info(f"cov EVD done, eigenvalue range = {self.cov_vals[0], self.cov_vals[-1]}")
        temp = spsolve(J, J_prev @ self.cov_vecs)
        temp_obs = H @ temp
        self.cov = temp @ np.diag(self.cov_vals) @ temp.T
        self.cov_obs = temp_obs @ np.diag(self.cov_vals) @ temp_obs.T

        # build G_hat: unscaled EKF version
        temp = spsolve(J, self.G_base_vecs)
        G_hat = temp @ np.diag(self.G_base_vals) @ temp.T
        G_hat_obs = (H @ temp) @ np.diag(self.G_base_vals) @ (H @ temp).T

        # estimate hyperparameters
        if self.estimate_params:
            self.scale_G, self.sigma_y = self.optimize_lmp((self.scale_G, self.sigma_y),
                                                           y, mean_obs, self.cov_obs, G_hat_obs)
        logger.info(f"estimated parameters: {self.scale_G, self.sigma_y}")

        # after parameter estimation: add on G
        self.cov += self.dt * self.scale_G**2 * G_hat
        self.cov_obs += self.dt * self.scale_G**2 * G_hat_obs

        # observation covariance
        S = self.cov_obs + (self.sigma_y**2 + 1e-8) * np.eye(len(y))
        S_chol = cho_factor(S, lower=True)

        # kalman updates: for high-dimensions this is the bottleneck
        HC = H @ self.cov
        self.mean += HC.T @ (cho_solve(S_chol, y_obs - mean_obs))
        self.cov -= HC.T @ (cho_solve(S_chol, HC))

        self.u_mean = self.mean[self.u_dof_indices]
        self.w.vector()[:] = np.copy(self.mean)
        self.w_prev.assign(self.w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    t = 0
    dt = 0.01
    nt = 500
    thin = 1
    nt_out = len([i for i in range(nt) if i % thin == 0])
    times = [(i + 1) * dt for i in range(nt) if i % thin == 0]

    # mismatched reynolds (100 =/= 150)
    burgers_data = BurgersSolve(dt=dt, L=2., nu=1 / 100)
    burgers_prior = BurgersSolve(dt=dt, L=2., nu=1 / 150)
    burgers = BurgersExtended(dt=dt, L=2., nu=1 / 150)
    burgers.set_lex_ordering()
    burgers.set_base_covariance(length=1.)
    burgers.set_covariance_parameters(None)

    u_dof_indices = np.array(burgers_data.V.sub(0).dofmap().dofs())
    obs_indices = u_dof_indices[::40]
    x_obs = burgers_data.V.tabulate_dof_coordinates()[obs_indices]
    x_u = burgers_data.V.tabulate_dof_coordinates()[u_dof_indices]
    H_obs = build_observation_operator(x_obs, burgers.V)
    logger.info(f"assimilating {len(obs_indices)} data points per iteration")

    logger.info(f"results being saved to {args.output}")
    output = h5py.File(args.output, "w")
    output.create_dataset("x", data=x_u)
    output.create_dataset("t", data=times)
    output.create_dataset("x_obs", data=x_obs)

    n_dof = len(u_dof_indices)
    kwargs = {"dtype": np.float64}
    mean_output = output.create_dataset("post_mean", (nt_out, n_dof), **kwargs)
    var_output = output.create_dataset("post_var", (nt_out, n_dof), **kwargs)
    prior_output = output.create_dataset("prior_mean", (nt_out, n_dof), **kwargs)
    y_output = output.create_dataset("y", (nt_out, len(obs_indices)), **kwargs)
    parameters_output = output.create_dataset("parameters", (nt_out, 2), **kwargs)

    i_save = 0
    for i in range(nt):
        t += dt

        logger.info(f"iteration {i + 1} / {nt}")
        burgers_data.timestep()
        burgers_prior.timestep()

        y_obs = burgers_data.w.vector()[:][obs_indices]
        y_obs += np.random.normal(scale=0.01, size=y_obs.shape)
        burgers.timestep(y_obs, H_obs)

        if i % thin == 0:
            # store outputs
            mean_output[i_save, :] = burgers.u_mean
            prior_output[i_save, :] = burgers_prior.w.vector()[:][u_dof_indices]
            var_output[i_save, :] = burgers.cov[burgers.u_dof_indices,
                                                burgers.u_dof_indices]
            y_output[i_save, :] = y_obs
            parameters_output[i_save, :] = burgers.scale_G, burgers.sigma_y
            i_save += 1

    output.close()
