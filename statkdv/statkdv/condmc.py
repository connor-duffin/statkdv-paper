import logging
import time

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy

from mpi4py import MPI
from scipy.linalg import cho_factor, cho_solve, cholesky, svd, solve, lstsq, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, truncnorm

logger = logging.getLogger(__name__)


class KdVEnsemble():
    def __init__(self, settings, parameters):
        """ StatFEM for KdV equation, ensemble method.

        Parameters
        ----------
        settings : dict
            Simulation settings for statFEM. Must contain `nx`, `nt`,
            `x_start`, `x_end`, `t_start`, `t_end`. Throws an error if any not
            included.
        parameters : dict
            Simulation coefficients for the statFEM. Must contain
            `alpha`, `beta`. Parameters `c`, `nu` optional (default to 0).

            `alpha`: steepening
            `beta`: dispersion
            `c`: wavespeed
            `nu`: linear damping
        """
        try:
            self.nx = settings["nx"]
            self.nt = settings["nt"]
            self.n_ensemble = settings["n_ensemble"]

            self.x_grid = np.linspace(settings["x_start"],
                                      settings["x_end"],
                                      settings["nx"],
                                      endpoint=False)
            self.t_grid = np.linspace(settings["t_start"],
                                      settings["t_end"],
                                      settings["nt"])
        except KeyError:
            logger.error("settings dictionary missing values")
            raise

        # take in parameters as appropriate
        # alpha and beta are required
        if "alpha" not in parameters or "beta" not in parameters:
            logger.error("parameters dictionary missing alpha and/or beta")
            raise KeyError
        else:
            self.alpha = parameters["alpha"]
            self.beta = parameters["beta"]

        if "c" not in parameters:
            logger.warning("no c provided: setting to zero")
            self.c = 0
        else:
            self.c = parameters["c"]

        if "nu" not in parameters:
            logger.warning("no nu provided: setting to zero")
            self.nu = 0
        else:
            self.nu = parameters["nu"]

        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dt = self.t_grid[1] - self.t_grid[0]

        self.mass = sp.diags(
            diagonals=[
                np.full(self.nx, self.dx / 2),
                np.full(self.nx - 1, self.dx / 2),
                self.dx / 2
            ],
            offsets=[0, 1, -(self.nx - 1)]
        )
        self.mixed = sp.diags(
            diagonals=[
                np.full(self.nx, -1),
                np.full(self.nx - 1, 1),
                1
            ],
            offsets=[0, 1, -(self.nx - 1)]
        )

        self.jacobian_base = (
            sp.kron([[self.dt * self.c, self.dt * self.beta, 0], [0, 0, -1], [-1, 0, 0]],
                    self.mixed / 2,
                    format="csr")
            + sp.kron([[self.dt * self.nu / 2, 0, 0], [0, 1 / 2, 0], [0, 0, 1 / 2]],
                      self.mass,
                      format="csr")
        )

        self.nugget = 1e-12
        self.scale_G = None
        self.length_G = None
        self.sigma_y = None
        self.estimate_G = True
        self.estimate_sigma_y = True
        self.n_eigen = 50

        self.u = None
        self.v = None
        self.w = None

        self.u_prev = None
        self.v_prev = None
        self.w_prev = None

        self.state = np.zeros((self.n_ensemble, 3 * self.nx))

    def set_initial_condition(self, u_expr, dep_var=None):
        """ Compute initial conditions with symbolic differentiation.

        Parameters
        ----------
        u_expr : string
            The sympy expression for the initial conditions.
        dep_var : sympy.symbol, optional
            Variable with which we should take derivatives with respect to in
            `u_expr`.
        """
        if type(u_expr) != str:
            logger.error("input expression must be a string")
            raise TypeError

        if dep_var is None:
            dep_var = sympy.symbols("x")
            logger.warning("assuming initial condition dependent variable is x")

        u_init = sympy.sympify(u_expr)
        v_init = sympy.diff(u_init, dep_var)
        w_init = sympy.diff(u_init, dep_var, 2)

        f_u_init = sympy.lambdify(dep_var, u_init, "numpy")
        f_v_init = sympy.lambdify(dep_var, v_init, "numpy")
        f_w_init = sympy.lambdify(dep_var, w_init, "numpy")

        self.u = np.zeros((self.n_ensemble, self.nx))
        self.v = np.zeros((self.n_ensemble, self.nx))
        self.w = np.zeros((self.n_ensemble, self.nx))

        self.u_prev = np.copy(np.broadcast_to(f_u_init(self.x_grid), (self.n_ensemble, self.nx))).astype("float64")
        self.v_prev = np.copy(np.broadcast_to(f_v_init(self.x_grid), (self.n_ensemble, self.nx))).astype("float64")
        self.w_prev = np.copy(np.broadcast_to(f_w_init(self.x_grid), (self.n_ensemble, self.nx))).astype("float64")

    def cov_square_exp(self, x_grid, scale, length):
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
        x = np.expand_dims(x_grid, 1)
        K = np.exp(-0.5 * pdist(x / length, metric="sqeuclidean"))
        K = squareform(K)
        np.fill_diagonal(K, 1)
        K = scale**2 * K
        return(K)

    def set_cov_square_exp(self, scale, length):
        """ Set up the covariance as a square-exponential

        Parameters
        ----------
        scale : float
            Scale parameter.
        length : float
            Length parameter.
        """
        K = self.cov_square_exp(self.x_grid, scale, length)
        self.scale_G = scale
        self.length_G = length

        G = self.dt * self.dx**2 * K + self.nugget * np.eye(self.nx)
        self.G_chol_fixed = cholesky(G, lower=True)
        self.estimate_G = False

    def set_sigma_y(self, sigma_y):
        self.sigma_y = sigma_y
        self.estimate_sigma_y = False

    def fem_lhs(self, u, v, w, u_prev, v_prev, w_prev, noise):
        """ Compute the FEM LHS

        Parameters
        ----------
        u : ndarray
        v : ndarray
        w : ndarray
        mean_u_prev : ndarray
        mean_v_prev : ndarray
        mean_w_prev : ndarray
        """
        mass, mixed = self.mass, self.mixed
        f1 = (mass @ (u - u_prev)
              + self.dt * self.nu * mass @ ((u + u_prev) / 2)
              + self.dt * self.alpha * mixed @ ((u + u_prev)**2 / 4) / 2
              + self.dt * self.beta * mixed @ ((w + w_prev) / 2)
              + self.dt * self.c * mixed @ ((u + u_prev) / 2)
              - noise)
        f2 = mass @ ((w + w_prev) / 2) - mixed @ ((v + v_prev) / 2)
        f3 = mass @ ((v + v_prev) / 2) - mixed @ ((u + u_prev) / 2)

        return np.concatenate((f1, f2, f3))

    def fem_lhs_update_jacobian(self, u, u_prev):
        """ Update the FEM LHS jacobian

        Parameters
        ----------
        u : ndarray
        u_prev : ndarray
        """
        jacobian = (self.jacobian_base +
                    sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                            self.mass + self.dt * self.alpha * self.mixed.multiply((u + u_prev) / 2) / 2,
                            format="csr"))
        return(jacobian)

    def fem_lhs_update_jacobian_prev(self, u, u_prev):
        """ Update the FEM LHS jacobian (previous)
        """
        jacobian = self.jacobian_base + (
            sp.kron(
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                -self.mass + self.dt * self.alpha * self.mixed.multiply((u + u_prev) / 2) / 2,
                format="csr"
            )
        )
        return(jacobian)

    def newton_timestep(self, u_prev, v_prev, w_prev, xi, n_iter=50, tol=1e-8):
        """ Compute the next timestep using Newton's method
        """
        u, v, w = np.copy(u_prev), np.copy(v_prev), np.copy(w_prev)
        noise = xi

        for i in range(n_iter):
            f_curr = self.fem_lhs(u, v, w, u_prev, v_prev, w_prev, noise)
            j_curr = self.fem_lhs_update_jacobian(u, u_prev)
            shift = spla.spsolve(j_curr, -f_curr)
            u += shift[0:self.nx]
            w += shift[self.nx:(2 * self.nx)]
            v += shift[(2 * self.nx):]

            if np.sum(np.abs(shift)) < tol:
                logger.debug("newton took {} iter".format(i))
                break
            elif i == n_iter - 1:
                raise StopIteration

        return(u, v, w)

    def project_forward(self, y, u, H, H_project):
        """ Least-squares projection from u -> y_obs.

        y : ndarray
            Data to project onto.
        u : ndarray
            Array to project from.
        H : ndarray
            Data observation operator.
        H_project : ndarray
            Projection observation operator.
        """
        X = (H @ u)[:, np.newaxis]**[0, 1]
        p, res, rnk, s = lstsq(X, y)
        y_project = p[0] + p[1] * H_project @ u
        return(y_project)

    def log_marginal_posterior(self, params, H, x_obs, y_obs, mean, cov_hat, J_curr, priors):
        if self.estimate_sigma_y:
            params = params.reshape(3,)
            scale_G, length_G, sigma_y = params
        else:
            params = params.reshape(2,)
            scale_G, length_G = params
            sigma_y = self.sigma_y

        K = self.cov_square_exp(self.x_grid, scale_G, length_G)
        K *= self.dt * self.dx**2
        G = np.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], K)
        temp_mat = spla.spsolve(J_curr, G.T, permc_spec="MMD_AT_PLUS_A")
        G_hat = spla.spsolve(J_curr, temp_mat.T, permc_spec="MMD_AT_PLUS_A")
        G_obs = H @ G_hat @ H.T

        # G = sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], K)
        # values, vectors = spla.eigsh(G, k=self.n_eigen)
        # temp_mat = H @ spla.spsolve(J_curr, vectors)
        # G_obs = temp_mat @ np.diag(values) @ temp_mat.T

        S = H @ cov_hat @ H.T + G_obs + (self.nugget + sigma_y**2) * np.eye(len(x_obs))
        S_chol = cholesky(S, lower=True)

        log_det = 2 * np.sum(np.log(np.diagonal(S_chol)))
        S_chol_inv_diff = solve_triangular(S_chol, y_obs - H @ mean, lower=True)

        x = np.expand_dims(self.x_grid, 1)
        dist = pdist(x, metric="sqeuclidean")
        dist = squareform(dist)

        S_dee_scale = 2 * G_obs / scale_G
        S_dee_sigma_y = 2 * sigma_y * np.eye(len(x_obs))

        # length parameter difficult due to length scale
        G_dee_length = np.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dist / length_G**3 * K)
        temp_mat = spla.spsolve(J_curr, G_dee_length.T, permc_spec="MMD_AT_PLUS_A")
        G_dee_length_hat = spla.spsolve(J_curr, temp_mat.T, permc_spec="MMD_AT_PLUS_A")
        S_dee_length = H @ G_dee_length_hat @ H.T

        # G_dee_length = sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dist / length_G**3 * K)
        # values, vectors = spla.eigsh(G_dee_length, k=self.n_eigen)
        # temp_mat = H @ spla.spsolve(J_curr, vectors)
        # S_dee_length = temp_mat @ np.diag(values) @ temp_mat.T

        S_chol = cho_factor(S, lower=True)
        S_inv = cho_solve(S_chol, np.eye(len(x_obs)))
        alpha = S_inv @ (y_obs - H @ mean)
        alpha_alpha_T = np.outer(alpha, alpha)

        lower, upper = 0., np.inf
        lp = -(
            - log_det / 2
            - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
            + truncnorm.logpdf(scale_G,
                               (lower - priors["scale_G_mean"]) / priors["scale_G_sd"],
                               (upper - priors["scale_G_mean"]) / priors["scale_G_sd"],
                               loc=priors["scale_G_mean"],
                               scale=priors["scale_G_sd"])
            + truncnorm.logpdf(length_G,
                               (lower - priors["length_G_mean"]) / priors["length_G_sd"],
                               (upper - priors["length_G_mean"]) / priors["length_G_sd"],
                               loc=priors["length_G_mean"],
                               scale=priors["length_G_sd"])
        )

        grad = [
            -((alpha_alpha_T - S_inv) * S_dee_scale.T).sum() / 2
            + (scale_G - priors["scale_G_mean"]) / priors["scale_G_sd"]**2,
            -((alpha_alpha_T - S_inv) * S_dee_length.T).sum() / 2
            + (length_G - priors["length_G_mean"]) / priors["length_G_sd"]**2
        ]
        if self.estimate_sigma_y:
            lp -= truncnorm.logpdf(sigma_y,
                                   (lower - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                   (upper - priors["sigma_y_mean"]) / priors["sigma_y_sd"],
                                   loc=priors["sigma_y_mean"],
                                   scale=priors["sigma_y_sd"])
            grad += [(-((alpha_alpha_T - S_inv) * S_dee_sigma_y.T).sum() / 2
                     + (sigma_y - priors["sigma_y_mean"]) / priors["sigma_y_sd"]**2)]
        return(lp.flatten(), np.array(grad))

    def optimize_lml(self, current_values, *args):
        bounds = [(1e-12, None), (1e-12, None)]
        inits = current_values[0:2]

        if self.estimate_sigma_y:
            bounds += [(1e-12, None)]
            if current_values[2] < 1e-10:
                current_values[2] += np.random.uniform(0, 0.01)
            inits += [current_values[2]]

        for i in range(100):
            try:
                params = minimize(fun=self.log_marginal_posterior,
                                  x0=inits, method="L-BFGS-B", args=(*args,),
                                  bounds=bounds, jac=True).x
                break
            except:
            # except LinAlgError or spla.ArpackError or spla.ArpackNoConvergence:
                logger.warning("Optimization failed: trying optim again with jittered inits")
            for init in inits:
                init += np.random.uniform(0, 0.01)
            if i == 99:
                raise StopIteration

        if self.estimate_sigma_y == False:
            params = np.append(params, self.sigma_y)

        return(params)

    def solve_ensemble(self, data=None, control=None, priors=None,
                       output_file=None, comm=None):
        """ Solve the problem using statFEM.

        Parameters
        ----------
        data: `dict`
            dict of data values. If `none` then a prior measure is returned. Otherwise
            must contain y, x_obs, and H (observation) operator.
        control: `dict` of values that tell statkdv how to behave.
        priors: `dict`
            Priors on the mismatch parameters.
        output_file: `str`
            File to put outputs into
        comm: MPI communicator
        """
        if self.u_prev is None or self.v_prev is None or self.w_prev is None:
            logger.error("initial values are None")
            raise ValueError

        # cleaner then self.{...}
        state = self.state
        u, v, w = self.u, self.v, self.w
        u_prev, v_prev, w_prev = self.u_prev, self.v_prev, self.w_prev
        xi = np.zeros_like(u)

        if data is None:
            conditional = False
            logger.info("not conditioning on data")
        else:
            conditional = True

        if comm is None:
            logger.warning("using default comm: MPI.COMM_WORLD")
            comm = MPI.COMM_WORLD

        rank = comm.Get_rank()
        size = comm.Get_size()

        priors_default = {
            "scale_G_mean": 1,
            "scale_G_sd": 1,
            "length_G_mean": 1,
            "length_G_sd": 1,
            "sigma_y_mean": 0,
            "sigma_y_sd": 1
        }

        control_default = {
            "newton_iter": 50,
            "newton_tol": 1e-8,
            "scale_G_range": [0, 0.2],
            "length_G_range": [0, 0.2],
            "sigma_y_range": [0, 0.01],
            "thin": 1,
            "noisefree": False,
            "save_ensemble": False,
            "project_data_forward": False
        }

        if priors is None:
            logger.warning("using default (weakly informative) priors")
        else:
            priors_default.update(priors)
        priors = priors_default

        if control is None:
            logger.warning("using default control")
        else:
            control_default.update(control)
        control = control_default

        if rank == 0:
            logger.info(f"priors set to {priors}")
            logger.info(f"control options set to {control}")
            logger.info(f"saving every {control['thin']} iterations")

        # set thinning options i.e. save every `thin`'th iterate
        thin = control["thin"]
        i_save = 1
        if self.nt % thin > 0:
            nt_save = self.nt // thin + 1
        else:
            nt_save = self.nt // thin

        if data is not None:
            # select data inside the representable spatial grid
            idx = data["x_grid"] <= self.x_grid[-1]
            data["y"] = data["y"][:, idx]
            data["x_grid"] = data["x_grid"][idx]
            H = data["H"][idx, :]

            if control["project_data_forward"]:
                if "x_project" in data or "H_project" in data:
                    logger.info("using projection to estimate covariance parameters")
                else:
                    logger.error("projection specified but not provided input "
                                 "(make sure that x_project or H_project is in the data dictionary)")
                    raise KeyError

        if rank == 0 and output_file is not None:
            logger.info("output file: " + output_file)
            output = h5py.File(output_file, "w")

            model_output = output.create_group("model")
            data_output = output.create_group("data")
            priors_output = output.create_group("priors")
            control_output = output.create_group("control")

            mean_output = model_output.create_dataset("mean",
                                                      (nt_save, self.nx),
                                                      compression="gzip")
            cov_output = model_output.create_dataset("covariance",
                                                     (nt_save, self.nx, self.nx),
                                                     compression="gzip")
            parameters_output = model_output.create_dataset("parameters", (nt_save - 1, 3))

            if control["save_ensemble"]:
                logger.warning("saving entire ensemble: this will use up a lot of disk space")
                ensemble_output = model_output.create_dataset("ensemble",
                                                              (nt_save, self.n_ensemble, 3 * self.nx))

            for key in priors.keys():
                priors_output[key] = priors[key]

            for key in control.keys():
                control_output[key] = control[key]

            if data is None:
                model_output.attrs["type"] = "marginal"
            else:
                model_output.attrs["type"] = "conditional"

                nx_obs = len(data["x_grid"])

                # save the mean, cov on the grid
                mean_obs_output = model_output.create_dataset("mean_obs",
                                                              (nt_save, nx_obs),
                                                              compression="gzip")
                cov_obs_output = model_output.create_dataset("covariance_obs",
                                                             (nt_save, nx_obs, nx_obs),
                                                             compression="gzip")

                y_output = data_output.create_dataset("y", (nt_save, nx_obs))
                data_output.create_dataset("x_grid", data=data["x_grid"])
                data_output.create_dataset("t_grid",
                                           data=self.t_grid[[i for i in range(self.nt) if i % thin == 0]])
                data_output.create_dataset("H", data=H.todense())

                y_output[0, :] = data["y"][0, :]

            model_output.create_dataset("x_grid", data=self.x_grid)
            model_output.create_dataset("t_grid",
                                        data=self.t_grid[[i for i in range(self.nt) if i % thin == 0]])

            model_output.attrs["method"] = "mc"

            model_output.attrs["alpha"] = self.alpha
            model_output.attrs["beta"] = self.beta
            model_output.attrs["c"] = self.c

            mean_output[0, :] = self.u_prev[0, :]
            cov_output[0, :, :] = np.zeros((self.nx, self.nx))
        elif output_file is None:
            logger.warning("not saving output to disk")
            output = None
        else:
            output = None
            H = None

        # split ensemble members, by index, across nodes
        if rank == 0:
            sims = [i for i in range(self.n_ensemble)]
            sims = [sims[i::size] for i in range(size)]
        else:
            sims = None
        sims = comm.scatter(sims)

        if data is not None:
            H = comm.bcast(H)

        # set inital values
        if self.estimate_G:
            self.scale_G, self.length_G = (np.random.uniform(*control["scale_G_range"]),
                                           np.random.uniform(*control["length_G_range"]))
        if self.estimate_sigma_y:
            self.sigma_y = np.random.uniform(*control["sigma_y_range"])

        # set to zero outside of the current simulation
        mask = np.ones(self.n_ensemble, np.bool)
        mask[sims] = 0
        u_prev[mask, :] = 0.
        v_prev[mask, :] = 0.
        w_prev[mask, :] = 0.

        for i in range(1, self.nt):
            start_time = time.time()
            # forecast step
            for sim in sims:
                if self.estimate_G:
                    xi[sim, :] = 0.
                else:
                    xi[sim, :] = self.G_chol_fixed @ np.random.normal(size=(self.nx,))

                u[sim, :], v[sim, :], w[sim, :] = self.newton_timestep(
                    u_prev[sim, :],
                    v_prev[sim, :],
                    w_prev[sim, :],
                    xi[sim, :],
                    control["newton_iter"],
                    control["newton_tol"]
                )
                state[sim, :] = np.concatenate((u[sim, :], v[sim, :], w[sim, :]))

            # store the current state
            state_all = np.array(comm.gather(state, root=0))
            u_prev_all = np.array(comm.gather(u_prev, root=0))
            if rank == 0:
                # HACK: sum over zeros in one dim
                state_all = np.sum(state_all, axis=0)
                u_prev_all = np.sum(u_prev_all, axis=0)

            if conditional:
                if rank == 0:
                    mean = np.sum(state_all, axis=0) / self.n_ensemble
                    cov = np.cov(state_all.T)
                    mean_u_prev = np.mean(u_prev_all, axis=0)
                    mean_u_prev_local = np.mean(u_prev[sims, :], axis=0)
                    J_curr = self.fem_lhs_update_jacobian(mean[0:self.nx],
                                                          mean_u_prev)

                    if control["project_data_forward"]:
                        y_project = self.project_forward(data["y"][i, :],
                                                         mean,
                                                         data["H"],
                                                         data["H_project"])
                        x_project = data["x_project"]
                        H_project = data["H_project"]
                    else:
                        y_project = data["y"][i, :]
                        x_project = data["x_grid"]
                        H_project = H

                    # self.scale_G, self.length_G = 0.1, 1.
                    self.scale_G, self.length_G, self.sigma_y = self.optimize_lml(
                        [self.scale_G, self.length_G, self.sigma_y],
                        H_project, x_project, y_project, mean, cov, J_curr, priors
                    )
                    logger.info(f"{self.scale_G, self.length_G, self.sigma_y}")

                    G = self.cov_square_exp(self.x_grid, self.scale_G, self.length_G)
                    G *= self.dt * self.dx**2  # scale down to smooth physics?
                    G[np.diag_indices_from(G)] += self.nugget
                    G_chol = cholesky(G, lower=True)
                else:
                    G_chol = None

                # full propagation, with xi
                G_chol = comm.bcast(G_chol)
                for sim in sims:
                    xi[sim, :] = G_chol @ np.random.normal(size=(self.nx,))

                    u[sim, :], v[sim, :], w[sim, :] = self.newton_timestep(
                        u_prev[sim, :],
                        v_prev[sim, :],
                        w_prev[sim, :],
                        xi[sim, :],
                        control["newton_iter"],
                        control["newton_tol"]
                    )
                    state[sim, :] = np.concatenate((u[sim, :], v[sim, :], w[sim, :]))

                state_all = np.array(comm.gather(state, root=0))

                if rank == 0:
                    # recipe from Mandel report
                    # HACK: gather makes 3d array --- sum over zeros in one dim
                    state_all = np.sum(state_all, axis=0)

                    U = state_all.T.copy()
                    U_mean = U @ (np.ones((self.n_ensemble, ))) / self.n_ensemble
                    U_obs = H @ U
                    A = U - np.outer(U_mean, np.ones((self.n_ensemble, )))
                    HA = H @ A
                    noise = self.sigma_y * np.random.normal(size=(self.n_ensemble, len(data["x_grid"])))
                    D = (data["y"][i, :] + noise).T
                    Y = D - U_obs
                    C_obs = HA @ HA.T / (self.n_ensemble - 1)
                    P = C_obs + (self.sigma_y**2 + self.nugget) * np.eye(len(data["x_grid"]))
                    M = solve(P, Y, sym_pos=True)
                    Z = HA.T @ M
                    U += A @ Z / (self.n_ensemble - 1)
                    U_updated = U.T  # for spreading around the place

                    # mean = np.sum(state_all, axis=0) / self.n_ensemble
                    # cov = np.cov(state_all.T)  # transpose for correct shape
                    # if (i - 1) % 10 == 0:
                    #     plt.plot(self.x_grid, U_mean[0:self.nx])
                    #     plt.title(f"Before conditioning, t = {self.t_grid[i]}")
                    #     plt.xlabel("x")
                    #     plt.ylabel("u")
                    #     plt.show()
                    #     plt.close()

                    # svd method
                    # u_svd, s, v_svd = svd(A)
                    # n_svd = len(s[s > 1e-10])
                    # A = u_svd[:, 0:n_svd] @ np.diag(s[0:n_svd]) @ v_svd[0:n_svd, :]
                    # z = U_obs @ (np.ones((self.n_ensemble, )))
                    # HA = U_obs - np.outer(z, np.ones((self.n_ensemble, ))) / self.n_ensemble
                    # chol_P = cho_factor(P, lower=True)
                    # M = cho_solve(chol_P, Y)

                    # ensrf
                    # TT = np.eye(self.n_ensemble) - HA.T @ solve(P, HA) / (self.n_ensemble - 1)
                    # T = cholesky(TT, lower=True)
                    # m_single = solve(P, data["y"][i, :] - H @ U_mean, sym_pos=True)
                    # V = A @ T / (np.sqrt(self.n_ensemble - 1))
                    # U_mean += A @ HA.T @ m_single / (self.n_ensemble - 1)
                    # U = np.expand_dims(U_mean, 1) + V

                    # old updating scheme
                    # S = H @ cov @ H.T + (self.sigma_y**2) * np.eye(len(data["x_grid"]))
                    # # U_updated = U + cov @ H.T @ (H @ cov @ H.T + R)^(-1) (D - H @ U)
                    # noise = self.sigma_y * np.random.normal(size=(self.n_ensemble, len(data["x_grid"])))
                    # D = (data["y"][i, :] + noise).T
                    # Y = D - H @ U
                    # U_updated = (U + cov @ H.T @ solve(S, Y)).T
                else:
                    U_updated = np.zeros((self.n_ensemble, 3 * self.nx))

                # broadcast to all nodes
                state_all = np.array(comm.bcast(U_updated, root=0))
                u[sims, :] = state_all[sims, 0:self.nx]
                v[sims, :] = state_all[sims, self.nx:(2 * self.nx)]
                w[sims, :] = state_all[sims, (2 * self.nx):]

            if rank == 0 and output_file is not None and i % thin == 0:
                mean = np.sum(state_all, axis=0) / self.n_ensemble
                cov = np.cov(state_all.T)

                mean_output[i_save, :] = mean[0:self.nx]
                cov_output[i_save, :, :] = cov[0:self.nx, 0:self.nx]

                if control["save_ensemble"]:
                    ensemble_output[i_save, :, :] = state_all[:]

                if conditional:
                    y_output[i_save, :] = data["y"][i, :]
                    mean_obs_output[i_save, :] = H @ mean
                    cov_obs_output[i_save, :, :] = H @ cov @ H.T
                    parameters_output[i_save - 1, :] = self.scale_G, self.length_G, self.sigma_y

                i_save += 1

            for sim in sims:
                u_prev[sim, :] = u[sim, :]
                v_prev[sim, :] = v[sim, :]
                w_prev[sim, :] = w[sim, :]

            end_time = time.time()
            if rank == 0:
                logger.info(f"iteration {i} took {end_time - start_time:.4f} s on root node")

        if rank == 0 and output_file is not None:
            output.close()
