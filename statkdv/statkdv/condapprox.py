import logging
import time

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy

import matplotlib.pyplot as plt

from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, cho_factor, cho_solve, lstsq, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, truncnorm


logger = logging.getLogger(__name__)


class KdVApprox():
    def __init__(self, settings, parameters):
        """ StatFEM for KdV equation, linearization method.

        Parameters
        ----------
        settings : dict
            Simulation settings for statFEM. Must contain `nx`, `nt`,
            `x_start`, `x_end`, `t_start`, `t_end`. Throws an error if any not
            included.
        parameters : dict
            Simulation coefficients for the statFEM. Must contain
            `alpha`, `beta`. Parameters `c`, `nu` optional (default to 0)

            `alpha`: steepening
            `beta`: dispersion
            `c`: wavespeed
            `nu`: linear damping
        """
        try:
            self.nx = settings["nx"]
            self.nt = settings["nt"]

            self.x_grid = np.linspace(settings["x_start"],
                                      settings["x_end"],
                                      settings["nx"],
                                      endpoint=False)

            self.t_grid = np.linspace(settings["t_start"],
                                      settings["t_end"],
                                      settings["nt"])
        except KeyError:
            print("settings dictionary missing values")
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

        # cache the base jacobian to save time
        self.jacobian_base = (
            sp.kron([[self.dt * self.c / 2, self.dt * self.beta / 2, 0], [0, 0, -1 / 2], [-1 / 2, 0, 0]],
                    self.mixed,
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

        self.mean_u_curr = None
        self.mean_v_curr = None
        self.mean_w_curr = None

        self.mean_u_prev = None
        self.mean_v_prev = None
        self.mean_w_prev = None

        self.cov_prev = np.zeros((3 * self.nx, 3 * self.nx))

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

        self.mean_u_prev = np.copy(np.broadcast_to(f_u_init(self.x_grid), (self.nx, ))).astype("float64")
        self.mean_v_prev = np.copy(np.broadcast_to(f_v_init(self.x_grid), (self.nx, ))).astype("float64")
        self.mean_w_prev = np.copy(np.broadcast_to(f_w_init(self.x_grid), (self.nx, ))).astype("float64")

        self.mean_u_curr = np.zeros((self.nx,))
        self.mean_v_curr = np.zeros((self.nx,))
        self.mean_w_curr = np.zeros((self.nx,))

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
        return K

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
        K *= self.dt**2 * self.dx**2
        self.G = np.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], K)
        self.estimate_G = False

    def set_sigma_y(self, sigma_y):
        self.sigma_y = sigma_y
        self.estimate_sigma_y = False

    def log_marginal_posterior(self, params, H, x_obs, y_obs, mean, cov_hat, J_curr, priors):
        if self.estimate_sigma_y:
            params = params.reshape(3, )
            scale_G, length_G, sigma_y = params
        else:
            params = params.reshape(2, )
            scale_G, length_G = params
            sigma_y = self.sigma_y

        K = self.cov_square_exp(self.x_grid, scale_G, length_G)
        K *= self.dt * self.dx**2
        G = sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], K)
        values, vectors = spla.eigsh(G, k=self.n_eigen)
        temp_mat = spla.spsolve(J_curr, vectors, permc_spec="MMD_AT_PLUS_A")
        G_obs = (H @ temp_mat) @ np.diag(values) @ (H @ temp_mat).T

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
        G_dee_length = sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dist / length_G**3 * K)
        values, vectors = spla.eigsh(G_dee_length, k=self.n_eigen)
        temp_mat = spla.spsolve(J_curr, vectors, permc_spec="MMD_AT_PLUS_A")
        S_dee_length = (H @ temp_mat) @ np.diag(values) @ (H @ temp_mat).T

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
        nugget = self.nugget
        bounds = 2 * [(1e-12, None)]
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

                if self.estimate_sigma_y == False:
                    params = np.append(params, self.sigma_y)

                self.nugget = nugget
                return(params)
            except:
                logger.warning("Optimization failed: trying optim again with jittered inits")
                inits = [i + np.random.uniform(0, 0.01) for i in inits]

                if i == 99:
                    return(current_values)

    def fem_lhs(self, u, v, w, mean_u_prev, mean_v_prev, mean_w_prev):
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
        u_prev, v_prev, w_prev = mean_u_prev, mean_v_prev, mean_w_prev
        mass, mixed = self.mass, self.mixed
        f1 = (mass @ (u - u_prev)
              + self.dt * self.nu * mass @ ((u + u_prev) / 2)
              + self.dt * self.alpha * mixed @ ((u + u_prev)**2 / 4) / 2
              + self.dt * self.beta * mixed @ ((w + w_prev) / 2)
              + self.dt * self.c * mixed @ ((u + u_prev) / 2))
        f2 = mass @ ((w + w_prev) / 2) - mixed @ ((v + v_prev) / 2)
        f3 = mass @ ((v + v_prev) / 2) - mixed @ ((u + u_prev) / 2)

        return np.concatenate((f1, f2, f3))

    def fem_lhs_update_jacobian_curr(self, u, mean_u_prev):
        """ Update the FEM LHS jacobian (current)

        Parameters
        ----------
        u : ndarray
        mean_u_prev : ndarray
        """
        jacobian = self.jacobian_base + (
            sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    self.mass + self.dt * self.alpha * self.mixed.multiply((u + mean_u_prev) / 2) / 2,
                    format="csr")
        )
        return(jacobian)

    def fem_lhs_update_jacobian_prev(self, u, mean_u_prev):
        """ Update the FEM LHS jacobian (previous)
        """
        jacobian = self.jacobian_base + (
            sp.kron(
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                -self.mass + self.dt * self.alpha * self.mixed.multiply((u + mean_u_prev) / 2) / 2,
                format="csr"
            )
        )
        return(jacobian)

    def newton(self, mean_u_prev, mean_v_prev, mean_w_prev, n_iter=50, tol=1e-8):
        """ Use Newton iterations to compute FEM timestep

        Parameters
        ----------
        mean_u_prev : ndarray
        mean_v_prev : ndarray
        mean_w_prev : ndarray
        """
        # initialize to previous values
        u, v, w = np.copy(mean_u_prev), np.copy(mean_v_prev), np.copy(mean_w_prev)

        for i in range(n_iter):
            f_curr = self.fem_lhs(u, v, w, mean_u_prev, mean_v_prev, mean_w_prev)
            j_curr = self.fem_lhs_update_jacobian_curr(u, mean_u_prev)
            shift = spla.spsolve(j_curr, -f_curr)
            u += shift[0:self.nx]
            w += shift[self.nx:(2 * self.nx)]
            v += shift[(2 * self.nx):]

            if np.sum(np.abs(shift)) < tol:
                logger.debug(f"newton used {i} iterations")
                break
            elif i == n_iter - 1:
                raise StopIteration

        return(u, v, w)

    def project_forward(self, y, u, H, H_project):
        """ Least-squares projection from u -> y

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

    def solve_fem(self, data=None, control=None, priors=None, output_file=None):
        """ Solve the problem using statFEM

        Parameters
        ----------
        data: dict, optional
            dict of data values. If `none` then a prior measure is returned. Otherwise
            must contain y, x_obs, and H (observation) operator.
        control: dict, optional
            values that tell statkdv how to behave.
        priors: dict, optional
            Priors on the mismatch parameters. Initializes to:
        output_file: str, optional
            `hdf5` file to export outputs to. This is constantly written to
            through the simulation.
        comm: MPI communicator, optional
        """
        if self.mean_u_prev is None or self.mean_v_prev is None or self.mean_w_prev is None:
            logger.error("initial values are None")
            raise ValueError

        if data is None:
            conditional = False
            logger.info("no conditioning: supply both y and H if desired")
        else:
            conditional = True

        priors_default = {"scale_G_mean": 1,
                          "scale_G_sd": 1,
                          "length_G_mean": 1,
                          "length_G_sd": 1,
                          "sigma_y_mean": 0,
                          "sigma_y_sd": 1}
        control_default = {"newton_iter": 50,
                           "newton_tol": 1e-8,
                           "scale_G_range": [0, 0.2],
                           "length_G_range": [0, 0.2],
                           "sigma_y_range": [0, 0.01],
                           "thin": 1,
                           "project_data_forward": False}

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

        # store in the interpolation matrix
        if data is not None:
            H = data["H"]

        logger.info(f"priors set to {priors}")
        logger.info(f"control options set to {control}")
        logger.info(f"saving every {control['thin']} iterations")

        # set thinning options
        # i.e. save every `thin`'th iterate
        thin = control["thin"]
        i_save = 1

        if self.nt % thin > 0:
            nt_save = self.nt // thin + 1
        else:
            nt_save = self.nt // thin

        if output_file is None:
            logger.warning("not saving output to disk")
        else:
            logger.debug("output file: " + output_file)
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

            for key in priors.keys():
                priors_output[key] = priors[key]

            for key in control.keys():
                control_output[key] = control[key]

            if data is None:
                model_output.attrs["type"] = "prior"
            else:
                model_output.attrs["type"] = "conditional"
                y_output = data_output.create_dataset("y", (nt_save, len(data["x_grid"])))
                data_output.create_dataset("x_grid", data=data["x_grid"])
                data_output.create_dataset("H", data=H.todense())
                y_output[0, :] = data["y"][0, :]
                mean_obs_output = model_output.create_dataset("mean_obs",
                                                              (nt_save, len(data["x_grid"])),
                                                              compression="gzip")
                cov_obs_output = model_output.create_dataset("covariance_obs",
                                                             (nt_save, len(data["x_grid"]), len(data["x_grid"])),
                                                             compression="gzip")

            model_output.create_dataset("x_grid", data=self.x_grid)
            model_output.create_dataset("t_grid", data=self.t_grid[[i for i in range(self.nt) if i % thin == 0]])

            model_output.attrs["method"] = "approx"

            model_output.attrs["alpha"] = self.alpha
            model_output.attrs["beta"] = self.beta
            model_output.attrs["c"] = self.c

            mean_output[0, :] = self.mean_u_prev
            cov_output[0, :, :] = np.zeros((self.nx, self.nx))

        if self.estimate_G:
            self.scale_G, self.length_G = [np.random.uniform(*control["scale_G_range"]),
                                           np.random.uniform(*control["length_G_range"])]
        if self.estimate_sigma_y:
            self.sigma_y = np.random.uniform(*control["sigma_y_range"])

        for i in range(1, self.nt):
            start_time = time.time()
            self.mean_u_curr, self.mean_v_curr, self.mean_w_curr = self.newton(self.mean_u_prev,
                                                                               self.mean_v_prev,
                                                                               self.mean_w_prev,
                                                                               control["newton_iter"],
                                                                               control["newton_tol"])
            mean = np.concatenate((self.mean_u_curr, self.mean_v_curr, self.mean_w_curr))

            J_curr = self.fem_lhs_update_jacobian_curr(self.mean_u_curr, self.mean_u_prev)
            J_prev = self.fem_lhs_update_jacobian_prev(self.mean_u_prev, self.mean_u_prev)
            M = J_prev @ self.cov_prev @ J_prev.T
            temp_mat = spla.spsolve(J_curr, M.T, permc_spec="MMD_AT_PLUS_A")
            cov_hat = spla.spsolve(J_curr, temp_mat.T, permc_spec="MMD_AT_PLUS_A")

            if self.estimate_G == False:
                temp_mat = spla.spsolve(J_curr, self.G.T, permc_spec="MMD_AT_PLUS_A")
                G_scaled = spla.spsolve(J_curr, temp_mat.T, permc_spec="MMD_AT_PLUS_A")
                cov = cov_hat + G_scaled

            if conditional:
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

                # jitter if getting into bad spot
                if self.sigma_y < 1e-10 and self.estimate_sigma_y:
                    self.sigma_y += np.random.uniform(*control["sigma_y_range"])

                self.scale_G, self.length_G, self.sigma_y = self.optimize_lml(
                    [self.scale_G, self.length_G, self.sigma_y],
                    H_project, x_project, y_project, mean, cov_hat, J_curr, priors
                )
                logger.info(f"parameters: {self.scale_G, self.length_G, self.sigma_y}")

                K = self.cov_square_exp(self.x_grid, self.scale_G, self.length_G)
                K *= self.dt * self.dx**2

                G = sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]], K)
                values, vectors = spla.eigsh(G, k=self.n_eigen)

                temp_mat = spla.spsolve(J_curr, vectors, permc_spec="MMD_AT_PLUS_A")
                G_hat = (temp_mat) @ np.diag(values) @ (temp_mat).T

                cov = cov_hat + G_hat
                S_chol = cho_factor(
                    H @ (cov_hat + G_hat) @ H.T + (self.nugget + self.sigma_y**2) * np.eye(len(data["x_grid"]))
                )

                # update mean & covariance
                mean += cov @ H.T @ (cho_solve(S_chol, data["y"][i, :] - H @ mean))
                cov -= cov @ H.T @ (cho_solve(S_chol, H @ cov))

            # set previous values
            self.cov_prev[:, :] = cov[:, :]
            self.mean_u_prev[:], self.mean_v_prev[:], self.mean_w_prev[:] = (
                mean[0:self.nx], mean[self.nx:(2 * self.nx)], mean[(2 * self.nx):]
            )

            if output_file is not None and i % thin == 0:
                mean_output[i_save, :] = mean[0:self.nx]
                cov_output[i_save, :, :] = cov[0:self.nx, 0:self.nx]

                # plt.plot(self.x_grid, mean[0:self.nx])
                # plt.fill_between(self.x_grid,
                #                  mean[0:self.nx] - 1.96 * np.sqrt(cov[0:self.nx, 0:self.nx].diagonal()),
                #                  mean[0:self.nx] + 1.96 * np.sqrt(cov[0:self.nx, 0:self.nx].diagonal()),
                #                  alpha=0.1)
                # plt.show()
                # plt.close()

                if conditional is True:
                    y_output[i_save, :] = data["y"][i, :]
                    mean_obs_output[i_save, :] = H @ mean
                    cov_obs_output[i_save, :, :] = H @ cov @ H.T
                    parameters_output[i_save - 1, :] = self.scale_G, self.length_G, self.sigma_y

                i_save += 1

            end_time = time.time()
            logger.info("iteration {} took {} s".format(i, end_time - start_time))
