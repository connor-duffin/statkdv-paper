import logging

from fenics import *

import numpy as np

from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm


logger = logging.getLogger(__name__)
NUGGET = 1e-10


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


def log_marginal_likelihood(params, mean_obs, cov_obs, x_obs, y_obs, priors, noisefree):
    """ Compute the log marginal likelihood.
    """
    if noisefree:
        params = params.reshape(2,)
        scale_delta, length_delta = params
        sigma_y = 0.
    else:
        params = params.reshape(3,)
        scale_delta, length_delta, sigma_y = params

    K_delta = cov_square_exp(x_obs, scale_delta, length_delta)

    S = cov_obs + K_delta + (NUGGET + sigma_y**2) * np.eye(len(x_obs))
    S_chol = cholesky(S, lower=True)

    log_det = 2 * np.sum(np.log(np.diagonal(S_chol)))
    S_chol_inv_diff = solve_triangular(S_chol, y_obs - mean_obs, lower=True)

    output = -(
        - log_det / 2
        - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
        + norm.logpdf(scale_delta,
                      priors["scale_delta_mean"],
                      priors["scale_delta_sd"])
        + norm.logpdf(length_delta,
                      priors["length_delta_mean"],
                      priors["length_delta_sd"])
    )
    if not noisefree:
        output -= norm.logpdf(sigma_y,
                              priors["sigma_y_mean"],
                              priors["sigma_y_sd"])
    return(output.flatten())


def log_marginal_likelihood_derivative(params, mean_obs, cov_obs, x_obs, y_obs, priors, noisefree):
    """ Compute the log marginal likelihood derivative.
    """
    if noisefree:
        params = params.reshape(2,)
        scale_delta, length_delta = params
        sigma_y = 0.
    else:
        params = params.reshape(3,)
        scale_delta, length_delta, sigma_y = params

    K_delta = cov_square_exp(x_obs, scale_delta, length_delta)
    S = cov_obs + K_delta + (NUGGET + sigma_y**2) * np.eye(len(x_obs))

    if len(x_obs.shape) == 1:
        x = np.expand_dims(x_obs, 1)
    else:
        x = x_obs

    dist = pdist(x, metric="sqeuclidean")
    dist = squareform(dist)

    S_dee_scale_delta = 2 * K_delta / scale_delta
    S_dee_length_delta = dist / length_delta**3 * K_delta

    if not noisefree:
        S_dee_sigma_y = 2 * sigma_y * np.eye(len(x_obs))

    S_chol = cho_factor(S, lower=True)
    S_inv = cho_solve(S_chol, np.eye(len(x_obs)))
    alpha = S_inv @ (y_obs - mean_obs)
    alpha_alpha_T = np.outer(alpha, alpha)

    output = -np.array([
        ((alpha_alpha_T - S_inv) * S_dee_scale_delta.T).sum() / 2
        - (scale_delta - priors["scale_delta_mean"]) / priors["scale_delta_sd"]**2,
        ((alpha_alpha_T - S_inv) * S_dee_length_delta.T).sum() / 2
        - (length_delta - priors["length_delta_mean"]) / priors["length_delta_sd"]**2,
    ])

    if not noisefree:
        output = np.concatenate((
            output,
            -np.array([
                (((alpha_alpha_T - S_inv) * S_dee_sigma_y.T).sum() / 2
                 -(sigma_y - priors["sigma_y_mean"]) / priors["sigma_y_sd"]**2)
            ])
        ))

    return(output)


def optimize_lml(*args, noisefree):
    """ Wrapper for scipy.minimize call.

    Runs optimization until a result is returned. If no convergence after 10
    tries then statkdv throws an error and exits.
    """
    global NUGGET
    max_optimizations = 10
    for i in range(max_optimizations):
        x0 = [np.random.uniform(0, 0.2),
              np.random.uniform(0, 0.2)]
        bounds = [(1e-12, None)] * 2
        if not noisefree:
            bounds += [(1e-12, None)]
            x0 += [np.random.uniform(0, 0.1)]
        try:
            params = minimize(fun=log_marginal_likelihood,
                              x0=x0,
                              method="L-BFGS-B",
                              args=(*args, noisefree),
                              bounds=bounds,
                              jac=log_marginal_likelihood_derivative).x
            NUGGET = 1e-10
            break
        except np.linalg.LinAlgError:
            NUGGET *= 10
            print("cholesky failed in LML optimization --- trying again")

    return(params)


def build_covariance_matrix(V, sigma, ell, kernel="square-exp"):
    """ Build the covariance matrix on function space V.

    From the fenics forums.
    """
    dim = V.mesh().coordinates().shape[1]

    u = TrialFunction(V)
    v = TestFunction(V)

    # build mass matrix
    m_form = inner(u, v) * dx
    M = dolfin_to_csr(assemble(m_form))

    # TODO: expand to different kernels
    if kernel == "square-exp":
        if dim == 1:
            kernel = Expression("pow(sigma, 2) * exp(-(pow(x[0] - x0, 2)) / (2 * pow(ell, 2)))",
                                sigma=sigma, ell=ell, x0=0., degree=8)
        elif dim == 2:
            kernel = Expression("pow(sigma, 2) * exp(-(pow(x[0] - x0, 2) + pow(x[1] - x1, 2)) / (2 * pow(ell, 2)))",
                                sigma=sigma, ell=ell, x0=0., x1=0., degree=8)
        else:
            print("cannot do high dimensions yet")
            raise ValueError
        x_ks = V.tabulate_dof_coordinates()
        C = np.zeros(M.shape)
        row_form = inner(kernel, v) * dx

        for row, x_k in enumerate(x_ks):
            if dim == 1:
                kernel.x0 = x_k[0]
            elif dim == 2:
                kernel.x0 = x_k[0]
                kernel.x1 = x_k[1]

            row_values = assemble(row_form).get_local()
            C[row, :] = row_values

        return(M @ C)
    elif kernel == "delta":
        return(sigma**2 * M.todense())
    else:
        logger.warning("kernel type not recognised - returning None")
        return(None)


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
        vals[jj] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    M = csr_matrix((vals, ij), shape=(nx, V.dim()))
    return(M)


def dolfin_to_csr(A):
    """ Convert assembled matrix to scipy CSR.
    """
    mat = as_backend_type(A).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return(csr)
