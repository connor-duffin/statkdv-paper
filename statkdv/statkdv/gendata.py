import logging

import h5py
import numpy as np

from scipy.linalg import cholesky
from scipy.spatial.distance import pdist, squareform

from .deterministic import KdV


logger = logging.getLogger(__name__)


def _cov_square_exp(x, scale, length):
    """ Evaluate the squared exponential function on the grid x.
    """
    x = np.expand_dims(x, 1)
    K = np.exp(-0.5 * pdist(x / length, metric="sqeuclidean"))
    K = squareform(K)
    np.fill_diagonal(K, 1)
    K = scale**2 * K
    return K


def _save_data(output_file, y, x, t):
    """ Helper function to write data to output file.
    """
    with h5py.File(output_file, "w") as f:
        data = f.create_group("data")
        data.create_dataset("y", (len(t), len(x)), data=y)
        data.create_dataset("x_grid", (len(x),), data=x)
        data.create_dataset("t_grid", (len(t),), data=t)


def generate_kdv_noise(
    kdv_parameters, kdv_setttings, kdv_initial, mismatch_parameters, output_file
):
    """ Generate KdV with random Gaussian noise.

    Parameters
    ----------
    kdv_parameters: dict
        Parameters to be passed to deterministic KdV function.
    kdv_settings: dict
        Settings to be passed to deterministic KdV function.
    kdv_initial: str
        Sympy expression for initial conditions (must have x as dependent variable).
    mismatch_parameters: dict
        Containts `rho`, scale factor, and `sigma_noise`, the scale parameters
        of the Gaussian noise.
    """
    try:
        rho = mismatch_parameters["rho"]
        sigma_noise = mismatch_parameters["sigma_noise"]
    except KeyError:
        logger.error("mismatch_parameters dict missing values")
        raise

    kdv = KdV(kdv_parameters, kdv_setttings)
    kdv.set_initial_condition(kdv_initial)
    kdv.solve_fem()

    nt = kdv.nt
    nx = kdv.nx

    y = np.zeros((nt, nx))
    for i in range(nt):
        y[i, :] = (
            rho * kdv.u_nodes[i, :]
            + np.random.normal(scale=sigma_noise, size=(nx, ))
        )

    _save_data(output_file, y, kdv.x_grid, kdv.t_grid)


def generate_kdv_noise_delta(
    kdv_parameters, kdv_setttings, kdv_initial, mismatch_parameters, output_file
):
    """ Generate KdV with random Gaussian noise and systematic mismatch.

    Parameters
    ----------
    kdv_parameters: dict
        Parameters to be passed to deterministic KdV function.
    kdv_settings: dict
        Settings to be passed to deterministic KdV function.
    kdv_initial: str
        Sympy expression for initial conditions (must have x as dependent
        variable).
    mismatch_parameters: dict
        Contains `rho`, scale factor, `sigma_noise`, the scale parameters
        of the Gaussian noise, `delta_scale`, the functional mismatch amplitude
        parameter, and `delta_length`, the functional mismatch length
        parameter.
    """
    try:
        rho = mismatch_parameters["rho"]
        sigma_noise = mismatch_parameters["sigma_noise"]
        delta_scale = mismatch_parameters["delta_scale"]
        delta_length = mismatch_parameters["delta_length"]
    except KeyError:
        logger.error("mismatch_parameters dict missing values")
        raise

    kdv = KdV(kdv_parameters, kdv_setttings)
    kdv.set_initial_condition(kdv_initial)
    kdv.solve_fem()

    nt = kdv.nt
    nx = kdv.nx

    cov_se = _cov_square_exp(kdv.x_grid, delta_scale, delta_length)
    cov_all_chol = cholesky(
        cov_se + sigma_noise**2 * np.eye(nx),
        lower=True
    )
    mismatch = cov_all_chol @ np.random.normal(size=(nx, ))

    y = np.zeros((nt, nx))
    for i in range(nt):
        y[i, :] = rho * kdv.u_nodes[i, :] + mismatch

    _save_data(output_file, y, kdv.x_grid, kdv.t_grid)
