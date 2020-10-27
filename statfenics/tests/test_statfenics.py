import numpy as np
import statfenics as sf

from scipy.sparse import csr_matrix

from fenics import *


def test_square_exp():
    x_grid = np.linspace(0, 1, 20)
    K = sf.cov_square_exp(x_grid, 1., 1.)

    assert K.shape == (20, 20)


def test_log_marginal_likelihood():
    x = np.linspace(0, 1, 20)
    params = np.array([1., 1.])
    mean_obs = np.sin(np.pi * x)
    cov_obs = np.eye(20)
    x_obs = np.copy(x)
    y_obs = 2 * mean_obs
    priors = {
        "scale_delta_mean": 0.,
        "scale_delta_sd": 1.,
        "length_delta_mean": 0.,
        "length_delta_sd": 1.
    }

    lml = sf.log_marginal_likelihood(
        params, mean_obs, cov_obs, x_obs, y_obs, priors, True
    )
    lml_prime = sf.log_marginal_likelihood_derivative(
        params, mean_obs, cov_obs, x_obs, y_obs, priors, True
    )
    params = sf.optimize_lml(mean_obs, cov_obs, x_obs, y_obs, priors, noisefree=True)

    assert lml.shape == (1, )
    assert lml_prime.shape == (2, )
    assert params.shape == (2, )


def test_build_covariance_matrix():
    # unit interval, P1 elements
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "P", 1)

    G = sf.build_covariance_matrix(V, 1., 1.)

    assert type(G) == np.ndarray
    assert G.shape == (9, 9)

    # unit square, P2 elements
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "P", 2)

    G = sf.build_covariance_matrix(V, 1., 1.)

    assert type(G) == np.ndarray
    assert G.shape == (289, 289)


def test_build_interpolation_matrix():
    # unit interval, P1 elements
    mesh = UnitIntervalMesh(32)
    V = FunctionSpace(mesh, "P", 1)

    x_obs = np.linspace(0, 1, 10).reshape((10, 1))
    H = sf.build_interpolation_matrix(x_obs, V)

    assert H.shape == (10, 33)
    assert type(H) == csr_matrix


def test_dolfin_to_csr():
    # unit interval, P1 elements
    mesh = UnitIntervalMesh(32)
    V = FunctionSpace(mesh, "P", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    form = u * v * dx
    M = assemble(form)

    M_csr = sf.dolfin_to_csr(M)

    assert M_csr.shape == (33, 33)
    assert type(M_csr) == csr_matrix
