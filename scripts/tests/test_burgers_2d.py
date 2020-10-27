import pytest

from fenics import *

import numpy as np
from burgers_2d_ekf import build_observation_operator, dolfin_to_csr, BurgersExtended



def test_init():
    burgers = BurgersExtended()

    assert burgers.nx == 64
    assert burgers.dt == 0.01
    assert burgers.L == 2.


def test_obs_operator():
    burgers = BurgersExtended(nx=32)
    u_dof_indices = np.array(burgers.V.sub(0).dofmap().dofs())
    obs_indices = u_dof_indices[::10]
    x_obs = burgers.V.tabulate_dof_coordinates()[obs_indices]

    w = burgers.w.vector()[:]
    u_obs = w[obs_indices]
    H = build_observation_operator(x_obs, burgers.V)
    n_obs = len(obs_indices)

    assert H.shape == (n_obs, 2048)
    assert np.linalg.norm(u_obs - H @ w, 1) < 1e-12

    # operator is what we think it should be
    H_test = np.zeros((len(obs_indices), len(burgers.mean)))
    for i, j in enumerate(obs_indices):
        H_test[i, j] = 1.

    assert np.allclose(H.A, H_test)


def test_lex_ordering():
    burgers = BurgersExtended(nx=32)
    burgers.set_lex_ordering()

    assert len(burgers.u_dof_coords) == 1024
    assert burgers.u_dof_coords_sorted.shape == (1024, 2)
    np.testing.assert_allclose((burgers.P @ burgers.P.T).toarray(), np.eye(2048))


def test_set_covariance_parameters():
    burgers = BurgersExtended(nx=32)

    burgers.set_covariance_parameters()
    assert burgers.estimate_params == True

    burgers.set_covariance_parameters((0.5, 0.005))
    assert burgers.scale_G == 0.5
    assert burgers.sigma_y == 0.005
    assert burgers.estimate_params == False


def test_base_covariance():
    burgers = BurgersExtended(nx=32)
    burgers.set_lex_ordering()
    burgers.set_base_covariance(np.sqrt(0.1))

    assert burgers.G_base_vals.shape == (128, )
    assert burgers.G_base_vecs.shape == (2048, 128)


def test_log_marginal_posterior():
    burgers = BurgersExtended(nx=32)
    burgers.set_lex_ordering()
    burgers.set_base_covariance(0.1)
    burgers.set_covariance_parameters((0.01, 0.005))

    u_dof_indices = np.copy(burgers.u_dof_indices)
    obs_indices = u_dof_indices[::10]
    x_obs = burgers.V.tabulate_dof_coordinates()[obs_indices]

    w = burgers.w.vector()[:]
    y_obs = 0.8 * w[obs_indices]
    n_obs = len(obs_indices)

    mean_obs = w[obs_indices]
    cov_obs = np.eye(n_obs)
    G_hat_obs = np.eye(n_obs)

    lp, grad = burgers.log_marginal_posterior((1., 1.),
                                              y_obs, mean_obs, cov_obs, G_hat_obs)

    assert type(lp) == np.float64
    assert len(grad) == 2

    h = 1e-6
    d = [1., 1.]
    lpf, _ = burgers.log_marginal_posterior((1. + h, 1. + h),
                                            y_obs, mean_obs, cov_obs, G_hat_obs)
    lpb, _ = burgers.log_marginal_posterior((1. - h, 1. - h),
                                            y_obs, mean_obs, cov_obs, G_hat_obs)
    grad_true = np.dot(grad, d)
    grad_est = (lpf - lpb) / (2 * h)
    rel_diff = np.abs(grad_true - grad_est) / np.abs(grad_true)
    assert rel_diff < 1e-8
