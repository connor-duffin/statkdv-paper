import os
import unittest

import h5py
import numpy as np

from scipy.sparse import csr_matrix
from statkdv.condmc import KdVEnsemble
from statkdv.utils import build_interpolation_matrix


class TestMC(unittest.TestCase):
    def setUp(self):
        settings = {
            "x_start": 0,
            "x_end": 2,
            "t_start": 0,
            "t_end": 0.1,
            "nx": 200,
            "nt": 11,
            "n_ensemble": 10
        }
        parameters = {
            "alpha": 1,
            "beta": 1
        }
        self.kdv = KdVEnsemble(settings, parameters)

    def test_init(self):
        # make sure endpoints are OK
        self.assertNotEqual(self.kdv.x_grid[-1], 2)
        self.assertEqual(self.kdv.t_grid[-1], 0.1)

        # default coefficients OK
        self.assertEqual(self.kdv.c, 0.)
        self.assertEqual(self.kdv.nu, 0.)

        # FEM matrices OK
        self.assertEqual(self.kdv.mass.shape, (200, 200))
        self.assertEqual(self.kdv.mixed.shape, (200, 200))
        self.assertEqual(self.kdv.jacobian_base.shape, (600, 600))

        # ensemble size OK
        self.assertEqual(self.kdv.state.shape, (10, 600))

        # parameters OK
        self.assertEqual(self.kdv.nugget, 1e-12)
        self.assertEqual(self.kdv.scale_G, None)
        self.assertEqual(self.kdv.length_G, None)
        self.assertEqual(self.kdv.sigma_y, None)
        self.assertEqual(self.kdv.estimate_G, True)
        self.assertEqual(self.kdv.estimate_sigma_y, True)

    def test_initial_condition(self):
        # test that all initial conditions are OK
        initial_condition = "cos(pi * x)"
        self.kdv.set_initial_condition(initial_condition)
        self.assertTrue(np.allclose(self.kdv.u_prev[0, :], np.cos(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.v_prev[0, :], -np.pi * np.sin(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.w_prev[0, :], -np.pi**2 * np.cos(np.pi * self.kdv.x_grid)))

        # test that ensemble size is OK
        self.assertEqual(self.kdv.u_prev.shape, (10, 200))
        self.assertEqual(self.kdv.v_prev.shape, (10, 200))
        self.assertEqual(self.kdv.w_prev.shape, (10, 200))

    def test_set_parameters(self):
        self.kdv.set_cov_square_exp(1, 1)
        self.assertFalse(self.kdv.estimate_G)

        # check that G, G_chol both created
        self.assertEqual(self.kdv.G_chol_fixed.shape, (200, 200))
        self.assertEqual(self.kdv.G_chol_fixed.shape, (200, 200))

        # check that G is lower triangular
        self.assertTrue(np.allclose(self.kdv.G_chol_fixed,
                                    np.tril(self.kdv.G_chol_fixed)))

        # noise
        self.kdv.set_sigma_y(0.1)
        self.assertFalse(self.kdv.estimate_sigma_y)
        self.assertEqual(self.kdv.sigma_y, 0.1)

    def test_fem_lhs(self):
        self.kdv.set_initial_condition("cos(pi * x)")
        self.kdv.set_cov_square_exp(1, 1)

        noise = self.kdv.G_chol_fixed @ np.random.normal(size=(self.kdv.nx,))

        u, v, w = (self.kdv.u_prev[0, :].copy(),
                   self.kdv.v_prev[0, :].copy(),
                   self.kdv.w_prev[0, :].copy())
        u_prev, v_prev, w_prev = (self.kdv.u_prev[0, :].copy(),
                                  self.kdv.v_prev[0, :].copy(),
                                  self.kdv.w_prev[0, :].copy())
        lhs = self.kdv.fem_lhs(u, v, w, u_prev, v_prev, w_prev, noise)
        self.assertEqual(len(lhs), 600)

    def test_fem_lhs_update_jacobian(self):
        # compute jacobian as if starting from initial values
        u = np.cos(np.pi * self.kdv.x_grid)
        j = self.kdv.fem_lhs_update_jacobian(u, u)

        self.assertEqual(type(j), csr_matrix)
        self.assertEqual(j.shape, (600, 600))

    def test_newton_timestep(self):
        self.kdv.set_initial_condition("cos(pi * x)")
        self.kdv.set_cov_square_exp(1, 1)

        u, v, w = self.kdv.newton_timestep(
            self.kdv.u_prev[0, :],
            self.kdv.v_prev[0, :],
            self.kdv.w_prev[0, :],
            xi=np.zeros_like(self.kdv.u_prev[0, :])
        )

        # test lengths equal
        self.assertEqual(len(u), len(self.kdv.u_prev[0, :]))
        self.assertEqual(len(v), len(self.kdv.u_prev[0, :]))
        self.assertEqual(len(w), len(self.kdv.u_prev[0, :]))

        # test expected failure
        with self.assertRaises(StopIteration):
            self.kdv.newton_timestep(
                self.kdv.u_prev[0, :],
                self.kdv.v_prev[0, :],
                self.kdv.w_prev[0, :],
                xi=np.zeros_like(self.kdv.u_prev[0, :]),
                n_iter=50,
                tol=1e-20
            )

    def test_project_forward(self):
        x = np.array([0.5, 1., 1.5])
        y = np.sin(1.1 * np.pi * x)
        H = build_interpolation_matrix(self.kdv.x_grid, x)[:, 0:200]

        u = np.sin(np.pi * self.kdv.x_grid)

        x_project = np.linspace(0, 2, 100, endpoint=False)
        H_project = build_interpolation_matrix(self.kdv.x_grid, x_project)[:, 0:200]

        y_project = self.kdv.project_forward(y, u, H, H_project)
        self.assertEqual(len(y_project), 100)

    def test_log_marginal_posterior(self):
        params = np.array([1, 1, 0.05])
        h = 1e-5

        # some fake data
        H = np.eye(self.kdv.nx, 3 * self.kdv.nx)
        x_obs = self.kdv.x_grid.copy()
        y_obs = 1.1 * np.cos(np.pi * self.kdv.x_grid)

        # fake solution
        mean = np.cos(np.pi * self.kdv.x_grid)
        mean_full = np.concatenate((mean, mean, mean))
        cov = np.eye(3 * self.kdv.nx)
        J_mean = self.kdv.fem_lhs_update_jacobian(mean, mean)

        # default priors
        priors = {"scale_G_mean": 1,
                  "scale_G_sd": 1,
                  "length_G_mean": 1,
                  "length_G_sd": 1,
                  "sigma_y_mean": 0,
                  "sigma_y_sd": 1}

        lp, grad = self.kdv.log_marginal_posterior(
            params, H, x_obs, y_obs, mean_full, cov, J_mean, priors
        )
        # test outputs
        self.assertEqual(lp.shape, (1,))
        self.assertEqual(grad.shape, (3,))

        lp_forward, _ = self.kdv.log_marginal_posterior(
            params + h, H, x_obs, y_obs, mean_full, cov, J_mean, priors
        )
        lp_backward, _ = self.kdv.log_marginal_posterior(
            params - h, H, x_obs, y_obs, mean_full, cov, J_mean, priors
        )
        grad_est = np.float((lp_forward - lp_backward) / (2 * h))
        grad_true = np.dot(grad, np.array([1., 1., 1.]))
        rel_diff = np.abs(grad_est - grad_true) / np.abs(grad_est)
        print(rel_diff)
        self.assertTrue(rel_diff < 1e-8)

    # def test_solve_prior(self):
    #     # test that solving without initial conditions fails
    #     with self.assertRaises(ValueError):
    #         self.kdv.solve_ensemble()

    #     self.kdv.set_initial_condition("cos(pi * x)")
    #     self.kdv.set_cov_square_exp(1, 1)

    #     # test that default behaviour is observed
    #     with self.assertLogs("statkdv.condmc", level="INFO") as cm:
    #         self.kdv.solve_ensemble()

    #     self.assertTrue(
    #         "INFO:statkdv.condmc:not conditioning on data" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condmc:using default comm: MPI.COMM_WORLD" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condmc:using default (weakly informative) priors" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condmc:using default control" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condmc:not saving output to disk" in cm.output
    #     )

    # def test_solve_conditional(self):
    #     self.kdv.set_initial_condition("cos(pi * x)")

    #     simdata = h5py.File("data/test-simdata.h5", "r")
    #     y = simdata["data/y"][:]
    #     x = simdata["data/x_grid"][:]
    #     H = build_interpolation_matrix(self.kdv.x_grid, x)
    #     data = {"y": y, "x_grid": x, "H": H}

    #     with self.assertLogs("statkdv.condmc", level="INFO") as cm:
    #         self.kdv.solve_ensemble(data=data, output_file="temp.h5")

    #     self.assertTrue("INFO:statkdv.condmc:output file: temp.h5" in cm.output)
    #     with h5py.File("temp.h5", "r") as f:
    #         groups = [name for name in f]
    #         self.assertTrue(groups == ["control", "data", "model", "priors"])

    #         data_datasets = [data for data in f["data"]]
    #         self.assertTrue(data_datasets == ["H", "t_grid", "x_grid", "y"])
    #         model_datasets = [data for data in f["model"]]
    #         self.assertTrue(model_datasets == ["covariance",
    #                                            "covariance_obs",
    #                                            "mean",
    #                                            "mean_obs",
    #                                            "parameters",
    #                                            "t_grid",
    #                                            "x_grid"])
    #         self.assertEqual(f["model/mean"][:].shape, (11, 200))
    #         self.assertEqual(f["model/covariance"][:].shape, (11, 200, 200))
    #         self.assertEqual(f["model/parameters"][:].shape, (10, 3))
    #         self.assertEqual(f["model/t_grid"][:].shape, (11,))
    #         self.assertEqual(f["model/x_grid"][:].shape, (200,))

    #     os.remove("temp.h5")

    # def test_solve_conditional_projection(self):
    #     """ Integration test for the posterior w projection
    #     """
    #     np.random.seed(420)
    #     self.kdv.set_initial_condition("cos(pi * x)")
    #     self.kdv.set_cov_square_exp(1, 1)

    #     simdata = h5py.File("data/test-simdata.h5", "r")
    #     y = simdata["data/y"][:]
    #     x = simdata["data/x_grid"][:]
    #     H = build_interpolation_matrix(self.kdv.x_grid, x)
    #     x_project = np.linspace(0, 2, 100, endpoint=False)
    #     H_project = build_interpolation_matrix(self.kdv.x_grid, x_project)

    #     data = {
    #         "y": y, "x_grid": x, "H": H, "x_project": x_project, "H_project": H_project
    #     }
    #     control = {"project_data_forward": True}
    #     with self.assertLogs("statkdv.condmc", level="INFO") as cm:
    #         self.kdv.solve_ensemble(data=data, control=control)

    #     self.assertTrue(
    #         "INFO:statkdv.condmc:using projection to estimate covariance parameters" in cm.output
    #     )


if __name__ == "__main__":
    unittest.main()
