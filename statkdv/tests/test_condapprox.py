import unittest
import os

import h5py
import numpy as np

from scipy.sparse import csr_matrix
from statkdv.condapprox import KdVApprox
from statkdv.deterministic import KdV
from statkdv.utils import build_interpolation_matrix


class TestApprox(unittest.TestCase):
    def setUp(self):
        settings = {"x_start": 0,
                    "x_end": 2,
                    "t_start": 0,
                    "t_end": 0.1,
                    "nx": 200,
                    "nt": 11}
        parameters = {"alpha": 1., "beta": 1.}
        self.kdv = KdVApprox(settings, parameters)

    def test_init(self):
        # test that inputs OK
        self.assertEqual(self.kdv.nx, 200)
        self.assertEqual(self.kdv.nt, 11)

        self.assertEqual(self.kdv.alpha, 1.)
        self.assertEqual(self.kdv.beta, 1.)
        self.assertEqual(self.kdv.nu, 0.)
        self.assertEqual(self.kdv.c, 0.)

        # test matrix sizes OK
        self.assertEqual(self.kdv.mass.shape, (200, 200))
        self.assertEqual(self.kdv.mixed.shape, (200, 200))
        self.assertEqual(self.kdv.jacobian_base.shape, (600, 600))

        # check the defaults
        self.assertEqual(self.kdv.nugget, 1e-12)
        self.assertEqual(self.kdv.scale_G, None)
        self.assertEqual(self.kdv.length_G, None)
        self.assertEqual(self.kdv.sigma_y, None)
        self.assertEqual(self.kdv.estimate_G, True)
        self.assertEqual(self.kdv.estimate_sigma_y, True)

    def test_initial_condition(self):
        initial_condition = "cos(pi * x)"
        self.kdv.set_initial_condition(initial_condition)
        self.assertTrue(np.allclose(self.kdv.mean_u_prev, np.cos(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.mean_v_prev, -np.pi * np.sin(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.mean_w_prev, -np.pi**2 * np.cos(np.pi * self.kdv.x_grid)))

    def test_set_covariance(self):
        self.kdv.set_cov_square_exp(1, 1)
        self.assertEqual(self.kdv.G.shape, (600, 600))
        self.assertFalse(self.kdv.estimate_G)

    def test_set_sigma_y(self):
        self.kdv.set_sigma_y(0.01)
        self.assertEqual(self.kdv.sigma_y, 0.01)
        self.assertFalse(self.kdv.estimate_sigma_y)

    def test_fem_lhs(self):
        self.kdv.set_initial_condition("cos(pi * x)")
        lhs = self.kdv.fem_lhs(self.kdv.mean_u_curr,
                               self.kdv.mean_v_curr,
                               self.kdv.mean_w_curr,
                               self.kdv.mean_u_prev,
                               self.kdv.mean_v_prev,
                               self.kdv.mean_w_prev)

        self.assertEqual(lhs.shape, (600,))

    def test_fem_lhs_update_jacobian(self):
        settings = {"x_start": 0,
                    "x_end": 2,
                    "t_start": 0,
                    "t_end": 0.1,
                    "nx": 200,
                    "nt": 11}
        parameters = {"alpha": 1., "beta": 1.}
        u = np.cos(np.pi * self.kdv.x_grid)
        kdv_determ = KdV(settings, parameters)
        kdv_determ.u_prev = np.copy(u)

        j_curr_true = kdv_determ.fem_lhs_update_jacobian(u, u)
        j_curr = self.kdv.fem_lhs_update_jacobian_curr(u, u)
        self.assertEqual(type(j_curr), csr_matrix)
        self.assertEqual(j_curr.shape, (600, 600))

        self.assertTrue(j_curr.nnz == j_curr_true.nnz)
        self.assertTrue(j_curr.shape == j_curr_true.shape)

        j_prev = self.kdv.fem_lhs_update_jacobian_prev(u, u)
        self.assertEqual(type(j_prev), csr_matrix)
        self.assertEqual(j_prev.shape, (600, 600))

    def test_project_forward(self):
        x = np.array([0.5, 1., 1.5])
        y = np.sin(1.1 * np.pi * x)
        H = build_interpolation_matrix(self.kdv.x_grid, x)[:, 0:200]

        u = np.sin(np.pi * self.kdv.x_grid)

        x_project = np.linspace(0, 2, 100, endpoint=False)
        H_project = build_interpolation_matrix(self.kdv.x_grid, x_project)[:, 0:200]

        y_project = self.kdv.project_forward(y, u, H, H_project)
        self.assertEqual(len(y_project), 100)

    def test_optimize_lml(self):
        initial_condition = "cos(pi * x)"
        self.kdv.set_initial_condition(initial_condition)

        # HACK: just copy from kdv.solve_fem()
        priors = {"scale_G_mean": 1, "scale_G_sd": 1,
                  "length_G_mean": 1, "length_G_sd": 1,
                  "sigma_y_mean": 0, "sigma_y_sd": 1}
        control = {"newton_iter": 50, "newton_tol": 1e-8,
                   "scale_G_range": [0, 0.2],
                   "length_G_range": [0, 0.2],
                   "sigma_y_range": [0, 0.01],
                   "thin": 1, "noisefree": True, "project_data_forward": False}

        mean = np.concatenate((self.kdv.mean_u_prev,
                               self.kdv.mean_v_prev,
                               self.kdv.mean_w_prev))
        J_curr = self.kdv.fem_lhs_update_jacobian_curr(self.kdv.mean_u_prev,
                                                       self.kdv.mean_u_prev)
        inits = [1., 1., 0.05]

        # test success case
        x = self.kdv.x_grid
        y = 2 * np.cos(1.1 * np.pi * x)
        H = build_interpolation_matrix(self.kdv.x_grid, x)
        cov = np.kron(np.eye(3), self.kdv.cov_square_exp(x, 1., 1.))
        params = np.array(self.kdv.optimize_lml(inits, H, x, y, mean,
                                                cov, J_curr, priors))
        self.assertEqual(params.shape, (3, ))

        # # test failure case
        # self.kdv.nugget = 0.
        # self.assertRaises(StopIteration,
        #                   self.kdv.optimize_lml,
        #                   inits, H, x, y, mean, cov, J_curr, priors)

        # test function outputs
        params = np.array(inits)
        lp, grad = self.kdv.log_marginal_posterior(
            params, H, x, y, mean, cov, J_curr, priors
        )
        self.assertTrue(type(lp), np.float)
        self.assertTrue(len(grad) == 3)

        # test the gradients
        h = 1e-6
        directions = np.array([1., 1., 1.])
        lp_forward, g = self.kdv.log_marginal_posterior(
            params + h * directions, H, x, y, mean, cov, J_curr, priors
        )
        lp_backward, g = self.kdv.log_marginal_posterior(
            params - h * directions, H, x, y, mean, cov, J_curr, priors
        )
        d_dot_grad_est = (lp_forward - lp_backward) / (2 * h)
        grad_error_rel = (np.abs(d_dot_grad_est - np.dot(directions, grad))
                          / np.abs(np.dot(directions, grad)))
        print(grad_error_rel)
        self.assertTrue(grad_error_rel < 1e-4)

    # def test_solve_fem(self):
    #     # test that solving without initial conditions fails
    #     with self.assertRaises(ValueError):
    #         self.kdv.solve_fem()

    #     self.kdv.set_initial_condition("cos(pi * x)")
    #     self.kdv.set_cov_square_exp(1, 1)

    #     u, v, w = self.kdv.newton(self.kdv.mean_u_prev,
    #                               self.kdv.mean_v_prev,
    #                               self.kdv.mean_w_prev)

    #     self.assertEqual(u.shape, (200,))
    #     self.assertEqual(v.shape, (200,))
    #     self.assertEqual(w.shape, (200,))

    #     # test newton iterations fail
    #     with self.assertRaises(StopIteration):
    #         control = {"newton_iter": 50, "newton_tol": 1e-20}
    #         self.kdv.solve_fem(control=control)

    # def test_solve_prior(self):
    #     self.kdv.set_initial_condition("cos(pi * x)")
    #     self.kdv.set_cov_square_exp(1, 1)

    #     with self.assertLogs("statkdv.condapprox", level="INFO") as cm:
    #         self.kdv.solve_fem()

    #     # test that logs are OK
    #     self.assertTrue(
    #         "INFO:statkdv.condapprox:no conditioning: supply both y and H if desired" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condapprox:using default (weakly informative) priors" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condapprox:using default control" in cm.output
    #     )
    #     self.assertTrue(
    #         "WARNING:statkdv.condapprox:not saving output to disk" in cm.output
    #     )

    # def test_solve_conditional(self):
    #     self.kdv.set_initial_condition("cos(pi * x)")

    #     simdata = h5py.File("data/test-simdata.h5", "r")
    #     y = simdata["data/y"][:]
    #     x = simdata["data/x_grid"][:]
    #     H = build_interpolation_matrix(self.kdv.x_grid, x)
    #     data = {"y": y, "x_grid": x, "H": H}

    #     # test that outputs are OK
    #     self.kdv.solve_fem(data=data, output_file="temp.h5")
    #     with h5py.File("temp.h5", "r") as f:
    #         groups = [name for name in f]
    #         self.assertTrue(groups == ["control", "data", "model", "priors"])

    #         data_datasets = [data for data in f["data"]]
    #         model_datasets = [data for data in f["model"]]

    #         self.assertTrue(data_datasets == ["H", "x_grid", "y"])
    #         self.assertTrue(model_datasets == ["covariance",
    #                                            "mean",
    #                                            "parameters",
    #                                            "t_grid",
    #                                            "x_grid"])

    #         self.assertEqual(f["model/mean"][:].shape, (11, 200))
    #         self.assertEqual(f["model/covariance"][:].shape, (11, 200, 200))
    #         self.assertEqual(f["model/parameters"][:].shape, (10, 3))
    #         self.assertEqual(f["model/t_grid"][:].shape, (11,))
    #         self.assertEqual(f["model/x_grid"][:].shape, (200,))

    #     os.remove("temp.h5")


if __name__ == "__main__":
    unittest.main()
