import unittest

import numpy as np
from fenics import *
from scipy.sparse import csr_matrix

from ks import build_interpolation_matrix, KSSolve, StatKS
from statfenics import dolfin_to_csr


class TestKSSolve(unittest.TestCase):
    def setUp(self):
        self.ks = KSSolve(0.01)

    def test_init(self):
        self.assertEqual(self.ks.dt, 0.01)
        self.assertEqual(len(self.ks.V.tabulate_dof_coordinates()), 1024)

        self.assertTrue(hasattr(self.ks, "V"))
        self.assertTrue(hasattr(self.ks, "V_base"))
        self.assertTrue(hasattr(self.ks, "F"))
        self.assertTrue(hasattr(self.ks, "J"))
        self.assertTrue(hasattr(self.ks, "J_prev"))

    def test_functionality(self):
        u_prev = self.ks.get_u()
        self.ks.timestep()
        u_curr = self.ks.get_u()
        self.assertTrue(np.allclose(u_prev, u_curr) == False)

    def test_interpolation_matrix(self):
        x = self.ks.mesh.coordinates()[:-1]
        x_obs = x[::10]
        H = build_interpolation_matrix(x_obs, self.ks.V)

        self.assertEqual(H.shape, (52, 1024))
        self.assertEqual(type(H), csr_matrix)


class TestStatKS(unittest.TestCase):
    def setUp(self):
        self.ks = StatKS(0.01)

    def test_init(self):
        self.assertEqual(self.ks.dt, 0.01)
        self.assertEqual(len(self.ks.V.tabulate_dof_coordinates()), 1024)

        self.assertTrue(hasattr(self.ks, "M"))
        self.assertTrue(hasattr(self.ks, "P"))

        self.assertEqual(len(self.ks.params), 3)
        self.assertEqual(self.ks.nugget, 1e-10)
        self.assertEqual(self.ks.n_eigen, 50)

    def test_permutation(self):
        u_split = np.repeat([1, 2], 512)
        u_fenics = np.zeros(1024)

        u_dof_indices = self.ks.V.sub(0).dofmap().dofs()
        v_dof_indices = self.ks.V.sub(1).dofmap().dofs()

        u_fenics[u_dof_indices] = 1.
        u_fenics[v_dof_indices] = 2.

        self.assertEqual(
            np.sum(np.abs(self.ks.P @ u_split - u_fenics)), 0
        )

    def test_log_marginal_posterior(self):
        self.ks.timestep()

        x = self.ks.mesh.coordinates()[:-1]
        x_obs = x[::10]
        H = build_interpolation_matrix(x_obs, self.ks.V)
        mean_obs = H @ self.ks.mean
        cov_obs = H @ self.ks.cov @ H.T
        y_obs = 1.2 * mean_obs
        J_curr = dolfin_to_csr(assemble(self.ks.J))
        lp, grad = self.ks.log_marginal_posterior((1., 1., 1.),
                                                  self.ks.M, self.ks.V_base, H, J_curr,
                                                  mean_obs, cov_obs, x_obs, y_obs)
        self.assertEqual(type(lp), np.float64)
        self.assertEqual(len(grad), 3)

        # check gradients
        h = 1e-8
        d = [1., 1., 1.]
        lpf, _ = self.ks.log_marginal_posterior((1. + h, 1. + h, 1. + h),
                                                self.ks.M, self.ks.V_base, H, J_curr,
                                                mean_obs, cov_obs, x_obs, y_obs)
        lpb, _ = self.ks.log_marginal_posterior((1. - h, 1. - h, 1. - h),
                                                self.ks.M, self.ks.V_base, H, J_curr,
                                                mean_obs, cov_obs, x_obs, y_obs)
        grad_true = np.dot(grad, d)
        grad_est = (lpf - lpb) / (2 * h)
        rel_diff = np.abs(grad_true - grad_est) / np.abs(grad_true)
        self.assertTrue(rel_diff < 1e-4)
