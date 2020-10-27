import unittest
import os
import h5py
import numpy as np

from scipy.sparse import csr_matrix
from statkdv.deterministic import KdV


class TestDeterministic(unittest.TestCase):
    def setUp(self):
        settings = {
            "x_start": 0,
            "x_end": 3,
            "t_start": 0,
            "t_end": 0.1,
            "nx": 200,
            "nt": 11
        }
        parameters = {
            "alpha": 1,
            "beta": 1e-3
        }
        self.kdv = KdV(settings, parameters)

    def test_init(self):
        self.assertEqual(self.kdv.c, 0.)
        self.assertEqual(self.kdv.nu, 0.)
        self.assertEqual(self.kdv.u_nodes.shape, (11, 200))
        self.assertEqual(self.kdv.mass.shape, (200, 200))
        self.assertEqual(self.kdv.mixed.shape, (200, 200))
        self.assertEqual(self.kdv.jacobian_base.shape, (600, 600))

    def test_initial_condition(self):
        initial_condition = "cos(pi * x)"
        self.kdv.set_initial_condition(initial_condition)
        self.assertTrue(np.allclose(self.kdv.u_prev,
                                    np.cos(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.v_prev,
                                    -np.pi * np.sin(np.pi * self.kdv.x_grid)))
        self.assertTrue(np.allclose(self.kdv.w_prev,
                                    -np.pi**2 * np.cos(np.pi * self.kdv.x_grid)))

        initial_condition = "Piecewise((x, x < 1), (2 - x, True))"
        self.kdv.set_initial_condition(initial_condition)
        self.assertTrue(np.allclose(self.kdv.w_prev, 0.))

    def test_fem_lhs(self):
        self.kdv.set_initial_condition("cos(pi * x)")
        u, v, w = self.kdv.u_prev, self.kdv.v_prev, self.kdv.w_prev
        u_prev, v_prev, w_prev = self.kdv.u_prev, self.kdv.v_prev, self.kdv.w_prev
        f = self.kdv.fem_lhs(u, v, w, u_prev, v_prev, w_prev)
        j = self.kdv.fem_lhs_update_jacobian(u, u_prev)

        self.assertEqual(len(f), 600)
        self.assertEqual(type(j), csr_matrix)
        self.assertEqual(j.shape, (600, 600))

    def test_newton(self):
        self.kdv.set_initial_condition("cos(pi * x)")
        u, v, w = self.kdv.newton(self.kdv.u_prev,
                                  self.kdv.v_prev,
                                  self.kdv.w_prev)

        # test lengths equal
        self.assertEqual(len(u), len(self.kdv.u_prev))
        self.assertEqual(len(v), len(self.kdv.u_prev))
        self.assertEqual(len(w), len(self.kdv.u_prev))

        # test expected failure
        with self.assertRaises(StopIteration):
            self.kdv.newton(self.kdv.u_prev, self.kdv.v_prev, self.kdv.w_prev,
                            n_iter=50, tol=1e-20)

    def test_solve_fem(self):
        # test that solving without initial conditions fails
        with self.assertRaises(ValueError):
            self.kdv.solve_fem()

        # check solution behaviour is all good
        self.kdv.set_initial_condition(
            "3 * 0.5 * (1 / cosh(1 / 2 * sqrt(0.5 / 1e-3) * (x - 1)))**2"
        )
        with self.assertLogs("statkdv.deterministic", level="INFO") as cm:
            self.kdv.solve_fem(output_file="temp.h5")

        self.assertTrue("WARNING:statkdv.deterministic:using default control" in
                        cm.output)

        # check that output is all good
        with h5py.File("temp.h5", "r") as f:
            groups = [name for name in f]
            self.assertEqual(groups, ["control", "model"])
            self.assertEqual(f["model"].attrs["alpha"], 1.)
            self.assertEqual(f["model"].attrs["beta"], 0.001)
            self.assertEqual(f["model"].attrs["c"], 0.)
            self.assertEqual(f["model"].attrs["nu"], 0.)

            self.assertEqual(f["model/u_nodes"][:].shape, (11, 200))
            self.assertEqual(f["model/v_nodes"][:].shape, (11, 200))
            self.assertEqual(f["model/w_nodes"][:].shape, (11, 200))

            analytical = 3 * 0.5 * (1 / np.cosh(0.5 * np.sqrt(0.5 / 1e-3) * (self.kdv.x_grid - 1 - 0.5 * 0.1)))**2
            self.assertTrue(np.allclose(np.sum(np.abs(f["model/u_nodes"][10, :] - analytical)),
                                        0.21722917))

        os.remove("temp.h5")


if __name__ == "__main__":
    unittest.main()
