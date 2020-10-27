import unittest
import numpy as np

from statkdv import utils


class TestUtils(unittest.TestCase):

    def test_interpolator(self):
        x_grid = np.linspace(0, 2, endpoint=False)
        u_grid = np.cos(np.pi * x_grid)
        interpolator = utils.Interpolator(u_grid, x_grid)

        self.assertEqual(interpolator.nx, 50)
        self.assertEqual(interpolator.h, x_grid[1] - x_grid[0])
        self.assertTrue(np.allclose(interpolator.u, u_grid))
        self.assertTrue(np.allclose(interpolator.x_grid, x_grid))

        self.assertTrue(np.allclose(interpolator.evaluate(1.), -1.))
        self.assertTrue(np.allclose(interpolator.evaluate(1.5), 0.))

    def test_build_interpolation_matrix(self):
        x_obs = np.linspace(0, 2, 10, endpoint=False)
        x_grid = np.linspace(0, 2, 50, endpoint=False)

        H = utils.build_interpolation_matrix(x_grid, x_obs)
        self.assertTrue(H.shape, (10, 150))
        self.assertTrue(np.allclose(H[:, ::5].diagonal(),
                                    np.array(10 * [1.])))
        # check all other entries are 0
        self.assertTrue(not np.any(H.todense()[:, 50:]))

    def test_gl_weights_nodes(self):
        weights_true = np.array([
            0.5688888888888889,
            0.4786286704993665,
            0.4786286704993665,
            0.2369268850561891,
            0.2369268850561891
        ])

        nodes_true = np.array([
            0.0000000000000000,
            -0.5384693101056831,
            0.5384693101056831,
            -0.9061798459386640,
            0.9061798459386640
        ])
        weights, nodes = utils.gl_weights_nodes(5)

        self.assertTrue(np.allclose(np.sort(weights_true), np.sort(weights)))
        self.assertTrue(np.allclose(np.sort(nodes_true), np.sort(nodes)))

    def test_gl_interval(self):
        def f(x):
            return(x**3 + 5 * x**2 - x + 5)

        self.assertTrue(np.allclose(40 / 3, utils.gl_interval(f)))
        # check on different grid
        self.assertTrue(np.allclose(84, utils.gl_interval(f, -8, 4)))

    def test_gl_grid(self):
        def f(x):
            return(x**3 + 5 * x**2 - x - 10)

        x_grid = np.linspace(-8, 4, 10, endpoint=False)
        self.assertTrue(np.allclose(utils.gl_grid(f, x_grid), -96.))





if __name__ == "__main__":
    unittest.main()
