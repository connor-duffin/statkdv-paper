import logging

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


logger = logging.getLogger(__name__)


class Interpolator():
    def __init__(self, u_grid, x_grid):
        """ FEM interpolant using piecewise linears

        Parameters
        ----------
        u_grid : np.ndarray
            coefficients at FEM nodes.

        x_grid : np.ndarray
            location of FEM nodes.
        """
        self.u = np.copy(u_grid)
        self.x_grid = np.copy(x_grid)

        self.h = x_grid[1] - x_grid[0]
        self.nx = len(x_grid)

    def evaluate(self, x):
        """ Evaluate at point x.

        Parameters
        ----------
        x : float, np.ndarray
            location(s) at which to evaluate the FEM interpolant
        """
        node = np.searchsorted(self.x_grid, x, side="right") - 1
        val = (self.u[node] * (self.x_grid[node] + self.h - x) / self.h
               + self.u[(node + 1) % self.nx] * (x - self.x_grid[node]) / self.h)
        return(val)


def build_interpolation_matrix(x_grid, x_obs):
    """ Create a matrix to interpolate FEM solution.

    Parameters
    ----------
    x_grid : np.ndarray
        Location of FEM nodes.

    x_obs : np.ndarray
        Location of observations.
    """

    n_grid = len(x_grid)
    n_obs = len(x_obs)
    h = x_grid[1] - x_grid[0]
    H = np.zeros((n_obs, 3 * n_grid))  # * 3 because of derivatives

    for i in range(n_obs):
        j = 0
        # increment until the upper bound
        while (x_obs[i] > x_grid[j]):
            j += 1

        if (np.abs(x_obs[i] - x_grid[j]) < 1e-8):
            H[i, j] = 1
        else:
            H[i, j - 1] = (x_grid[j] - x_obs[i]) / h
            H[i, j] = (x_obs[i] - x_grid[j - 1]) / h

    return sp.csr_matrix(H)  # return csr for matrix-vector products


def gl_weights_nodes(n):
    """ Compute Gauss quadrature weights and nodes.

    Exact (to machine precision) for polynomials up to order 2n - 1. Uses the
    Golub-Welsch algorithm.

    Parameters
    ----------
    n : int
        Number of Gauss quadrature points.
    """

    diag = np.zeros(n)
    n_array = np.array(range(n - 1))
    off_diag = (n_array + 1) / np.sqrt((2 * n_array + 1) * (2 * n_array + 3))
    nodes, v = la.eigh_tridiagonal(diag, off_diag)
    weights = 2 * v[0, :]**2

    return(weights, nodes)


def gl_interval(f, a=-1, b=1, n=5):
    """ Integrate f on interval [a, b] with Gauss-Legendre quadrature.

    Parameters
    ----------
    f : function
        Function to be integrated.
    a : float, optional
        Integration lower limit.
    b : float, optional
        Integration upper limit.
    n : int, optional
        Number of Gauss quadrature nodes.
    """

    weights, nodes = gl_weights_nodes(n)
    return(np.sum(
        (b - a) / 2 * weights * f((b - a) / 2 * nodes + (b + a) / 2)
    ))


def gl_grid(f, x_grid, n=5):
    """ Integrate a function over grid.

    Integrate f with Gauss-Legendre quadrature on a uniformly spaced grid by
    summing across constituent elements.

    Parameters
    ----------
    f : function
        Function to be integrated.
    x_grid : np.ndarray
        Grid on which to integrate over.
    n : int, optional
        Number of integration nodes.
    """

    weights, nodes = gl_weights_nodes(n)
    h = x_grid[1] - x_grid[0]
    integral = 0
    for i in range(len(x_grid)):
        integral += np.sum(
            h / 2 * weights * f(h / 2 * nodes + (2 * x_grid[i] + h) / 2)
        )
    return(integral)
