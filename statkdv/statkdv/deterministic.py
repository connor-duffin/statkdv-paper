import logging
import time

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy


logger = logging.getLogger(__name__)


class KdV():
    """ Solve the KdV equation using FEM

    Parameters
    ----------
    settings : dict

    parameters : dict


    Attributes
    ----------
    nx: int
        Number of FEM nodes
    nt: int
        Number of timesteps
    dx: float
        FEM gridspace
    dt: float
        Timestep gridspace
    x_grid: 1d array
        Array of FEM nodes
    t_grid: 1d array
        Array of timepoints
    alpha: float
        KdV steepening parameter
    beta: float
        KdV dispersion parameter
    c: float
        KdV speed parameter
    mass: 2d array
        FEM mass matrix
    mixed: 2d array
        FEM mixed derivative matrix
    jacobian_base: 2d array
        base matrix for the jacobian matrix
    u: 1d array
        current FEM solution
    v: 1d array
        first spatial derivative of current FEM solution
    w: 1d array
        second spatial derivative of current FEM solution
    u_prev: 1d array
        previous FEM solution
    v_prev: 1d array
        previous spatial derivative of current FEM solution
    w_prev: 1d array
        previous spatial derivative of current FEM solution
    """
    def __init__(self, settings, parameters):
        try:
            self.nx = settings["nx"]
            self.nt = settings["nt"]
            self.x_grid = np.linspace(
                settings["x_start"],
                settings["x_end"],
                settings["nx"],
                endpoint=False
            )
            self.t_grid = np.linspace(
                settings["t_start"],
                settings["t_end"],
                settings["nt"]
            )
        except KeyError:
            logger.error("settings dictionary missing values")
            raise

        # take in parameters as appropriate
        # alpha and beta are required
        if "alpha" not in parameters or "beta" not in parameters:
            logger.error("parameters dictionary missing values")
            raise KeyError
        else:
            self.alpha = parameters["alpha"]
            self.beta = parameters["beta"]

        if "c" not in parameters:
            logger.warning("no c provided: setting to zero")
            self.c = 0
        else:
            self.c = parameters["c"]

        if "nu" not in parameters:
            logger.warning("no nu provided: setting to zero")
            self.nu = 0
        else:
            self.nu = parameters["nu"]

        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dt = self.t_grid[1] - self.t_grid[0]

        self.mass = sp.diags(
            diagonals=[
                np.full(self.nx, self.dx / 2),
                np.full(self.nx - 1, self.dx / 2),
                self.dx / 2
            ],
            offsets=[0, 1, -(self.nx - 1)]
        )

        self.mixed = sp.diags(
            diagonals=[
                np.full(self.nx, -1),
                np.full(self.nx - 1, 1),
                1
            ],
            offsets=[0, 1, -(self.nx - 1)]
        )

        self.jacobian_base = (
            sp.kron(
                [[self.dt * self.c, self.dt * self.beta, 0], [0, 0, -1], [-1, 0, 0]],
                self.mixed / 2,
                format="csr"
            )
            + sp.kron(
                [[(1 + self.dt * self.nu / 2), 0, 0], [0, 1 / 2, 0], [0, 0, 1 / 2]],
                self.mass,
                format="csr"
            )
        )

        self.u = None
        self.v = None
        self.w = None

        self.u_prev = None
        self.v_prev = None
        self.w_prev = None

        self.u_nodes = np.zeros((self.nt, self.nx))

    def set_initial_condition(self, u_expr, dep_var=None):
        """ Set the initial conditions.

        Parameters
        ----------
        u_expr : str
            Sympy expression for u value, as a string.
        dep_var : sympy symbol, optional
            Variable with which to take derivatives of u_expr with. Defaults to x.
        """
        if type(u_expr) != str:
            logger.error("input expression must be a string")
            raise TypeError

        if dep_var is None:
            dep_var = sympy.symbols("x")
            logger.warning("assuming initial condition dependent variable is x")

        u_init = sympy.sympify(u_expr)
        v_init = sympy.diff(u_init, dep_var)
        w_init = sympy.diff(u_init, dep_var, 2)

        f_u_init = sympy.lambdify(dep_var, u_init, "numpy")
        f_v_init = sympy.lambdify(dep_var, v_init, "numpy")
        f_w_init = sympy.lambdify(dep_var, w_init, "numpy")

        self.u_prev = np.zeros(self.nx)
        self.v_prev = np.zeros(self.nx)
        self.w_prev = np.zeros(self.nx)

        self.u_prev[:] = f_u_init(self.x_grid)
        self.v_prev[:] = f_v_init(self.x_grid)
        self.w_prev[:] = f_w_init(self.x_grid)

        self.u_nodes[0, :] = self.u_prev

    def fem_lhs(self, u, v, w, u_prev, v_prev, w_prev):
        """ LHS of FEM expression.
        """
        mass, mixed = self.mass, self.mixed
        f1 = (
            mass @ (u - u_prev)
            + (self.dt * self.nu) * mass @ ((u + u_prev) / 2)
            + (self.dt * self.alpha) * mixed @ ((u + u_prev)**2 / 4) / 2
            + (self.dt * self.beta) * mixed @ ((w + w_prev) / 2)
            + (self.dt * self.c) * mixed @ ((u + u_prev) / 2)
        )
        f2 = mass @ ((w + w_prev) / 2) - mixed @ ((v + v_prev) / 2)
        f3 = mass @ ((v + v_prev) / 2) - mixed @ ((u + u_prev) / 2)

        return(np.concatenate((f1, f2, f3)))

    def fem_lhs_update_jacobian(self, u, u_prev):
        """ Update the base Jacobian of the FEM expression.
        """
        jacobian = (self.jacobian_base
                    + sp.kron([[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                              self.dt * self.alpha * self.mixed.multiply((u + self.u_prev) / 2) / 2,
                              format="csr"))
        return(jacobian)

    def newton(self, u_prev, v_prev, w_prev, n_iter=50, tol=1e-8):
        """ Complete one newton iteration.
        """
        u, v, w = np.copy(u_prev), np.copy(v_prev), np.copy(w_prev)

        for i in range(n_iter):
            f_curr = self.fem_lhs(u, v, w, u_prev, v_prev, w_prev)
            j_curr = self.fem_lhs_update_jacobian(u, u_prev)
            shift = spla.spsolve(j_curr, -f_curr)
            u += shift[0:self.nx]
            w += shift[self.nx:(2 * self.nx)]
            v += shift[(2 * self.nx):]
            if np.sum(np.abs(shift)) < tol:
                logger.debug("newton took {} iter".format(i))
                break
            elif i == n_iter - 1:
                logger.error("max newton iterations")
                raise StopIteration

        return(u, v, w)

    def solve_fem(self, control=None, output_file=None):
        """ Solve KdV with the given configuration using FEM.

        Parameters
        ----------
        control : dict, optional
            Tells FEM how to behave. If `None` is given then the default is used.
        output_file : str, optional
            Location of output file. Stored with HDF5.
        """
        if self.u_prev is None or self.v_prev is None or self.w_prev is None:
            logger.error("all initial values are None")
            raise ValueError

        if control is None:
            logger.warning("using default control")
            control = {"newton_iter": 50, "newton_tol": 1e-8}

        if output_file is not None:
            output = h5py.File(output_file, "w")
            model_output = output.create_group("model")
            control_output = output.create_group("control")

            model_output.attrs["type"] = "deterministic"

            model_output.create_dataset("x_grid", data=self.x_grid)
            model_output.create_dataset("t_grid", data=self.t_grid)

            model_output.attrs["alpha"] = self.alpha
            model_output.attrs["beta"] = self.beta
            model_output.attrs["c"] = self.c
            model_output.attrs["nu"] = self.nu

            for key in control.keys():
                control_output[key] = control[key]

            u_nodes = model_output.create_dataset("u_nodes", (self.nt, self.nx))
            v_nodes = model_output.create_dataset("v_nodes", (self.nt, self.nx))
            w_nodes = model_output.create_dataset("w_nodes", (self.nt, self.nx))
            u_nodes[0, :] = self.u_prev[:]
            v_nodes[0, :] = self.v_prev[:]
            w_nodes[0, :] = self.w_prev[:]
        else:
            logger.warning("not saving output to disk")

        for i in range(1, self.nt):
            start_time = time.time()

            self.u, self.v, self.w = self.newton(self.u_prev,
                                                 self.v_prev,
                                                 self.w_prev,
                                                 control["newton_iter"],
                                                 control["newton_tol"])
            self.u_prev[:], self.v_prev[:], self.w_prev[:] = self.u, self.v, self.w

            if output_file is not None:
                u_nodes[i, :] = self.u[:]
                v_nodes[i, :] = self.v[:]
                w_nodes[i, :] = self.w[:]

            self.u_nodes[i, :] = self.u[:]
            end_time = time.time()
            logger.info(f"iteration {i} took {end_time - start_time} s")

        if output_file is not None:
            output.close()
