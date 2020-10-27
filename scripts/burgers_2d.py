import argparse
import logging

from dolfin import *

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.linalg import solve as scipy_solve
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, spsolve
from scipy.spatial.distance import pdist, squareform
from scipy.stats import truncnorm


np.random.seed(27)
parameters["reorder_dofs_serial"] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_level(50)


class PeriodicBoundary(SubDomain):
    def __init__(self, L):
        SubDomain.__init__(self)
        self.L = L

    # Domain is left and bottom boundary, and not the two corners (0, 1) and (1, 0)
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], self.L)) or
                        (near(x[0], self.L) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.L) and near(x[1], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1] - self.L
        elif near(x[0], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1]
        else:   # near(x[1], self.L)
            y[0] = x[0]
            y[1] = x[1] - self.L


class BurgersSolve():
    def __init__(self, dt=0.01, nx=64, L=2.5, nu=0.01):
        self.mesh = mesh = RectangleMesh(Point(0., 0.), Point(L, L), nx, nx)
        element = FiniteElement("CG", triangle, 1)
        self.V = V = FunctionSpace(mesh,
                                   MixedElement([element, element]),
                                   constrained_domain=PeriodicBoundary(L))

        # functions
        self.w = Function(V)
        self.w_prev = Function(V)
        (u, v) = split(self.w)
        (r, s) = split(TestFunction(V))
        (u_prev, v_prev) = split(self.w_prev)

        # set initial values on the mesh
        ic = Expression(("sin(pi * (x[0] + x[1]))",
                         "sin(pi * (x[0] + x[1]))"), degree=8)
        self.w.interpolate(ic)
        self.w_prev.interpolate(ic)

        dt = Constant(dt)
        nu = Constant(nu)

        w_half = (self.w + self.w_prev) / 2
        u_half = (u + u_prev) / 2
        v_half = (v + v_prev) / 2
        self.F = ((u - u_prev) * r * dx
                  + dt * inner(w_half, grad(u_half)) * r * dx
                  + dt * nu * inner(grad(u_half), grad(r)) * dx
                  + (v - v_prev) * s * dx
                  + dt * inner(w_half, grad(v_half)) * s * dx
                  + dt * nu * inner(grad(v_half), grad(s)) * dx)
        self.J = derivative(self.F, self.w)

    def timestep(self):
        solve(self.F == 0, self.w, J=self.J)
        self.w_prev.assign(self.w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    t = 0
    thin = 1
    dt = 0.01
    nt = 500
    nt_out = len([i for i in range(nt) if i % thin == 0])
    times = [(i + 1) * dt for i in range(nt) if i % thin == 0]

    burgers = BurgersSolve(dt=dt, L=2., nx=64, nu=1 / 100)
    u_dof_indices = np.array(burgers.V.sub(0).dofmap().dofs())
    n_dof = len(u_dof_indices)
    x_u = burgers.V.tabulate_dof_coordinates()[u_dof_indices]

    output = h5py.File(args.output, "w")
    output.create_dataset("x", data=x_u)
    output.create_dataset("t", data=times)
    u_nodes = output.create_dataset("u_nodes", (nt_out, n_dof))

    i_save = 0
    for i in range(nt):
        t += dt
        logger.info(f"(deterministic) timestep {i + 1} / {nt}")
        burgers.timestep()

        if i % thin == 0:
            w = np.copy(burgers.w.vector()[:])
            u_nodes[i_save, :] = w[u_dof_indices]
            i_save += 1

    output.close()
