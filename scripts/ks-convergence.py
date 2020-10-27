from fenics import *

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt


import h5py

from scipy.stats import linregress
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import cho_factor, cho_solve

set_log_level(40)


# Sub domain for periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 32 * np.pi


class KSSolve():
    def __init__(self, nx, dt, nu=1.):
        self.mesh = IntervalMesh(nx, 0., 32 * np.pi)
        P1 = FiniteElement('P', interval, 1)
        element = MixedElement([P1, P1])
        V_base = self.V_base = FunctionSpace(self.mesh, P1,
                                             constrained_domain=PeriodicBoundary())
        V = self.V = FunctionSpace(self.mesh, element,
                                   constrained_domain=PeriodicBoundary())

        # solution function
        self.u = Function(V)
        u1, u2 = split(self.u)

        # variational functions
        self.du = TrialFunction(V)
        v1, v2 = TestFunctions(V)

        # initial value
        u_init = Expression(("sin(x[0] / 16)",
                             "-sin(x[0] / 16) / pow(16, 2)"), degree=8)
        self.u_prev = interpolate(u_init, V)
        u1_prev, u2_prev = split(self.u_prev)

        # variational definition
        F1 = ((u1 - u1_prev) / dt * v1 * dx
            - nu * u1.dx(0) * v1.dx(0) * dx
            - u2.dx(0) * v1.dx(0) * dx
            + u1 * u1.dx(0) * v1 * dx)
        F2 = -u1.dx(0) * v2.dx(0) * dx - u2 * v2 * dx
        self.F = F1 + F2
        self.J = derivative(self.F, self.u, self.du)
        self.J_prev = derivative(self.F, self.u_prev, self.du)

    def get_u(self):
        return(self.u.vector()[:])

    def timestep(self):
        solve(self.F == 0, self.u, J=self.J)
        self.u_prev.assign(self.u)



reference = KSSolve(4096, 1e-3)
for i in range(100):
    reference.timestep()

u_ref, v = reference.u.split()

nx = [32, 64, 128, 256, 512]
log_h = []
log_error = []
for n in nx:
    ks = KSSolve(n, 1e-3)
    for i in range(100):
        ks.timestep()

    u, v = ks.u.split()
    log_h += [np.log(1 / n)]
    log_error += [np.log(errornorm(u_ref, u))]

lm = linregress(log_h, log_error)

plt.plot(log_h, log_error, "o--", label=f"Slope: {lm.slope:.4f}")
plt.fill([-4.5, -4., -4.], [-7.5, -7.5, -6.5], ls="--", facecolor='none', edgecolor='black')
plt.text(-3.98, -7.05, r"$\Delta y = 2$")
plt.text(-4.25, -7.73, r"$\Delta x = 1$")
plt.title("KS discretization error ($L_2$)")
plt.ylabel(r"$\log \Vert u - u_h \Vert_{L_2}$")
plt.xlabel(r"Log mesh size: $\log(h)$")
plt.legend()
plt.savefig("figures/ks-L2-convergence.pdf")
