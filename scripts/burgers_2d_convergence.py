from fenics import *

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import h5py

from scipy.stats import linregress


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


def compute_error(n):
    L = 2.
    mesh = mesh = RectangleMesh(Point(0., 0.), Point(L, L), n, n)
    element = FiniteElement("CG", triangle, 1)
    V = V = FunctionSpace(mesh,
                          MixedElement([element, element]),
                          constrained_domain=PeriodicBoundary(L))

    # functions
    w = Function(V)
    w_prev = Function(V)
    (u, v) = split(w)
    (r, s) = split(TestFunction(V))
    (u_prev, v_prev) = split(w_prev)

    # set initial values on the mesh
    ic = Expression(("sin(pi * (x[0] + x[1]))",
                     "sin(pi * (x[0] + x[1]))"), degree=8)

    w.interpolate(ic)
    w_prev.interpolate(ic)

    h = 1 / n
    return(h, errornorm(ic, w, "L2"))


nx = [2**i for i in range(4, 9)]
log_h = []
log_error = []
for i, n in enumerate(nx):
    print(f"Solving for n = {n} (h = {1 / n:.8f})")
    h, error = compute_error(n)
    log_h += [np.log(h)]
    log_error += [np.log(error)]

    if i > 0:
        print((log_error[i] - log_error[i - 1]) / (log_h[i] - log_h[i - 1]))

lm = linregress(log_h, log_error)
print("All done, lm result:", lm)
print("log errors: ", log_error)
fig, ax = plt.subplots(1, 1, dpi=600, constrained_layout=True)
ax.plot(log_h, log_error, "o--", label=f"Slope: {lm.slope:.4f}")
ax.fill([-4.0, -3.5, -3.5], [-6., -6., -5.], ls="--", facecolor='none', edgecolor='black')
ax.text(-3.75, -6.18, r"$\Delta x = 1$")
ax.text(-3.49, -5.5, r"$\Delta y = 2$")
ax.set_title(r"Burgers equation discretization error ($L_2$)")
ax.set_ylabel(r"$\log \Vert u - u_h \Vert_{L_2}$")
ax.set_xlabel(r"Log mesh size: $\log(h)$")
ax.legend()
plt.savefig("figures/burgers-convergence.png")
