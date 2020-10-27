import argparse
import logging

import numpy as np
import sympy

import matplotlib.pyplot as plt
from scipy.stats import linregress

from statkdv import utils
from statkdv.deterministic import KdV


logging.basicConfig(level=logging.INFO)


def sech(x):
    return(1 / np.cosh(x))


def compute_error(simulation):
    x_grid = simulation.x_grid
    t_grid = simulation.t_grid
    u_nodes = simulation.u_nodes

    time_index = -2  # last point of the sim
    u = utils.Interpolator(u_nodes[time_index, :], x_grid)
    log_error = np.log(np.sqrt(utils.gl_grid(lambda x: (u.evaluate(x)
                                                        - 3 * 0.5 * sech(1 / 2 * np.sqrt(0.5 / 1e-3) * (x - 1 - 0.5 * t_grid[time_index]))**2)**2,
                                             x_grid,
                                             n=20)))
    return(log_error)


# run for 5 discretization levels (200 -> 1000)
# spinup 100 timesteps, dt = 1e-3
# settings
settings = {"x_start": 0,
            "x_end": 3,
            "t_start": 0,
            "t_end": 1e-2,
            "nt": 101}

# inits
initial_condition = "3 * 0.5 * (1 / cosh(1 / 2 * sqrt(0.5 / 1e-3) * (x - 1)))**2"

# parameters
parameters = {"alpha": 1., "beta": 1e-3, "c": 0.}

nx = [200, 400, 600, 800, 1000, 2000]
log_h = []
log_error = []

for n in nx:
    settings["nx"] = n
    kdv = KdV(settings, parameters)
    kdv.set_initial_condition(initial_condition, sympy.symbols("x"))
    kdv.solve_fem()
    log_h += [np.log(kdv.dx)]
    log_error += [compute_error(kdv)]

lm = linregress(log_h, log_error)
print(lm)

plt.plot(log_h, log_error, "o--", label=f"Slope: {lm.slope:.4f}")
plt.fill([-5., -4.5, -4.5], [-9.5, -9.5, -8.5], ls="--", facecolor='none', edgecolor='black')
plt.text(-4.48, -9.0, r"$\Delta y = 2$")
plt.text(-4.75, -9.7, r"$\Delta x = 1$")
plt.title("KdV discretization error ($L_2$)")
plt.ylabel(r"$\log \Vert u - u_h \Vert_{L_2}$")
plt.xlabel(r"Log mesh size: $\log(h)$")
plt.legend()
plt.savefig("figures/kdv-L2-convergence.pdf")
