import argparse
import json
import time

import numpy as np
import h5py
import sympy

from dedalus import public as de
from statkdv.utils import build_interpolation_matrix

import logging
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--output", type=str)
args = parser.parse_args()

# settings etc
settings = {"x_start": 0, "x_end": 20, "nx": 400,
            "t_start": 0, "t_end": 50, "nt": 2001,
            "nx_out": 20,
            "nx_spectral": 1024,
            "nt_spectral": 2001}
parameters = {"alpha": 1, "beta": 1e-2, "epsilon": 20, "c": 0}
initial_condition = "-0.3 * (1 / cosh(x - 15))**2"

# process initial conditions
x_sym = sympy.symbols("x")
u_init = sympy.sympify(initial_condition)
f_u_init = sympy.lambdify(x_sym, u_init, "numpy")

nt_out = settings["nt"] - 1
dt = (settings["t_end"] - settings["t_start"]) / (settings["nt_spectral"] - 1)

x_basis = de.Fourier("x",
                     settings["nx_spectral"],
                     interval=(settings["x_start"], settings["x_end"]),
                     dealias=1)
domain = de.Domain([x_basis], np.float64)

# define the problem
problem = de.IVP(domain, variables=["u", "ux", "uxx"])
problem.parameters["alpha"] = parameters["alpha"]
problem.parameters["beta"] = parameters["beta"]
problem.parameters["c"] = parameters["c"]
problem.parameters["epsilon"] = parameters["epsilon"]
problem.add_equation("dt(u) + beta * dx(uxx) + c * dx(u) = -alpha * u * ux - epsilon * u**3 * ux")
problem.add_equation("ux - dx(u) = 0")
problem.add_equation("uxx - dx(ux) = 0")

solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_wall_time = 300
solver.stop_iteration = settings["nt_spectral"] - 1

x = domain.grid(0)
u = solver.state["u"]
ux = solver.state["ux"]
uxx = solver.state["uxx"]

u["g"] = f_u_init(x)
u.differentiate(0, out=ux)
ux.differentiate(0, out=uxx)

step = settings["nx_spectral"] // settings["nx_out"]
u_list = [np.copy(u["g"][:])]
t_list = [solver.sim_time]

logger.info("Starting loop")
start_time = time.time()

while solver.proceed:
    solver.step(dt)
    if solver.iteration % int(np.rint(settings["nt_spectral"] / nt_out)) == 0:
        u_list.append(np.copy(u["g"][:]))
        t_list.append(solver.sim_time)
    if solver.iteration % 1000 == 0:
        logger.info("Iteration: %i, Time: %e, dt: %e" % (solver.iteration, solver.sim_time, dt))

end_time = time.time()
logger.info("Iterations: %i" % solver.iteration)
logger.info("Sim end time: %f" % solver.sim_time)
logger.info("Run time: %.2f sec" % (end_time - start_time))

t_grid = np.asarray(t_list)
x_grid = x[::step]
x_solve = np.linspace(settings["x_start"],
                    settings["x_end"],
                    settings["nx"],
                    endpoint=False)
u_grid = np.array(u_list)

with h5py.File(args.output, "w") as f:
    if args.save_model:
        model = f.create_group("model")
        model.create_dataset("x_grid", data=x)
        model.create_dataset("t_grid", data=t_grid)
        model.create_dataset("u_nodes", data=u_grid)
    else:
        u_grid = u_grid[:, ::step]
        u_grid += np.random.normal(scale=0.001, size=u_grid.shape)

        data = f.create_group("data")
        data.create_dataset("t_grid", data=t_grid)
        data.create_dataset("x_grid", data=x_grid[x_grid < x_solve[-1]])
        data.create_dataset("H", data=build_interpolation_matrix(x_grid=x_solve,
                                                                x_obs=x_grid[x_grid < x_solve[-1]]).todense())
        data.create_dataset("y", data=u_grid[:, x_grid < x_solve[-1]])

