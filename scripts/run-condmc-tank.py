""" Compute the prior/posterior using statkdv, for the tank

    Arguments
    ---------
    --output: string
        output file location
    --prior: toggle
        toggles whether a prior or posterior measure will be computed.
"""
import argparse
import logging

import numpy as np
import sympy

from mpi4py import MPI
from statkdv.condmc import KdVEnsemble
from statkdv.utils import build_interpolation_matrix
from common import tankmeta as tm


np.random.seed(27)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
parser.add_argument("--prior", action="store_true")
args = parser.parse_args()


logging.info(f"Initializing with settings: {tm.statkdv_settings}")
logging.info(f"Initializing with parameters: {tm.parameters}")
logging.info(f"Initializing with priors: {tm.priors}")
kdv = KdVEnsemble(tm.statkdv_settings, tm.parameters)
kdv.set_sigma_y(np.sqrt(1.3588e-8))
# kdv.n_eigen = 200
kdv.nugget = 1e-10

# cfl: dx / dt
cfl = (
    (tm.statkdv_settings["x_end"] / tm.statkdv_settings["nx"])
    / (tm.statkdv_settings["t_end"] / tm.statkdv_settings["nt"])
)
logging.info(f"CFL satisfied: {tm.parameters['c'] <= cfl}")


# don't read data if not conditioning
if args.prior:
    logging.info("Computing prior measure")
    data = None
else:
    logging.info("Computing posterior measure")
    wave = tm.load_data()
    # snap obs to time grid
    wave_processed = tm.interpolate_data_time(wave, kdv.t_grid)

    # locations from Horn thesis
    x_obs = np.array([1.47, 3.02, 4.57])
    H = (build_interpolation_matrix(kdv.x_grid, x_obs)
        + build_interpolation_matrix(kdv.x_grid, 12. - x_obs))

    # locations to project to
    x_project = np.linspace(0.06, 6, num=99, endpoint=False)
    H_project = (build_interpolation_matrix(kdv.x_grid, x_project)
                 + build_interpolation_matrix(kdv.x_grid, 12. - x_project))

    data = {"H": H,
            "y": wave_processed.iloc[:, 1:4].values,
            "t_grid": wave_processed.time.values,
            "x_grid": x_obs,
            "H_project": H_project,
            "x_project": x_project}

tm.control["project_data_forward"] = True
logging.info("Starting simulation")
kdv.set_initial_condition(tm.statkdv_initial_condition, sympy.symbols("x"))
kdv.solve_ensemble(data=data,
                   control=tm.control,
                   priors=tm.priors,
                   output_file=args.output,
                   comm=MPI.COMM_WORLD)
