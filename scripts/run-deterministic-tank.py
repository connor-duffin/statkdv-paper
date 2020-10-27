import argparse
import logging

import sympy

from statkdv.deterministic import KdV
from common.tankmeta import statkdv_settings, statkdv_initial_condition, parameters


# some settings
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
args = parser.parse_args()

if args.output is None:
    logging.warning("not saving output to disk")

# sanity check cfl: dx / dt
cfl = ((statkdv_settings["x_end"] / statkdv_settings["nx"])
       / (statkdv_settings["t_end"] / statkdv_settings["nt"]))
logging.info(f"CFL: c <= dx / dt: {parameters['c']} <= {cfl}")

parameters["nu"] = 0.003
kdv = KdV(statkdv_settings, parameters)
kdv.set_initial_condition(statkdv_initial_condition, sympy.symbols("x"))
kdv.solve_fem(output_file=args.output)
