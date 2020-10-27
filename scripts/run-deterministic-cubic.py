import argparse
import logging

import sympy

from statkdv.deterministic import KdV


# some settings
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
args = parser.parse_args()

if args.output is None:
    logging.warning("not saving output to disk")

settings = {"x_start": 0, "x_end": 20, "nx": 400,
            "t_start": 0, "t_end": 50, "nt": 2001}
parameters = {"alpha": 1, "beta": 1e-2, "c": 0}
initial_conditions = "-0.3 * (1 / cosh(x - 15))**2"
control = {"newton_iter": 50, "newton_tol": 1e-8, "thin": 10}

kdv = KdV(settings, parameters)
kdv.set_initial_condition(initial_conditions, sympy.symbols("x"))
kdv.solve_fem(output_file=args.output)
