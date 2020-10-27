import argparse
import logging
import numpy as np
import h5py
import sympy

from mpi4py import MPI
from scipy.sparse import csr_matrix
from statkdv.condmc import KdVEnsemble


def none_or_str(string):
    if string == "None":
        return None
    else:
        return string


np.random.seed(27)
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=none_or_str, default=None)
parser.add_argument("--output", type=str)
args = parser.parse_args()

settings = {"x_start": 0, "x_end": 20,
            "t_start": 0, "t_end": 50,
            "nx": 400, "nt": 2001,
            "nx_out": 20, "n_ensemble": 400}
parameters = {"alpha": 1, "beta": 1e-2, "c": 0}
initial_conditions = "-0.3 * (1 / cosh(x - 15))**2"
control = {"newton_iter": 50,
           "newton_tol": 1e-8,
           "thin": 10,
           "noisefree": False,
           "save_ensemble": False}

if args.data is not None:
    with h5py.File(args.data, "r") as f:
        data = {}
        for key in f["data"].keys():
            data[key] = f["data"][key][:]
        # HACK: convert to csr matrix here
        data["H"] = csr_matrix(data["H"])
else:
    data = None

kdv = KdVEnsemble(settings, parameters)
kdv.set_initial_condition(initial_conditions, sympy.symbols("x"))

# SI values: 0.0025, 0.00125, 0.00025
# just plug and play for whichever you want
scale, length = 0.0025, 1.
logging.info(f"scale and length set to {scale, length}")
if data is None: kdv.set_cov_square_exp(scale, length)
kdv.solve_ensemble(data=data, control=control, output_file=args.output, comm=MPI.COMM_WORLD)
