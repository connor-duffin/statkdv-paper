import argparse
import logging
import numpy as np
import h5py
import sympy

from statkdv.condapprox import KdVApprox
from scipy.sparse import csr_matrix


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

settings = {
    "x_start": 0,
    "x_end": 20,
    "t_start": 0,
    "t_end": 50,
    "nx": 400,
    "nt": 2001,
    "nx_out": 20,
    "nx_spectral": 1024,
    "nt_spectral": 50001,
    "n_ensemble": 400
}
parameters = {
    "alpha": 1,
    "beta": 1e-2,
    "c": 0,
    "scale": 0.1,
    "length": 1
}
initial_conditions = "-0.3 * (1 / cosh(x - 15))**2"
control = {
    "newton_iter": 50,
    "newton_tol": 1e-8,
    "thin": 10,
    "noisefree": false,
    "save_ensemble": false
}

if args.data is not None:
    with h5py.File(args.data, "r") as f:
        data = {}
        for key in f["data"].keys():
            data[key] = f["data"][key][:]
    # HACK: convert to csr matrix here
    data["H"] = csr_matrix(data["H"])
else:
    data = None

kdv = KdVApprox(settings, parameters)
kdv.set_initial_condition(initial_conditions, sympy.symbols("x"))

if data is None: kdv.set_cov_square_exp(parameters["scale"],
                                        parameters["length"])

kdv.solve_fem(data=data, control=control, output_file=args.output)
