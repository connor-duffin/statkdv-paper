# statkdv: solve the KdV equation statistically
Connor Duffin

Solve the KdV equation using the statistical finite element method. Requires

* `numpy`
* `scipy`
* `h5py`
* `mpi4py`

Includes methods to solve KdV deterministically, generate prior distributions,
and generate posterior distributions (conditioned on data).

To run tests `cd` into this directory and run `python3 -m unittest`.
