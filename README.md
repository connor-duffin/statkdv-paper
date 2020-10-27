# Reproduce results in the statKdV paper

This repo contains the code to reproduce the analysis in the statKdV paper.

Code is given to run the statistical finite element method on simulated data
and on experimental data. Simulated data is generated with `dedalus` and
experimental data is located in the `data/` directory. Code is also given to
run the Burgers and KS examples with `fenics`.

To run this code you will need

* `statkdv` (included: `pip install -e statkdv`)
* `statfenics` (included: `pip install -e statfenics`)
* `fenics`
* `dedalus`
* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `h5py`
* `mpi4py`
* `sympy`

Details for the installation of `fenics` and `dedalus` can be found on their
respective websites.

In order for compute times to be a shorter, some of the resolutions aren't as
high as in the paper. Editing the appropriate files in `inputs/` should hopefully
set these up appropriately.

Any bugs/questions/comments, please don't hesitate to email me at
`connor.duffin@research.uwa.edu.au`
