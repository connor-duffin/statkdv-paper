import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from common.figset import set_plot_style


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
args = parser.parse_args()

# predefine the input files
prior = "outputs/cubic-prior-enkf-0.0025.h5"
data = "outputs/cubic-reference.h5"
posterior = "outputs/cubic-posterior-enkf.h5"

# HACK: go from 1: to match dimensions
with h5py.File(prior, "r") as f:
    x_prior = f["model/x_grid"][:]
    t_prior = f["model/t_grid"][:]
    prior = f["model/mean"][:, 0:len(x_prior)]

with h5py.File(data, "r") as f:
    x_data = f["model/x_grid"][:]
    t_data = f["model/t_grid"][:]
    data = f["model/u_nodes"][:, :]

with h5py.File(posterior, "r") as f:
    x_post = f["model/x_grid"][:]
    t_post = f["model/t_grid"][:]
    post = f["model/mean"][:, 0:len(x_post)]


set_plot_style()
fig, axs = plt.subplots(1, 3,
                        constrained_layout=True,
                        dpi=400,
                        figsize=(15 / 2.54, 5 / 2.54),
                        sharex=True,
                        sharey=True)

# set color scales
pcolor_settings = {"vmax": np.max([prior.max(), data.max(), post.max()]),
                   "vmin": np.min([prior.min(), data.min(), post.min()]),
                   "cmap": "coolwarm"}

xmesh, tmesh = np.meshgrid(x_prior, t_prior)
im = axs[0].pcolormesh(xmesh, tmesh, prior, **pcolor_settings)
axs[0].set_xlabel("x")
axs[0].set_ylabel("t (s)")
axs[0].set_title("Prior mean:\n" + r"$\mathrm{E}[u^n | \theta, \Lambda]$", fontsize="large")
axs[0].text(-0.05, 1.10, "C",
            fontsize=12,
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[0].transAxes)

xmesh, tmesh = np.meshgrid(x_data, t_data)
im = axs[1].pcolormesh(xmesh, tmesh, data, **pcolor_settings)
axs[1].set_xlabel("x")
axs[1].set_title("DGP", fontsize="large")

xmesh, tmesh = np.meshgrid(x_post, t_post)
im = axs[2].pcolormesh(xmesh, tmesh, post, **pcolor_settings)
cbar = plt.colorbar(im, ax=axs[2], shrink=1)
cbar.ax.set_ylabel("Wave amplitude")
axs[2].set_xlabel("x")
axs[2].set_title("Posterior mean:\n" + r"$\mathrm{E}[u^n | y_{1:n}, \theta_{1:n}, \sigma_{1:n}, \Lambda]$", fontsize="large")
fig.savefig(args.output)
