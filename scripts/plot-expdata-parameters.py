import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from seaborn import set_palette

from common.figset import set_plot_style


parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with h5py.File(args.simulation, "r") as f:
    t_grid = f["model/t_grid"][:]
    parameters = f["model/parameters"][:, :2]

labels = (r"$\tau_{n}$", r"$\ell_{n}$")

set_plot_style()
colors = ['#fe9929', '#7bccc4', '#0868ac']  # orange, turquoise, blue
set_palette(colors)
fig, axs = plt.subplots(1, len(labels),
                        constrained_layout=True,
                        dpi=400,
                        figsize=(12 / 2.54, 4 / 2.54),
                        sharex=True)

# HACK: add plot for global  labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none",
                top=False,
                bottom=False,
                left=False,
                right=False)
plt.axis("off")

for ax, column, label in zip(axs, parameters.T, labels):
    ax.plot(t_grid[1:], np.abs(column[:]), color=colors[2], linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel(label)
    ax.set_xlabel("t (s)")

axs[0].text(-0.05, 1.10, "C",
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0].transAxes)
plt.title("Estimated parameters", fontsize="large")
fig.savefig(args.output)
