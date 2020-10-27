import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from common import figset


parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with h5py.File(args.simulation, "r") as f:
    t_grid = f["model/t_grid"][:]
    parameters = f["model/parameters"][:]

if np.shape(parameters)[1] == 2:
    labels = (r"$\tau_n$", r"$\ell_n$")
else:
    labels = (r"$\tau_n$", r"$\ell_n$", r"$\sigma_n$")

print(np.mean(parameters[-100:, 2]))
plt.hist(parameters[-100:, 2], bins=50)
plt.show()
plt.close()

figset.set_plot_style()
fig, axs = plt.subplots(1, len(labels),
                        constrained_layout=True,
                        dpi=600,
                        figsize=(16 / 2.54, 4 / 2.54),
                        sharex=True)

# HACK: add for global labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none",
                top=False,
                bottom=False,
                left=False,
                right=False)
plt.axis("off")

axs = axs.flatten()
for ax, column, label in zip(axs, parameters.T, labels):
    ax.plot(t_grid[1:], np.abs(column), color="#0868ac")
    ax.set_yscale("log")
    ax.set_ylabel(label)
    ax.set_xlabel("t (s)")

# plot the actual noise value
axs[2].plot(t_grid[1:], np.repeat(0.001, len(t_grid[1:])), "--", color="#7bccc4")
axs[0].text(-0.05, 1.1, "D",
            fontsize=12,
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[0].transAxes)
plt.title("Estimated parameters", fontsize="large")
fig.savefig(args.output)
