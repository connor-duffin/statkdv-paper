import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from seaborn import set_palette, color_palette

from common import figset


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
args = parser.parse_args()


priors = ["outputs/cubic-prior-enkf-0.0025.h5",
          "outputs/cubic-prior-enkf-0.00125.h5",
          "outputs/cubic-prior-enkf-0.00025.h5"]
solution = "outputs/cubic-deterministic.h5"

with h5py.File(solution, "r") as f:
    x_fem = f["model/x_grid"][:]
    # HACK: stride through 1::10 to match dimensions
    u_fem = f["model/u_nodes"][1::10]

with h5py.File(priors[0], "r") as f:
    x_grid = f["model/x_grid"][:]
    t_grid = f["model/t_grid"][:]

idx = np.round(np.linspace(0, len(t_grid) - 1, 4, endpoint=False)).astype(int)

figset.set_plot_style()
colors = ["#0868ac", "#43a2ca", "#7bccc4", "#fe9929"]
colors.reverse()
set_palette(colors)

fig, axs = plt.subplots(2, 2,
                        constrained_layout=True,
                        dpi=300,
                        figsize=(21 / 2.54, 8 / 2.54),
                        sharex=True,
                        sharey=True)
axs = axs.flatten()
# HACK: add subplot to label all axes, hide ticks
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none",
                top=False,
                bottom=False,
                left=False,
                right=False)
plt.axis("off")

labels = [r"$\tau^2 = 0.0025^2$",
          r"$\tau^2 = 0.00125^2$",
          r"$\tau^2 = 0.00025^2$"]
for i in range(4):
    alpha = 0.2
    axs[i].plot(x_fem, u_fem[idx[i], :], '.', label="FEM soln.")
    for label, prior in zip(labels, priors):
        with h5py.File(prior, "r") as f:
            mean = f["model/mean"][idx[i], :]
            cov = f["model/covariance"][idx[i], :, :]

        axs[i].plot(x_grid, mean[0:len(x_grid)], label=label)
        axs[i].fill_between(
            x_grid,
            mean[0:len(x_grid)] - 1.96 * np.sqrt(np.abs(cov[0:len(x_grid), 0:len(x_grid)].diagonal())),
            mean[0:len(x_grid)] + 1.96 * np.sqrt(np.abs(cov[0:len(x_grid), 0:len(x_grid)].diagonal())),
            color=colors[2],
            alpha=alpha
        )
        alpha += 0.15

    axs[i].text(0.9, 0.075,
                fr"t = {t_grid[idx[i]]:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[i].transAxes)

    if i == 0:
        axs[i].legend(loc="upper left")

axs[0].set_ylabel("u")
axs[2].set_ylabel("u")
axs[2].set_xlabel("x")
axs[3].set_xlabel("x")
plt.xlabel("$x$")
plt.ylabel("displacement")
fig.savefig(args.output)
