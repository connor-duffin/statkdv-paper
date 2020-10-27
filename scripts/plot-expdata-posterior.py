import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from seaborn import set_palette

from common.tankmeta import load_data, add_soliton_operator, add_solitons
from common.figset import set_plot_style


parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()


def load_solution(filename):
    with h5py.File(filename, "r") as f:
        nodes = f["model/u_nodes"][:]
        x_grid = f["model/x_grid"][:]
        t_grid = f["model/t_grid"][:]

    u_obs, x_obs = add_solitons(nodes, x_grid)
    return(u_obs, x_obs, t_grid)


with h5py.File(args.simulation, "r") as f:
    x_grid = f["model/x_grid"][:]
    t_grid = f["model/t_grid"][:]

    mean_obs = f["model/mean_obs"][:]
    covariance_obs = f["model/covariance_obs"][:]

    mean = f["model/mean"][:]
    cov = f["model/covariance"][:]

    x_obs = f["data/x_grid"][:]
    y_obs = f["data/y"][:]

data = load_data()
u_sim, x_sim, t_sim = load_solution("outputs/tank-deterministic.h5")
idx = np.searchsorted(x_sim, np.array([1.47, 3.02, 4.57]))

H = add_soliton_operator(x_grid)
x_grid = x_grid[x_grid <= 6.]

set_plot_style()
colors = ['#fe9929', '#7bccc4', '#0868ac']  # orange, turquoise, blue
set_palette(colors)
fig, axs = plt.subplots(3, 2,
                        dpi=400,
                        constrained_layout=True,
                        figsize=(18 / 2.54, 11 / 2.54),
                        gridspec_kw={"width_ratios": [2, 1]},
                        sharex="col")
fig.set_constrained_layout_pads(wspace=0.1)


axs[0, 0].plot(data.time.values, data.wg01.values, label="Data")
axs[0, 0].plot(t_sim, u_sim[:, idx[0]], label="eKdV")
axs[0, 0].plot(t_grid[1:], mean_obs[1:, 0], label="Posterior")
axs[0, 0].fill_between(t_grid[1:],
                       mean_obs[1:, 0] - 1.96 * np.sqrt(covariance_obs[1:, 0, 0]),
                       mean_obs[1:, 0] + 1.96 * np.sqrt(covariance_obs[1:, 0, 0]),
                       color=colors[2],
                       alpha=0.5)
axs[0, 0].set_title("Posterior at wave gauge locations", fontsize="large")
axs[0, 0].text(-0.05, 1.10, "A",
               fontsize=12,
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 0].transAxes)
axs[0, 0].legend()

axs[1, 0].plot(data.time.values, data.wg02.values)
axs[1, 0].plot(t_sim, u_sim[:, idx[1]], label="eKdV")
axs[1, 0].plot(t_grid[1:], mean_obs[1:, 1])
axs[1, 0].fill_between(t_grid[1:],
                       mean_obs[1:, 1] - 1.96 * np.sqrt(covariance_obs[1:, 1, 1]),
                       mean_obs[1:, 1] + 1.96 * np.sqrt(covariance_obs[1:, 1, 1]),
                       color=colors[2],
                       alpha=0.5)
axs[1, 0].set_ylabel("displacement (m)")

axs[2, 0].plot(data.time.values, data.wg03.values)
axs[2, 0].plot(t_sim, u_sim[:, idx[2]], label="eKdV")
axs[2, 0].plot(t_grid[1:], mean_obs[1:, 2])
axs[2, 0].fill_between(t_grid[1:],
                       mean_obs[1:, 2] - 1.96 * np.sqrt(covariance_obs[1:, 2, 2]),
                       mean_obs[1:, 2] + 1.96 * np.sqrt(covariance_obs[1:, 2, 2]),
                       color=colors[2],
                       alpha=0.5)
axs[2, 0].set_xlabel("t (s)")

idx = [250, 500, 750]
for i in range(3):
    index = idx[i]
    mean_curr = mean[index, :]
    cov_curr = cov[index, :, :]

    axs[i, 1].plot(x_grid, H @ mean_curr, color=colors[2], label="Posterior")
    axs[i, 1].fill_between(x_grid,
                           H @ mean_curr - 1.96 * np.sqrt(np.abs(H @ cov_curr @ H.T).diagonal()),
                           H @ mean_curr + 1.96 * np.sqrt(np.abs(H @ cov_curr @ H.T).diagonal()),
                           alpha=0.25,
                           color=colors[2])
    axs[i, 1].plot(x_obs, y_obs[index, :], ".", color=colors[0], label="Data")
    # axs[i, 1].set_ylim(-0.03, 0.03)
    axs[i, 1].text(0.85, 0.05,
                   fr"t = {t_grid[index]}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=axs[i, 1].transAxes)
    if i == 0:
        axs[i, 1].legend()
        axs[i, 1].set_title("Posterior profile", fontsize="large")
        axs[i, 1].text(-0.1, 1.10, "B",
                       fontsize=12,
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=axs[i, 1].transAxes)

    if i == 2:
        axs[i, 1].set_xlabel("x (m)")


fig.savefig(args.output)
