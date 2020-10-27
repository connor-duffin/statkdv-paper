import h5py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


simulation = "outputs/ks-posterior.h5"
output_dir = "figures/"

with h5py.File(simulation, "r") as f:
    x = f["x"][:].flatten()
    t = f["t"][:].flatten()
    x_obs = f["x_obs"][:]

    prior = f["prior"][:]
    mean = f["mean"][:]
    cov = f["cov"][:]
    y = f["y"][:]
    params = f["parameters"][:]

# profiles
fig, axs = plt.subplots(1, 4,
                        figsize=(12, 3),
                        dpi=600,
                        sharex=True,
                        constrained_layout=True)
axs.flatten()
for ax, i in zip(axs, [0, 125, 250, 375]):
    ax.plot(x, mean[i, :], label="Posterior")
    ax.fill_between(x,
                    mean[i, :] - 1.96 * np.sqrt(cov[i, :, :].diagonal()),
                    mean[i, :] + 1.96 * np.sqrt(cov[i, :, :].diagonal()),
                    alpha=0.2)
    ax.plot(x, prior[i, :], alpha=0.5, label="Prior")
    ax.plot(x_obs, y[i, :], ".", color="black", label="Data")
    ax.set_title(f"Time = {t[i]}")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("u")
        ax.legend()
fig.savefig(output_dir + "ks-profiles.pdf")

# data
fig, axs = plt.subplots(1, 4,
                        figsize=(12, 3),
                        dpi=600,
                        sharex=True,
                        constrained_layout=True)
axs.flatten()
for ax, i in zip(axs, [100, 200, 300, 400]):
    ax.plot(x_obs, y[i, :], color="black", label="Data")
    ax.set_title(f"Time = {t[i]}")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("u")
        ax.legend()
fig.savefig(output_dir + "ks-data.pdf")

# covariances
fig, axs = plt.subplots(1, 4,
                        figsize=(12, 3),
                        dpi=600,
                        sharey=True,
                        sharex=True)
axs.flatten()
for ax, i in zip(axs, [0, 100, 200, 300, 400]):
    im = ax.imshow(cov[i, :, :])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(f"Time = {t[i]:.2f}")

fig.savefig(output_dir + "ks-cov.pdf")

# heatmap
x_mesh, t_mesh = np.meshgrid(x, t)
cmin = np.min(prior)
cmax = np.max(prior)
fig, axs = plt.subplots(1, 2,
                        figsize=(8, 3),
                        dpi=600,
                        sharex=True,
                        sharey=True,
                        constrained_layout=True)
axs[0].pcolormesh(x_mesh, t_mesh, prior, cmap="coolwarm", vmin=cmin, vmax=cmax)
axs[0].set_title("Prior mean")
axs[0].set_xlabel("x")
axs[0].set_ylabel("t")
im = axs[1].pcolormesh(x_mesh, t_mesh, mean, cmap="coolwarm", vmin=cmin, vmax=cmax)
plt.colorbar(im, ax=axs[1])
axs[1].set_title("Post mean")
axs[1].set_xlabel("x")
fig.savefig(output_dir + "ks-mesh.png")

# parameters
labels = [r"$\tau_n$", "$\ell_n$", "$\sigma_n$"]
fig, axs = plt.subplots(1, 3,
                        dpi=600,
                        figsize=(9, 3),
                        sharex=True,
                        constrained_layout=True)
for ax, i in zip(axs, range(3)):
    ax.plot(t, params[:, i])
    ax.set_title(labels[i])
    ax.set_yscale("log")
    ax.set_xlabel("t")
    if i == 2:
        axs[i].plot(t, np.repeat(0.05, len(t)), "--", label="True parameter")

fig.savefig(output_dir + "ks-parameters.pdf")
