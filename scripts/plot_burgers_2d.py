import h5py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# read in data etc
dgp = h5py.File("outputs/burgers-dgp.h5", "r")
post = h5py.File("outputs/burgers-posterior.h5", "r")

x_dgp = dgp["x"][:]
t_dgp = dgp["t"][:]
u_dgp = dgp["u_nodes"][:]

u_prior = post["prior_mean"][:]
u_post = post["post_mean"][:]
var = post["post_var"][:]
x = post["x"][:]
t = post["t"][:]
y = post["y"]
x_obs = post["x_obs"]
params = post["parameters"][:]
idx = np.linspace(0, len(t), 4, endpoint=False).astype(int)

# RMSE
error_prior = np.sqrt(np.sum((u_prior - u_dgp)**2, 1)) / np.sqrt(np.sum(u_dgp**2, 1))
error_post = np.sqrt(np.sum((u_post - u_dgp)**2, 1)) / np.sqrt(np.sum(u_dgp**2, 1))

plt.plot(t, error_prior, label="Prior RMSE")
plt.plot(t, error_post, label="Posterior RMSE")
plt.xlabel("t")
plt.ylabel(r"$\Vert u - u_{DGP} \Vert_2 / \Vert u_{DGP} \Vert_2$")
plt.title("Relative $L^2$ error")
plt.legend()
plt.savefig("figures/burgers-rmse.pdf")
plt.close()

# post surfaces w/data
fig = plt.figure(dpi=600, figsize=(12, 3))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    c1 = ax.plot_trisurf(x[:, 0], x[:, 1], u_post[idx[i], :], label="Posterior mean")
    ax.scatter(x_obs[:, 0], x_obs[:, 1], y[idx[i], :], color="orange", label="Data")

    if i == 0:
        c1._facecolors2d = c1._facecolors3d
        c1._edgecolors2d = c1._edgecolors3d
        fig.legend(loc="center left")

    ax.view_init(azim=-30, elev=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Time t = {t[idx[i]]:.2f}")

fig.savefig("figures/burgers-surfaces.png")

# heatmaps of the DGP solution for 4 times
fig, axs = plt.subplots(1, 4,
                        figsize=(15, 3),
                        dpi=600,
                        sharex=True,
                        sharey=True,
                        constrained_layout=True)
idx_dgp = np.linspace(0, len(t_dgp), 4, endpoint=False).astype(int)
for i, ax in enumerate(axs):
    im = ax.tricontourf(x_dgp[:, 0], x_dgp[:, 1], u_dgp[idx_dgp[i], :])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"DGP, t = {t[idx[i]]:.2f}")

axs[0].set_ylabel("y")
axs[1].set_xlabel("x")
fig.savefig("figures/burgers-dgp-means.png")
plt.close()

# posterior plots
# prior data
fig = plt.figure(dpi=600, figsize=(12, 4), constrained_layout=True)
for i in range(len(idx)):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    ax.plot_trisurf(x[:, 0], x[:, 1], u_prior[idx[i], :])
    ax.scatter(x_obs[:, 0], x_obs[:, 1], y[idx[i], :], color="orange")
    ax.view_init(azim=-45, elev=0)

fig.savefig("figures/burgers-prior-data.png")

# post data
fig = plt.figure(dpi=600, figsize=(12, 4), constrained_layout=True)
for i in range(len(idx)):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    ax.plot_trisurf(x[:, 0], x[:, 1], u_post[idx[i], :])
    ax.scatter(x_obs[:, 0], x_obs[:, 1], y[idx[i], :], color="orange")
    ax.view_init(azim=-45, elev=0)

fig.savefig("figures/burgers-post-data.png")

# post DGP
fig = plt.figure(dpi=600, figsize=(12, 4), constrained_layout=True)
for i in range(len(idx)):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    ax.plot_trisurf(x[:, 0], x[:, 1], u_dgp[idx[i], :])
    ax.plot_trisurf(x[:, 0], x[:, 1], u_post[idx[i], :], alpha=0.5, color="orange")
    ax.view_init(azim=-30, elev=0)

fig.savefig("figures/burgers-post-dgp.png")

# prior DGP
fig = plt.figure(dpi=600, figsize=(12, 4), constrained_layout=True)
for i in range(len(idx)):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    ax.plot_trisurf(x[:, 0], x[:, 1], u_dgp[idx[i], :])
    ax.plot_trisurf(x[:, 0], x[:, 1], u_prior[idx[i], :], alpha=0.5, color="orange")
    ax.view_init(azim=-30, elev=0)

fig.savefig("figures/burgers-prior-dgp.png")

# mesh w/observation locations on top
fig, ax = plt.subplots(1, 1, dpi=600, constrained_layout=True)
ax.triplot(x[:, 0], x[:, 1])
ax.scatter(x_obs[:, 0], x_obs[:, 1], color="orange", zorder=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("FEM mesh and observation locations")
plt.savefig("figures/burgers-mesh-obs-locations.png")

# post mean heatmap
fig, axs = plt.subplots(1, 4,
                        figsize=(15, 3),
                        dpi=600,
                        sharex=True,
                        sharey=True,
                        constrained_layout=True)
for i, ax in enumerate(axs):
    im = ax.tricontourf(x[:, 0], x[:, 1], u_post[idx[i], :])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Posterior mean, t = {t[idx[i]]:.2f}")

axs[0].set_ylabel("y")
axs[2].set_xlabel("x")
fig.savefig("figures/burgers-post-means.png")

# post variance heatmap
fig, axs = plt.subplots(1, 4,
                        figsize=(15, 3),
                        dpi=600,
                        sharex=True,
                        sharey=True,
                        constrained_layout=True)
for i, ax in enumerate(axs):
    im = ax.tricontourf(x[:, 0], x[:, 1], var[idx[i], :])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Posterior var., t = {t[idx[i]]:.2f}")

axs[0].set_ylabel("y")
axs[2].set_xlabel("x")
fig.savefig("figures/burgers-post-vars.png")

# parameter estimates
labels = [r"$\tau_n$", "$\sigma_n$"]
fig, axs = plt.subplots(1, 2, dpi=600, figsize=(5, 2),
                        constrained_layout=True)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.plot(t, params[:, i])
    ax.set_ylabel(labels[i])
    ax.set_yscale("log")
    ax.set_xlabel("time (nondimensional)")

axs[1].plot(t, np.repeat(0.01, len(t)), "--", color="orange")
fig.savefig("figures/burgers-parameters.png")
