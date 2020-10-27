import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from common.figset import set_plot_style


def cm_to_inch(*args):
    output = [x / 2.54 for x in args]
    return(output)


def uncertainty_bounds(ax, x, mean, cov):
    ax.fill_between(x,
                    mean - 1.96 * np.sqrt(np.abs(cov.diagonal())),
                    mean + 1.96 * np.sqrt(np.abs(cov.diagonal())),
                    alpha=0.2,
                    color=colors[1])


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
args = parser.parse_args()

fig_settings = {"constrained_layout": True,
                "figsize": cm_to_inch(18, 6),
                "sharex": True,
                "sharey": True}

prior = "outputs/cubic-prior-enkf-0.0025.h5"
reference = "outputs/cubic-reference.h5"
posterior = "outputs/cubic-posterior-enkf.h5"

with h5py.File(reference, "r") as f:
    x_ref = f["model/x_grid"][:]
    # HACK: stride through 1::10 to match dimensions
    u_ref = f["model/u_nodes"][1::10]

with h5py.File(prior, "r") as f:
    x_prior = f["model/x_grid"][:]
    t_prior = f["model/t_grid"][:]

    prior_mean = f["model/mean"][:]

with h5py.File(posterior, "r") as f:
    x_post = f["model/x_grid"][:]
    t_post = f["model/t_grid"][:]
    thin = f["control/thin"][()]

    post_mean = f["model/mean"][:]

    x_data = f["data/x_grid"][:]
    y_data = f["data/y"][:]


nt = len(t_prior)
idx = list(map(int, [0.20 * nt, 0.80 * nt]))


# read in the covariance matrices, at the appropriate dims
with h5py.File(prior, "r") as f:
    prior_cov = f["model/covariance"][idx, 0:len(x_prior), 0:len(x_prior)]

with h5py.File(posterior, "r") as f:
    post_cov = f["model/covariance"][idx, 0:len(x_post), 0:len(x_post)]


set_plot_style()
colors = ['#fe9929', '#0868ac']  # orange, turquoise, blue
sns.set_palette(colors)
fig, axs = plt.subplots(2, 2, **fig_settings)
fig.set_constrained_layout_pads(wspace=0.1)
for a in fig.axes:
    a.tick_params(axis='y',
                  which='both',
                  left=True,
                  right=False,
                  labelleft=True)

# add subplot to label whole figure
# HACK: hide ticks/labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

axs[0, 0].set_title("Prior vs. DGP", fontsize="large")
axs[0, 0].plot(x_ref, u_ref[idx[0], :], label="DGP")
axs[0, 0].plot(x_prior, prior_mean[idx[0], 0:len(x_prior)], label="Prior")
uncertainty_bounds(axs[0, 0],
                   x_prior,
                   prior_mean[idx[0], 0:len(x_prior)],
                   prior_cov[0, :])
axs[0, 0].text(0.9, 0.1,
               fr"t = {t_prior[idx[0]]:.2f}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 0].transAxes)
axs[0, 0].text(-0.05, 1.10, "A",
               fontsize=12,
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 0].transAxes)
axs[0, 0].legend(loc="upper left")

axs[1, 0].plot(x_ref, u_ref[idx[1], :])
axs[1, 0].plot(x_prior, prior_mean[idx[1], 0:len(x_prior)])
uncertainty_bounds(axs[1, 0],
                   x_prior,
                   prior_mean[idx[1], 0:len(x_prior)],
                   prior_cov[1, :])
axs[1, 0].text(0.9, 0.1,
               fr"t = {t_prior[idx[1]]:.2f}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[1, 0].transAxes)

axs[0, 1].set_title("Posterior vs. DGP", fontsize="large")
axs[0, 1].plot(x_data, y_data[idx[0], :], ".", label="Data")
axs[0, 1].plot(x_ref, u_ref[idx[0], :], color=colors[0], label="DGP")
axs[0, 1].plot(x_post, post_mean[idx[0], 0:len(x_post)], label="Posterior")
uncertainty_bounds(axs[0, 1],
                   x_post,
                   post_mean[idx[0], 0:len(x_post)],
                   post_cov[0, :])
axs[0, 1].text(-0.05, 1.10, "B",
               fontsize=12,
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 1].transAxes)
axs[0, 1].text(0.9, 0.1,
               fr"t = {t_prior[idx[0]]:.2f}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 1].transAxes)
axs[0, 1].legend(loc=1)

axs[1, 1].plot(x_data, y_data[idx[1], :], ".")
axs[1, 1].plot(x_ref, u_ref[idx[1], :], color=colors[0])
axs[1, 1].plot(x_post, post_mean[idx[1], 0:len(x_post)])
uncertainty_bounds(axs[1, 1],
                   x_post,
                   post_mean[idx[1], 0:len(x_post)],
                   post_cov[1, :])
axs[1, 1].text(0.9, 0.075,
               fr"t = {t_prior[idx[1]]:.2f}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[1, 1].transAxes)

axs[0, 0].set_ylabel("u")
axs[1, 0].set_ylabel("u")

axs[1, 0].set_xlabel("x")
axs[1, 1].set_xlabel("x")
fig.savefig(args.output)
