import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from common.figset import set_plot_style
from common.tankmeta import add_soliton_operator


parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with h5py.File(args.simulation, "r") as f:
    x_post = f["model/x_grid"][:]
    t_post = f["model/t_grid"][:]
    post = f["model/mean"][:, 0:len(x_post)]

H = add_soliton_operator(x_post)
x_post = x_post[x_post <= 6]
post = (H @ post.T).T
set_plot_style()
fig, ax = plt.subplots(constrained_layout=True,
                       figsize=(6 / 2.54, 4 / 2.54),
                       dpi=600)

xmesh, tmesh = np.meshgrid(x_post, t_post)
im = ax.pcolormesh(xmesh, tmesh, post, cmap="coolwarm")
plt.colorbar(im, ax=ax)
ax.set_xlabel("x (m)")
ax.set_ylabel("t (s)")
ax.set_title("Posterior mean")
ax.text(-0.05, 1.10, "D",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes)


fig.savefig(args.output)
