import seaborn as sns


figure_fullpage = {"constrained_layout": True, "dpi": 300, "figsize": (12, 6)}  # width, height
figure_halfpage = {"constrained_layout": True, "dpi": 300, "figsize": (6, 4.125)}
figure_quartpage = {"constrained_layout": True, "dpi": 300, "figsize": (4.125, 3)}

figure_smallsquare = {"constrained_layout": True, "dpi": 300, "figsize": (4.125, 2.75)}
figure_bigsquare = {"constrained_layout": True, "dpi": 300, "figsize": (12, 12)}


def set_plot_style(transparent=False):
    sns.set(
        context="paper",
        font_scale=0.9,
        palette="deep",
        style="ticks",
        rc={"savefig.transparent": transparent}
    )
