import matplotlib.pyplot as plt

from lampe.plots import corner


def multi_corner_plots(samples_list, legends, colors, title, **kwargs):
    fig = None
    for s, l, c in zip(samples_list, legends, colors):
        fig = corner(s, legend=l, color=c, figure=fig, smooth=2, **kwargs)
        plt.suptitle(title)


def plot_distributions(dist_list, colors, labels, dim=1, hist=False):
    if dim == 1:
        for d, c, l in zip(dist_list, colors, labels):
            plt.hist(
                d, bins=100, color=c, alpha=0.3, density=True, label=l,
            )

    elif dim == 2:
        for d, c, l in zip(dist_list, colors, labels):
            if not hist:
                plt.scatter(
                    d[:, 0], d[:, 1], color=c, alpha=0.3, label=l,
                )
            else:
                plt.hist2d(
                    d[:, 0].numpy(),
                    d[:, 1].numpy(),
                    bins=100,
                    cmap=c,
                    alpha=0.7,
                    density=True,
                    label=l,
                )
    else:
        print("Not implemented.")


## =============== plots for normalizing flows ==============================


def flow_vs_reference_distribution(
    samples_ref, samples_flow, z_space=True, dim=1, hist=False
):
    if z_space:
        title = (
            r"Base-Distribution vs. Inverse Flow-Transformation (of $\Theta \mid x_0$)"
        )
        labels = [
            r"Ref: $\mathcal{N}(0,1)$",
            r"NPE: $T_{\phi}^{-1}(\Theta;x_0) \mid x_0$",
        ]
    else:
        title = r"True vs. Estimated distributions at $x_0$"
        labels = [r"Ref: $p(\Theta \mid x_0)$", r"NPE: $p(T_{\phi}(Z;x_0))$"]

    if hist:
        colors = ["Blues", "Oranges"]
    else:
        colors = ["blue", "orange"]
    plot_distributions(
        [samples_ref, samples_flow], colors=colors, labels=labels, dim=dim, hist=hist,
    )
    plt.title(title)

    if dim == 1:
        plt.xlabel("z")
        plt.xlim(-5, 5)

    elif dim == 2:
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
    plt.legend()
