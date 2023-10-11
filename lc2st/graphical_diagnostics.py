# Graphical diagnostics for the validation of conditional density estimators,
# in particular in the context of SBI
# - L-C2ST diagnostics
# - plot estimated (vs. true) p.d.f.s

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lampe.plots import corner
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy.stats import uniform


# ==== Utility functions  ====


def PP_vals(RV_samples, alphas):
    """Compute the PP-values: empirical c.d.f. of a random variable (RV).
    Used for Probability - Probabiity (P-P) plots.

    Args:
        RV_samples (np.array): samples from the random variable.
        alphas (list, np.array): alpha values to evaluate the c.d.f.

    Returns:
        pp_vals (list): empirical c.d.f. values for each alpha.
    """
    pp_vals = [np.mean(RV_samples <= alpha) for alpha in alphas]
    return pp_vals


def confidence_region_null(alphas, N=1000, conf_alpha=0.05, n_trials=1000):
    u_pp_values = {}
    for t in range(n_trials):
        u_samples = uniform().rvs(N)
        u_pp_values[t] = pd.Series(PP_vals(u_samples, alphas))
    lower_band = pd.DataFrame(u_pp_values).quantile(q=conf_alpha / 2, axis=1)
    upper_band = pd.DataFrame(u_pp_values).quantile(q=1 - conf_alpha / 2, axis=1)

    plt.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)


# ==== Diagnostics for L-C2ST ====


def pp_plot_lc2st(
    probas, probas_null, labels, colors, pp_vals_null=None, ax=None, **kwargs
):
    """Probability - Probability (P-P) plot for the classifier predicted
    class probabilities in (L)C2ST to assess the validity of a (or several)
    density estimator(s).

    Args:
        probas (list of np.arrays): list of predicted class probabilities for each case
            (i.e. associated to each density estimator).
        probas_null (list of ): list of predicted class probabilities in each trial
            under the null hypothesis.
        labels (list): list of labels for every density estimator.
        colors (list): list of colors for every density estimator.
        pp_vals_null (dict): dictionary of PP-values for each trial under the null
            hypothesis.
        ax (matplotlib.axes.Axes): axes to plot on.
        **kwargs: keyword arguments for matplotlib.pyplot.plot.

    Returns:
        ax (matplotlib.axes.Axes): axes with the P-P plot.
    """
    if ax == None:
        ax = plt.gca()
    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * len(probas), alphas)
    ax.plot(
        alphas, pp_vals_dirac, "--", color="black",
    )

    if pp_vals_null is None:
        pp_vals_null = {}
        for t in range(len(probas_null)):
            pp_vals_null[t] = pd.Series(PP_vals(probas_null[t], alphas))

    low_null = pd.DataFrame(pp_vals_null).quantile(0.05 / 2, axis=1)
    up_null = pd.DataFrame(pp_vals_null).quantile(1 - 0.05 / 2, axis=1)
    ax.fill_between(
        alphas,
        low_null,
        up_null,
        color="grey",
        alpha=0.2,
        # label="95% confidence region",
    )

    for p, l, c in zip(probas, labels, colors):
        pp_vals = pd.Series(PP_vals(p, alphas))
        ax.plot(alphas, pp_vals, label=l, color=c, **kwargs)
    return ax


def compute_dfs_with_probas_marginals(probas, P_eval):
    """Compute dataframes with predicted class probabilities for each
    (1d and 2d) marginal sample of the density estimator.
    Used in `eval_space_with_proba_intensity`.

    Args:
        probas (np.array): predicted class probabilities on test data.
        P_eval (torch.Tensor): corresponding sample from the density estimator
            (test data directly or transformed test data in the case of a
            normalizing flow density estimator).

    Returns:
        dfs (dict of pd.DataFrames): dict of dataframes if predicted probabilities
        for each marginal dimension (keys).
    """
    dim = P_eval.shape[-1]
    dfs = {}
    for i in range(dim):
        P_i = P_eval[:, i].numpy().reshape(-1, 1)
        df = pd.DataFrame({"probas": probas})
        df["z"] = P_i[:, 0]
        dfs[f"{i}"] = df

        for j in range(i + 1, dim):
            P_ij = P_eval[:, [i, j]].numpy()
            df = pd.DataFrame({"probas": probas})
            df["z_1"] = P_ij[:, 0]
            df["z_2"] = P_ij[:, 1]
            dfs[f"{i}_{j}"] = df
    return dfs


def eval_space_with_proba_intensity(
    df_probas,
    dim=1,
    z_space=True,
    n_bins=20,
    vmin=0,
    vmax=1,
    cmap=cm.get_cmap("Spectral_r"),
    show_colorbar=True,
    ax=None,
):
    """Plot 1d or 2d marginal histogram of samples of the density estimator
    with probabilities as color intensity.

    Args:
        df_probas (pd.DataFrame): dataframe with predicted class probabilities
            as obtained from `compute_dfs_with_probas_marginals`.
        dim (int): dimension of the marginal histogram to plot.
        z_space (bool): whether to plot the histogram in latent space
            of the base distribution of a normalizing flow. If False, plot
            in the original space of the estimated density.
        n_bins (int): number of bins for the histogram.
        vmin (float): minimum value for the color intensity.
        vmax (float): maximum value for the color intensity.
        cmap (matplotlib.colors.Colormap): colormap for the color intensity.
        show_colorbar (bool): whether to show the colorbar.
        ax (matplotlib.axes.Axes): axes to plot on.

    Returns:
        ax (matplotlib.axes.Axes): axes with the plot.
    """
    if ax is None:
        ax = plt.gca()

    if dim == 1:
        _, bins, patches = ax.hist(df_probas.z, n_bins, density=True, color="green")
        df_probas["bins"] = np.select(
            [df_probas.z <= i for i in bins[1:]], list(range(n_bins))
        )
        # get mean predicted proba for each bin
        weights = df_probas.groupby(["bins"]).mean().probas

        id = list(set(range(n_bins)) - set(df_probas.bins))
        patches = np.delete(patches, id)
        bins = np.delete(bins, id)

        norm = Normalize(vmin=vmin, vmax=vmax)

        for w, p in zip(weights, patches):
            p.set_facecolor(cmap(w))  # color is mean predicted proba

        if show_colorbar:
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    elif dim == 2:
        if z_space:
            legend = r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$"
        else:
            legend = r"$\hat{p}(\Theta\sim q_{\phi}(\theta \mid x_0) \mid x_0)$"

        _, x, y = np.histogram2d(df_probas.z_1, df_probas.z_2, bins=n_bins)
        df_probas["bins_x"] = np.select(
            [df_probas.z_1 <= i for i in x[1:]], list(range(n_bins))
        )
        df_probas["bins_y"] = np.select(
            [df_probas.z_2 <= i for i in y[1:]], list(range(n_bins))
        )
        # get mean predicted proba for each bin
        prob_mean = df_probas.groupby(["bins_x", "bins_y"]).mean().probas

        weights = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            for j in range(n_bins):
                try:
                    weights[i, j] = prob_mean.loc[i].loc[j]
                except KeyError:
                    # if no sample in bin, set color to white
                    weights[i, j] = np.nan

        norm = Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                facecolor = cmap(norm(weights.T[j, i]))
                # if no sample in bin, set color to white
                if weights.T[j, i] == np.nan:
                    facecolor = "white"
                rect = Rectangle(
                    (x[i], y[j]),
                    x[i + 1] - x[i],
                    y[j + 1] - y[j],
                    facecolor=facecolor,  # color is mean predicted proba
                    edgecolor="none",
                )
                ax.add_patch(rect)
        if show_colorbar:
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=legend)

    else:
        print("Not implemented.")

    return ax


# ================= Plots p.d.f.s ======================


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


# plots for normalizing flows
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
