# Graphical diagnostics for the validation of conditional density estimators,
# in particular in the context of SBI. They help interpret the results of the
# following test:
#
# 1. Simulation Based Calibration (SBC) [Talts et al. (2018)]
# 2. (Local) Classifier Two Sample Test (C2ST) (can be used for any SBI-algorithm)
#    - [Lopez et al. (2016)](https://arxiv.org/abs/1602.05336))
#    - [Lee et al. (2018)](https://arxiv.org/abs/1812.08927))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from scipy.stats import hmean, uniform


# ==== Functions applicable for both tests ====


def PP_vals(RV_values, alphas):
    pp_vals = [np.mean(RV_values <= alpha) for alpha in alphas]
    return pp_vals


def confidence_region_null(alphas, N=1000, conf_alpha=0.05, n_trials=1000):
    u_pp_values = {}
    for t in range(n_trials):
        u_samples = uniform().rvs(N)
        u_pp_values[t] = pd.Series(PP_vals(u_samples, alphas))
    lower_band = pd.DataFrame(u_pp_values).quantile(q=conf_alpha / 2, axis=1)
    upper_band = pd.DataFrame(u_pp_values).quantile(q=1 - conf_alpha / 2, axis=1)

    plt.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)


def box_plot_lc2st(
    scores, scores_null, labels, colors, title=r"Box plot", conf_alpha=0.05
):
    import matplotlib.cbook as cbook

    data = scores_null
    stats = cbook.boxplot_stats(data)[0]
    stats["q1"] = np.quantile(data, conf_alpha)
    stats["q3"] = np.quantile(data, 1 - conf_alpha)
    stats["whislo"] = min(data)
    stats["whishi"] = max(data)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    bp = ax.bxp([stats], widths=0.1, vert=False, showfliers=False, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgray")
    ax.set_label(r"95% confidence interval$")
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(stats["whislo"] - np.std(data), max(scores) + np.std(data))

    for s, l, c in zip(scores, labels, colors):
        plt.text(s, 0.9, l, color=c)
        plt.scatter(s, 1, color=c, zorder=10)

    fig.set_size_inches(5, 2)
    plt.title(title)


# ==== 1. SBC: PP-plot for SBC validation method =====


def sbc_plot(
    sbc_ranks,
    colors,
    labels,
    alphas=np.linspace(0, 1, 100),
    confidence_int=True,
    conf_alpha=0.05,
    title="SBC",
):
    """PP-plot for SBC validation method:
    Empirical distribution of the SBC ranks computed for every parameter seperately.

    inputs:
    - sbc_ranks: numpy array, size: (N, dim)
        For example one can use the output of sbi.analysis.sbc.run_sbc computed on
        N samples of the joint (Theta, X).
    - colors: list of strings, length: dim
    - labels: list of strings, length: dim
    - alphas: numpy array, size: (K,)
        Default is np.linspace(0,1,100).
    - confidence_int: bool
        Whether to show the confidence region (acceptance of the null hypothesis).
        Default is True.
    - conf_alpha: alpha level of the (1-conf-alpha)-confidence region.
        Default is 0.05, for a confidence level of 0.95.
    - title: sting
        Title of the plot.
    """
    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        plt.plot(alphas, sbc_cdf / sbc_cdf.max(), color=colors[i], label=labels[i])

    if confidence_int:
        # Construct uniform histogram.
        N = len(sbc_ranks)
        confidence_region_null(alphas=alphas, N=N, conf_alpha=conf_alpha)

    plt.ylabel("empirical CDF", fontsize=15)
    plt.xlabel("ranks", fontsize=15)
    plt.title(title, fontsize=18)
    plt.legend()


# ==== 2. (Local) Classifier Two Sample Test (C2ST) ====

# PP-plot of clasifier predicted class probabilities


def pp_plot_c2st(
    probas, probas_null, labels, colors, pp_vals_null=None, ax=None, **kwargs
):
    if ax == None:
        ax = plt.gca()
    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * len(probas), alphas)
    ax.plot(
        alphas,
        pp_vals_dirac,
        "--",
        color="black",
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


# Interpretability plots for C2ST: regions of high/low predicted class probabilities


def compute_dfs_with_probas_marginals(probas, P_eval):
    dim = P_eval.shape[-1]
    dfs = {}
    for i in range(dim):
        P_i = P_eval[:, i].numpy().reshape(-1, 1)
        df = pd.DataFrame({"probas": probas})
        df["z"] = P_i[:,0]
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

