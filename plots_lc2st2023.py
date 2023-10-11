import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tueplots import fonts, axes
import matplotlib.gridspec as gridspec
from matplotlib import cm

import pandas as pd
import torch

from scipy.stats import binom, uniform

from lc2st.graphical_diagnostics import (
    PP_vals,
    compute_dfs_with_probas_marginals,
    eval_space_with_proba_intensity,
    pp_plot_lc2st,
)

plt.rcParams.update(fonts.neurips2022())
plt.rcParams.update(axes.color(base="black"))
plt.rcParams["legend.fontsize"] = 23.0
plt.rcParams["xtick.labelsize"] = 23.0
plt.rcParams["ytick.labelsize"] = 23.0
plt.rcParams["axes.labelsize"] = 23.0
plt.rcParams["font.size"] = 23.0
plt.rcParams["axes.titlesize"] = 27.0

alpha_fill_between = 0.2
linewidth = 2.0

# ======== FIGURE 1 ==========

METRICS_DICT = {
    "acc_single_class": {
        "label": r"$\hat{t}_{\mathrm{Acc}_0}$",
        "color": "orange",
        "linestyle": "--",
    },
    "acc_ref": {
        "label": r"$\hat{t}_{\mathrm{Acc}}$",
        "color": "orange",
        "linestyle": "-",
    },
    "mse_single_class": {
        "label": r"$\hat{t}_{\mathrm{MSE}_0}$",
        "color": "blue",
        "linestyle": "--",
    },
    "mse_ref": {
        "label": r"$\hat{t}_{\mathrm{MSE}}$",
        "color": "blue",
        "linestyle": "-",
    },
    "max_single_class": {
        "label": r"$\hat{t}_{\mathrm{Max}_0}$",
        "color": "magenta",
        "linestyle": "--",
    },
    "max_ref": {
        "label": r"$\hat{t}_{\mathrm{Max}}$",
        "color": "magenta",
        "linestyle": "-",
    },
}


def plot_plot_c2st_single_eval_shift(
    shift_list,
    t_stats_dict,
    TPR_dict,
    TPR_std_dict,
    shift_name,
    clf_name,
):
    plt.rcParams["figure.figsize"] = (10, 5)

    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=False, constrained_layout=True
    )
    # plot theoretical H_0 value for t-stats
    axs[0].plot(
        shift_list,
        [0.5] * len(shift_list),
        color="black",
        linestyle="--",
        label=r"$t \mid \mathcal{H}_0$",
    )
    for t_stat_name, t_stats in t_stats_dict.items():
        if "max" in t_stat_name:
            continue
        if "mse" in t_stat_name:
            t_stats = np.array(t_stats) + 0.5
            METRICS_DICT[t_stat_name]["label"] += r" (+0.5)"
        axs[0].plot(
            shift_list,
            t_stats,
            label=METRICS_DICT[t_stat_name]["label"],
            color=METRICS_DICT[t_stat_name]["color"],
            linestyle=METRICS_DICT[t_stat_name]["linestyle"],
            # alpha=0.8,
            linewidth=linewidth,
        )
        axs[1].plot(
            shift_list,
            TPR_dict[t_stat_name],
            label=METRICS_DICT[t_stat_name]["label"],
            color=METRICS_DICT[t_stat_name]["color"],
            linestyle=METRICS_DICT[t_stat_name]["linestyle"],
            # alpha=0.8,
            zorder=100,
            linewidth=linewidth,
        )
        err = np.array(TPR_std_dict[t_stat_name])
        axs[1].fill_between(
            shift_list,
            np.array(TPR_dict[t_stat_name]) - err,
            np.array(TPR_dict[t_stat_name]) + err,
            alpha=alpha_fill_between,
            color=METRICS_DICT[t_stat_name]["color"],
        )
    if shift_name == "variance":
        axs[0].set_xlabel(r"$\sigma^2$")
        axs[1].set_xlabel(r"$\sigma^2$")
    else:
        axs[0].set_xlabel(f"{shift_name} shift")
        axs[1].set_xlabel(f"{shift_name} shift")

    # axs[0].set_ylabel("test statistic")
    axs[0].set_ylim(0.38, 1.01)
    axs[0].set_yticks([0.5, 1.0])
    axs[0].legend()
    axs[0].set_title("Optimal Bayes (statistics)")
    # axs[1].set_ylabel("empirical power")
    axs[1].set_yticks([0.0, 0.5, 1.0])
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_title(f"{clf_name}-C2ST (power)")

    return fig


# ======== FIGURE 2 ========== #

METHODS_DICT = {
    r"oracle C2ST ($\hat{t}_{Acc}$)": {
        "test_name": "c2st",
        "t_stat_name": "accuracy",
        "color": "grey",
        "linestyle": "-",
        "marker": "d",
        "markersize": 10,
    },
    r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)": {
        "test_name": "c2st",
        "t_stat_name": "mse",
        "color": "grey",
        "linestyle": "-",
        "marker": "o",
        "markersize": 10,
    },
    r"$\ell$-C2ST ($\hat{t}_{\mathrm{MSE}_0}$)": {
        "test_name": "lc2st",
        "t_stat_name": "mse",
        "color": "blue",
        "linestyle": "-",
        "marker": "o",
        "markersize": 10,
    },
    r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "mse",
        "color": "blue",
        "linestyle": "--",
        "marker": "*",
        "markersize": 16,
    },
    r"$\ell$-C2ST-NF-perm ($\hat{t}_{\mathrm{MSE}_0}$)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "mse",
        "color": "darkblue",
        "linestyle": "--",
        "marker": "o",
        "markersize": 10,
    },
    r"$\ell$-C2ST ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st",
        "t_stat_name": "div",
        "color": "magenta",
        "linestyle": "-",
        "marker": "o",
        "markersize": 10,
    },
    r"$\ell$-C2ST-NF ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "div",
        "color": "magenta",
        "linestyle": "--",
        "marker": "*",
        "markersize": 16,
    },
    r"$\ell$-C2ST-NF-perm ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "div",
        "color": "red",
        "linestyle": "--",
        "marker": "o",
        "markersize": 10,
    },
    r"$local$-HPD": {
        "test_name": "lhpd",
        "t_stat_name": "mse",
        "color": "orange",
        "linestyle": "-",
        "marker": "^",
        "markersize": 10,
    },
    "SBC": {
        "test_name": "sbc",
        "colors": ["#E697A1", "#E95b88", "#C92E45", "#490816"],
    },
}

avg_result_keys = {
    "TPR": "reject",
    "p_value_mean": "p_value",
    "p_value_std": "p_value",
    "t_stat_mean": "t_stat",
    "t_stat_std": "t_stat",
    "run_time_mean": "run_time",
    "run_time_std": "run_time",
}


def plot_sbibm_results_n_train_n_cal(
    results_n_train,
    results_n_cal,
    methods_mse,
    methods_all,
    n_train_list,
    n_cal_list,
    plot_p_value=False,
    title=None,
):
    plt.rcParams["figure.figsize"] = (28, 5)
    plt.rcParams["legend.fontsize"] = 25.0
    plt.rcParams["xtick.labelsize"] = 32.0
    plt.rcParams["ytick.labelsize"] = 32.0
    plt.rcParams["axes.labelsize"] = 32.0
    plt.rcParams["font.size"] = 32.0
    plt.rcParams["axes.titlesize"] = 40.0

    fig, axs = plt.subplots(
        nrows=1, ncols=4, sharex=False, sharey=False, constrained_layout=True
    )
    # ==== t_stats of L-C2ST(-NF) w.r.t to oracle ====

    # plot theoretical H_0 value for mse t-stats
    axs[0].plot(
        np.arange(len(n_train_list)),
        np.ones(len(n_train_list)) * 0.0,
        "--",
        color="black",
        label=r"$t \mid \mathcal{H}_0$",
    )
    axs[0].legend()
    # plot estimated T values
    # for i, methods in enumerate([methods_acc, methods_mse]): # t_Max is not used in the paper
    for method in methods_mse:
        if (
            "perm" in method  # the permuation test is only used for the null hypothesis
            or "HPD" in method  # HPD does not have a comparable t-statistic
        ):
            continue
        test_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]

        axs[0].plot(
            np.arange(len(n_train_list)),
            results_n_train[test_name]["t_stat_mean"][t_stat_name],
            # label=method,
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            # alpha=0.8,
            linewidth=linewidth,
        )
        err = np.array(results_n_train[test_name]["t_stat_std"][t_stat_name])
        axs[0].fill_between(
            np.arange(len(n_train_list)),
            np.array(results_n_train[test_name]["t_stat_mean"][t_stat_name]) - err,
            np.array(results_n_train[test_name]["t_stat_mean"][t_stat_name]) + err,
            alpha=0.2,
            color=METHODS_DICT[method]["color"],
        )
    axs[0].legend()  # loc="lower left")
    axs[0].set_xticks(
        np.arange(len(n_train_list)), [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
    )
    axs[0].set_ylabel("test statistic")
    axs[0].set_ylim(0.48, 1.02)
    axs[0].set_yticks([0.5, 0.75, 1.0])
    axs[0].set_ylabel("test statistic")
    axs[0].set_ylim(-0.01, 0.26)
    axs[0].set_yticks([0.0, 0.12, 0.25])
    axs[0].set_xlabel(r"$N_{\mathrm{train}}$ ($N_{\mathrm{cal}}=10^4$)")

    if plot_p_value:
        # ==== p-value of all methods w.r.t to oracle ===

        # plot alpha-level
        axs[1].plot(
            np.arange(len(n_train_list)),
            np.ones(len(n_train_list)) * 0.05,
            "--",
            color="black",
            label=r"$\alpha$-level: 0.05",
        )

        # plot estimated p-values
        for method in methods_all:
            if "Max" in method:
                continue  # t_Max is not used in the paper

            test_name = METHODS_DICT[method]["test_name"]
            t_stat_name = METHODS_DICT[method]["t_stat_name"]
            axs[1].plot(
                np.arange(len(n_train_list)),
                results_n_train[test_name]["p_value_mean"][t_stat_name],
                # label=method,
                color=METHODS_DICT[method]["color"],
                linestyle=METHODS_DICT[method]["linestyle"],
                marker=METHODS_DICT[method]["marker"],
                markersize=METHODS_DICT[method]["markersize"],
                # alpha=0.8,
                linewidth=linewidth,
            )
            low = np.array(results_n_train[test_name]["p_value_min"][t_stat_name])
            high = np.array(results_n_train[test_name]["p_value_max"][t_stat_name])
            axs[1].fill_between(
                np.arange(len(n_train_list)),
                low,
                high,
                alpha=alpha_fill_between,
                color=METHODS_DICT[method]["color"],
            )
        # axs[1].legend(loc="upper left")
        axs[1].set_xticks(
            np.arange(len(n_train_list)), [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
        )
        # axs[1].set_xlabel(r"$N_{\mathrm{train}}$ ($N_{\mathrm{cal}}=10^4$)")
        axs[1].set_ylabel("p-value (min / max)")

    else:
        # plot rejection rate of all methods w.r.t to oracle
        for method in methods_all:
            if "Max" in method:
                continue  # t_Max is not used in the paper

            test_name = METHODS_DICT[method]["test_name"]
            t_stat_name = METHODS_DICT[method]["t_stat_name"]

            axs[1].plot(
                np.arange(len(n_train_list)),
                results_n_train[test_name]["TPR_mean"][t_stat_name],
                label=method,
                color=METHODS_DICT[method]["color"],
                linestyle=METHODS_DICT[method]["linestyle"],
                marker=METHODS_DICT[method]["marker"],
                markersize=METHODS_DICT[method]["markersize"],
                # alpha=0.8,
                linewidth=linewidth,
            )

            err = np.array(results_n_train[test_name]["TPR_std"][t_stat_name])
            axs[1].fill_between(
                np.arange(len(n_train_list)),
                np.array(results_n_train[test_name]["TPR_mean"][t_stat_name]) - err,
                np.array(results_n_train[test_name]["TPR_mean"][t_stat_name]) + err,
                alpha=alpha_fill_between,
                color=METHODS_DICT[method]["color"],
            )

        # axs[1].legend(loc="lower left")
        axs[1].set_xticks(
            np.arange(len(n_train_list)), [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
        )
        axs[1].set_ylim(-0.04, 1.04)
        axs[1].set_yticks([0, 0.5, 1])
        axs[1].set_xlabel(r"$N_{\mathrm{train}}$ ($N_{\mathrm{cal}}=10^4$)")
        axs[1].set_ylabel("power (TPR)")

    # plot emp power as function of n_cal
    for axi, result_name in zip([axs[2], axs[3]], ["TPR", "FPR"]):
        for method in methods_all:
            if "Max" in method:
                continue
            test_name = METHODS_DICT[method]["test_name"]
            t_stat_name = METHODS_DICT[method]["t_stat_name"]
            axi.plot(
                np.arange(len(n_cal_list)),
                results_n_cal[test_name][result_name + "_mean"][t_stat_name],
                label=method,
                color=METHODS_DICT[method]["color"],
                linestyle=METHODS_DICT[method]["linestyle"],
                marker=METHODS_DICT[method]["marker"],
                markersize=METHODS_DICT[method]["markersize"],
                # alpha=0.8,
                linewidth=linewidth,
            )
            err = np.array(results_n_cal[test_name][result_name + "_std"][t_stat_name])
            axi.fill_between(
                np.arange(len(n_cal_list)),
                np.array(results_n_cal[test_name][result_name + "_mean"][t_stat_name])
                - err,
                np.array(results_n_cal[test_name][result_name + "_mean"][t_stat_name])
                + err,
                alpha=alpha_fill_between,
                color=METHODS_DICT[method]["color"],
            )
        # add emp power as function of n_cal
        axi.set_xlabel(r"$N_{\mathrm{cal}}$ ($N_{\mathrm{train}}=10^3$)")
        axi.set_xticks(
            np.arange(len(n_cal_list)),
            [r"$100$", r"$500$", r"$1000$", r"$2000$", r"$5000$", r"$10^{4}$"],
        )

    # plot significance level
    axs[3].plot(
        np.arange(len(n_cal_list)),
        np.ones(len(n_cal_list)) * 0.05,
        "--",
        color="black",
        label=r"$\alpha$-level: $0.05$",
    )
    axs[3].legend()

    axs[2].set_ylabel(r"power (TPR)")
    axs[2].set_ylim(-0.04, 1.04)
    axs[2].set_yticks([0, 0.5, 1])

    axs[3].set_ylabel(r"type I error (FPR)")
    axs[3].set_ylim(-0.04, 1.04)
    axs[3].set_yticks([0, 0.5, 1])

    if title is not None:
        plt.suptitle(title)

    return fig


### ======== FIGURE 3 ========== ###


def global_coverage_pp_plots(
    alphas,
    sbc_ranks,
    hpd_ranks,
    confidence_int=True,
    conf_alpha=0.05,
    n_trials=1000,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # plot identity function
    lims = [np.min([0, 0]), np.max([1, 1])]
    ax.plot(lims, lims, "--", color="black", alpha=0.75)
    if confidence_int:
        # conf_alpha = conf_alpha  / len(sbc_ranks[0])  # bonferonni correction
        # Construct uniform histogram.
        N = len(sbc_ranks)
        u_pp_values = {}
        for t in range(n_trials):
            u_samples = uniform().rvs(N)
            u_pp_values[t] = pd.Series(PP_vals(u_samples, alphas))
        lower_band = pd.DataFrame(u_pp_values).quantile(q=conf_alpha / 2, axis=1)
        upper_band = pd.DataFrame(u_pp_values).quantile(q=1 - conf_alpha / 2, axis=1)

        ax.fill_between(
            alphas, lower_band, upper_band, color="grey", alpha=alpha_fill_between
        )

    # sbc ranks
    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        ax.plot(
            alphas,
            sbc_cdf / sbc_cdf.max(),
            color=METHODS_DICT["SBC"]["colors"][i],
            label=rf"$\mathrm{{SBC}} - \theta_{i+1}$",
            linewidth=linewidth,
        )

    # hpd_values
    alphas = torch.linspace(0.0, 1.0, len(hpd_ranks))
    ax.plot(
        alphas,
        hpd_ranks,
        color=METHODS_DICT[r"$local$-HPD"]["color"],
        label=r"$\mathrm{HPD}(\theta)$",
        linewidth=linewidth,
    )
    ax.set_ylabel("empirical CDF")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("rank")

    ax.legend(loc="upper left")
    ax.set_aspect("equal", "box")
    return ax


def plot_local_t_stats_gain(
    gain_dict,
    t_stats_obs,
    t_stats_obs_null,
    methods=[r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)"],
    labels=[r"$\hat{t}_{\mathrm{MSE}_0}(x_{\mathrm{o}})$ / $\ell$-C2ST-NF"],
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    # test statistics
    gain_list_t_stats = list(gain_dict.keys())
    for i, method in enumerate(methods):
        method_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]
        t_stats = np.array(
            [
                t_stats_obs[method_name][t_stat_name][i]
                for i in range(len(gain_list_t_stats))
            ]
        )
        t_stats_null_low = np.array(
            [
                np.quantile(t_stats_obs_null[method_name][g][t_stat_name], q=0.05 / 2)
                for g in gain_list_t_stats
            ]
        )
        t_stats_null_high = np.array(
            [
                np.quantile(
                    t_stats_obs_null[method_name][g][t_stat_name], q=1 - 0.05 / 2
                )
                for g in gain_list_t_stats
            ]
        )

        ax.plot(
            gain_list_t_stats,
            t_stats,
            label=labels[i],
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            # alpha=0.8,
            linewidth=linewidth,
        )
    ax.plot(
        gain_list_t_stats,
        np.ones(len(gain_list_t_stats)) * 0.0,
        "--",
        color="black",
        label=r"$t\mid\mathcal{H}_0$",
    )
    ax.fill_between(
        gain_list_t_stats,
        t_stats_null_low,
        t_stats_null_high,
        alpha=alpha_fill_between,
        color="grey",
        label=r"95% CR",
    )

    ax.yaxis.set_tick_params(which="both", labelleft=True)
    ax.set_xticks(gain_list_t_stats)

    ax.set_xlabel(r"gain ($g_\mathrm{o}$)")
    ax.set_ylabel("test statistic")
    ax.legend()
    return ax


def local_pp_plot(probas_obs, pp_vals_null_obs, method, text="", ax=None):
    if ax is None:
        ax = plt.gca()

        pp_plot_lc2st(
            ax=ax,
            probas=[probas_obs],
            pp_vals_null=pp_vals_null_obs,
            probas_null=None,
            colors=[METHODS_DICT[method]["color"]],
            labels=[""],
            linewidth=linewidth,
        )
        ax.text(
            0.0,
            0.9,
            text,
            bbox=dict(
                facecolor="none",
                edgecolor=METHODS_DICT[method]["color"],
                boxstyle="round,pad=0.2",
            ),
        )

    ax.set_ylabel(r"empirrical CDF")
    ax.set_yticks([0.0, 0.5, 1.0])

    ax.set_aspect("equal")

    return ax


def local_tstats_with_pp_plots(
    gain_dict,
    t_stats_obs,
    t_stats_obs_null,
    gain_list_pp_plots,
    probas_obs,
    probas_obs_null,
    p_values_obs,
    methods=[r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)"],
    labels=[r"$\hat{t}_{\mathrm{MSE}_0}(x_{\mathrm{o}})$ / $\ell$-C2ST-NF"],
    colors_g0=["#32327B", "#3838E2", "#52A9F5"],
):
    plt.rcParams["figure.figsize"] = (10, 10)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3)

    ax = fig.add_subplot(gs[0, :])

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax0)
    axes1 = [ax0, ax1, ax2]

    for ax1 in [ax1, ax2]:
        ax1.set_yticklabels([])
        ax1.set_xticks([0.0, 0.5, 1.0])

    # test statistics
    ax = plot_local_t_stats_gain(
        gain_dict,
        t_stats_obs,
        t_stats_obs_null,
        methods=methods,
        labels=labels,
        ax=ax,
    )
    ax.set_title("Local Test statistics")

    # pp-plots
    method = r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)"
    method_name = METHODS_DICT[method]["test_name"]
    t_stat_name = METHODS_DICT[method]["t_stat_name"]
    probas_obs = probas_obs[method_name]
    probas_obs_null = probas_obs_null[method_name]
    p_values_obs = p_values_obs[method_name][t_stat_name]
    for n, (axs1, g) in enumerate(zip(axes1, gain_list_pp_plots)):
        pp_plot_lc2st(
            ax=axs1,
            probas=[probas_obs[g]],
            probas_null=probas_obs_null[g],
            colors=[colors_g0[n]],
            labels=[""],
        )
        axs1.text(
            0.0,
            0.9,
            r"$g_\mathrm{o}=$"
            + f"{g}",  # + "\n" + r"$p-value=$" + f"{p_values_obs[num_g]}",
            fontsize=23,
        )
        plt.setp(axs1.spines.values(), color=colors_g0[n])

        # axs1.set_xlabel(r"$\alpha$", fontsize=23)
        if n == 1:
            axs1.set_title(
                r"Local PP-plots"  # for class 0: $1-\hat{d}(Z,x_0), Z\sim \mathcal{N}(0,1)$"
            )
            # axs1.legend(loc="lower right")
    axes1[0].set_ylabel("empirical CDF")
    axes1[0].set_yticks([0.0, 0.5, 1.0])

    handles_1 = ax.get_legend_handles_labels()
    handles_2 = axs1.get_legend_handles_labels()
    ax.legend(
        handles=handles_1[0] + handles_2[0],
        # title="1D-plots for",
        loc="upper right",
        # bbox_to_anchor=ax.get_position().get_points()[1]
        # + np.array([1.6, -0.08]),
    )

    # plt.subplots_adjust(wspace=None, hspace=0.4)

    for ax1 in axes1:
        ax1.set_aspect("equal")
    # ax.set_aspect("equal")
    # for j in [0, 2]:
    #     axes[0][j].set_visible(False)
    ax.set_xlim(-20.1, 20.1)
    # fig.align_ylabels()

    return fig


def global_vs_local_tstats(
    sbc_alphas,
    sbc_ranks,
    hpd_ranks,
    gain_dict,
    t_stats_obs,
    t_stats_obs_null,
    methods,
    labels,
    alpha=0.05,
    n_trials=1000,
):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(11, 5), constrained_layout=True)
    axs[0] = global_coverage_pp_plots(
        sbc_alphas,
        sbc_ranks,
        hpd_ranks,
        confidence_int=True,
        ax=axs[0],
        n_trials=n_trials,
        conf_alpha=alpha,
    )

    axs[1] = plot_local_t_stats_gain(
        gain_dict,
        t_stats_obs,
        t_stats_obs_null,
        methods=methods,
        labels=labels,
        ax=axs[1],
    )
    axs[1].set_ylim(-0.01, 0.11)
    axs[1].set_yticks([0.0, 0.05, 0.1])
    axs[1].set_xlim(-20.5, 20.5)  # no +/-25 (outside of the prior)
    return fig


# Simulator parameters
PARAMETER_DICT = {
    0: {"label": r"$C$", "low": 10.0, "high": 300.0, "ticks": [100, 250]},
    1: {"label": r"$\mu$", "low": 50.0, "high": 500.0, "ticks": [200, 400]},
    2: {"label": r"$\sigma$", "low": 100.0, "high": 5000.0, "ticks": [1000, 3500]},
    3: {"label": r"$g$", "low": -22.0, "high": 22.0, "ticks": [-20, 0, 20]},
}


def plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st(
    theta_gt,
    probas,
    P_eval,
    n_bins=20,
    cmap=cm.get_cmap("Spectral_r"),
):
    plt.rcParams["figure.figsize"] = (9, 9)

    fig, axs = plt.subplots(
        nrows=4, ncols=4, sharex=False, sharey=False, constrained_layout=False
    )

    dfs = compute_dfs_with_probas_marginals(probas, P_eval=P_eval)

    for i in range(4):
        eval_space_with_proba_intensity(
            df_probas=dfs[f"{i}"],
            dim=1,
            z_space=False,
            n_bins=n_bins,
            vmin=0.2,
            vmax=0.8,
            cmap=cmap,
            show_colorbar=False,
            ax=axs[i][i],
        )

        # plot ground truth dirac
        axs[i][i].axvline(x=theta_gt[i], ls="--", c="black", linewidth=linewidth)

        for j in range(i + 1, 4):
            eval_space_with_proba_intensity(
                df_probas=dfs[f"{i}_{j}"],
                dim=2,
                z_space=False,
                n_bins=n_bins,
                vmin=0.2,
                vmax=0.8,
                cmap=cmap,
                show_colorbar=False,
                ax=axs[j][i],
            )

            # plot points
            axs[j][i].scatter(theta_gt[i], theta_gt[j], color="black", s=15)
            axs[i][j].set_visible(False)


    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.82, 0.1, 0.075, 0.8])
    plt.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=cm.colors.Normalize(vmin=0.2, vmax=0.8)),
        cax=cax,
        label=r"Predicted Probability ($\ell$-C2ST-NF)",
    )

    fig.suptitle(
        r"Pair-plots of $q_{\phi}(\theta \mid x_\mathrm{o})$: "
        + "\n"
        + r"$\Theta = T_{\phi}(Z ; x_\mathrm{o}), \quad Z\sim \mathcal{N}(0,\mathbf{1}_4)$",
        y=1.0,
    )

    # set value range
    axs[0][0].set_yticks([])
    axs[3][3].set_xticks(PARAMETER_DICT[3]["ticks"])

    for j in range(4):
        if j != 0:
            axs[j][0].set_ylabel(PARAMETER_DICT[j]["label"])
            axs[j][0].set_xlim(PARAMETER_DICT[0]["low"], PARAMETER_DICT[0]["high"])
            axs[j][0].set_xticks(PARAMETER_DICT[0]["ticks"])
        axs[3][j].set_xlabel(PARAMETER_DICT[j]["label"])
        for i in range(4):
            if j != 3:
                axs[j][i].set_xticklabels([])
            if i != 0:
                axs[j][i].set_yticklabels([])
            if i != j:
                axs[j][i].set_ylim(PARAMETER_DICT[j]["low"], PARAMETER_DICT[j]["high"])
                axs[j][i].set_yticks(PARAMETER_DICT[j]["ticks"])

            axs[j][i].set_xlim(PARAMETER_DICT[i]["low"], PARAMETER_DICT[i]["high"])
            axs[j][i].set_xticks(PARAMETER_DICT[i]["ticks"])

    return fig
