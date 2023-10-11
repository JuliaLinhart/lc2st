# =============================================================================

#   SCRIPT FOR SCATTER PLOTS OF TEST STATISTICS (MSE) FOR THE SBIBM TASKS
#                           C2ST vs. L-C2ST(-NF)

# =============================================================================

# DESCRIPTION:
# We compare the test statistics (MSE) of the C2ST and L-C2ST(-NF) methods for
# the different NPEs.
#
# > SBIBM tasks: two_moons, slcp, gaussian_mixture, gaussian_linear_uniform, bernoulli_glm, bernoulli_glm_raw
#
# > Reference Observations:
#   - task: 10 precomputed and loaded from the sbibm task
#   - empirical: 100 randomly generated using prior and simulator

# USAGE:
# >> python figure6_lc2st_2023.py --task <> --observations <task/empirical> --method <lc2st/lc2st_nf>


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sbibm

from pathlib import Path
from scipy import stats
from lc2st.vanillaC2ST import sbibm_clf_kwargs
from experiment_utils_sbibm import (
    generate_data_one_run,
    compute_test_results_npe_one_run,
)
from tueplots import fonts, axes

# Plotting parameters
plt.rcParams.update(fonts.neurips2022())
plt.rcParams.update(axes.color(base="black"))
plt.rcParams["legend.fontsize"] = 23.0
plt.rcParams["xtick.labelsize"] = 23.0
plt.rcParams["ytick.labelsize"] = 23.0
plt.rcParams["axes.labelsize"] = 23.0
plt.rcParams["font.size"] = 23.0
plt.rcParams["axes.titlesize"] = 27.0
plt.rcParams["figure.figsize"] = (5, 5)


METHODS_DICT = {
    "c2st": r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)",
    "lc2st": {
        "name": r"$\ell$-C2ST ($\hat{t}_{\mathrm{MSE}_0}$)",
        "marker": "o",
        "colors": plt.get_cmap("RdPu", 6),
    },
    "lc2st_nf": {
        "name": r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)",
        "marker": "*",
        "colors": plt.get_cmap("RdPu", 6),
    },
}

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Classifier / Test parameters
CROSS_VAL = False
N_ENSEMBLE = 1
NB_HPD_LEVELS = 11
ALPHA = 0.05

# Test statistics
ALL_METRICS = ["accuracy", "mse"]  # , "div"]

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="lc2st")
parser.add_argument(
    "--task",
    type=str,
    default="two_moons",
    choices=[
        "two_moons",
        "slcp",
        "gaussian_mixture",
        "gaussian_linear_uniform",
        "bernoulli_glm",
        "bernoulli_glm_raw",
    ],
)

parser.add_argument(
    "--observations",
    "-o",
    type=str,
    default="task",
    choices=["task", "empirical"],
)
parser.add_argument(
    "--sbibm_obs",
    action="store_true",
    help="Use observations from sbibm for empirical exp.",
)
args = parser.parse_args()

# Task path
PATH_EXPERIMENT = Path(f"saved_experiments/lc2st_2023/exp_2/{args.task}")

# Get task (prior, simulator)
task = sbibm.get_task(args.task)
simulator = task.get_simulator()
prior = task.get_prior()

# Get reference observations
if args.observations == "task":
    NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Loading observations {NUM_OBSERVATION_LIST}")
    observation_list = [
        task.get_observation(num_observation=n_obs) for n_obs in NUM_OBSERVATION_LIST
    ]
    print()
    observation_dict = dict(zip(NUM_OBSERVATION_LIST, observation_list))
    TEST_SIZE = len(NUM_OBSERVATION_LIST)
    save_load_data = True
    task_observations = True
elif args.observations == "empirical":
    if args.sbibm_obs:
        NUM_OBSERVATION_LIST = list(range(1, 101))
        print(f"Loading observations {NUM_OBSERVATION_LIST}")
        TEST_SIZE = len(NUM_OBSERVATION_LIST)
        task.num_observations = TEST_SIZE
        task.observation_seeds = task.observation_seeds + [
            1000000 + i + 5 for i in range(10, TEST_SIZE)
        ]

        if "bernoulli_glm" in args.task:
            task._setup()
        else:
            task._setup(create_reference=False)
        observation_list = [
            task.get_observation(num_observation=n_obs)
            for n_obs in NUM_OBSERVATION_LIST
        ]
        print()
        observation_dict = dict(zip(NUM_OBSERVATION_LIST, observation_list))
        task_observations = True

    else:
        TEST_SIZE = 100
        theta_test = prior(TEST_SIZE)
        x_test = simulator(theta_test)
        observation_dict = {i + 1: x_test[i][None, :] for i in range(len(x_test))}
        task_observations = False
    save_load_data = False


# Data parameters
n_cal = 10000
n_eval = 10000
dim_theta = prior(num_samples=1).shape[-1]

# Path-params to save results for given test params
test_params = f"alpha_{ALPHA}_n_trials_null_{0}"
eval_params = f"n_eval_{n_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"

# Classifier parameters
sbibm_kwargs = sbibm_clf_kwargs(ndim=dim_theta)

# kwargs for c2st_scores function
kwargs_c2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "clf_kwargs": sbibm_kwargs,
}
# kwargs for lc2st_scores function
kwargs_lc2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "single_class_eval": True,
    "clf_kwargs": sbibm_kwargs,
}

# kwargs for lhpd_scores function
# same as in the original code from zhao et al https://github.com/zhao-david/CDE-diagnostics
lhpd_clf_kwargs = {"alpha": 0, "max_iter": 25000}
kwargs_lhpd = {
    "n_alphas": NB_HPD_LEVELS,
    "n_ensemble": N_ENSEMBLE,
    "clf_kwargs": lhpd_clf_kwargs,
}

# Paths to save results
result_path = PATH_EXPERIMENT / "scatter_plots"
if not os.path.exists(result_path):
    os.makedirs(result_path)

if os.path.exists(result_path / f"results_{args.method}_dict_{args.observations}.pkl"):
    results_dict = torch.load(result_path / f"results_{args.method}_dict_{args.observations}.pkl")
else:
    results_n_train_path = Path(f"results") / test_params / eval_params
    if args.observations == "empirical":
        results_n_train_path = results_n_train_path / "empirical"

    # Load or generate data
    data_path = PATH_EXPERIMENT / "scatter_plots" / f"data_{args.observations}_obs.pkl"
    if os.path.exists(data_path):
        data_samples = torch.load(data_path)
    else:
        data_samples = generate_data_one_run(
            n_cal=n_cal,
            n_eval=n_eval,
            task=task,
            observation_dict=observation_dict,
            # n_train_list=[args.n_train],
            n_train_list=[100, 1000, 10000, 100000],
            task_path=PATH_EXPERIMENT,
            save_data=save_load_data,  # save data to disk
            load_cal_data=save_load_data,  # load calibration data from disk
            load_eval_data=save_load_data,  # load evaluation data from disk
            seed=RANDOM_SEED,  # fixed seed for reproducibility
            task_observations=task_observations,
        )
        torch.save(data_samples, data_path)


    # Compute/ load and plot results

    # methods to compute results for
    methods = ["c2st", "lc2st", "lc2st_nf"]
    observation_num_list = list(data_samples["npe_obs"]["cal"][100].keys())
    observation_list = [observation_dict[num_obs] for num_obs in observation_num_list]
    observation_dict = dict(zip(observation_num_list, observation_list))

    # Compute test statistics
    results_dict = {}
    for c, n_train in enumerate([100, 1000, 10000, 100000]):
        results_dict[n_train], train_runtime_n = compute_test_results_npe_one_run(
            alpha=ALPHA,
            data_samples=data_samples,
            n_train=n_train,
            observation_dict=observation_dict,
            kwargs_c2st=kwargs_c2st,
            kwargs_lc2st=kwargs_lc2st,
            kwargs_lhpd=kwargs_lhpd,
            n_trials_null=0,
            t_stats_null_c2st_nf=None,
            t_stats_null_lc2st_nf=None,
            t_stats_null_lhpd=None,
            t_stats_null_dict_npe={m: None for m in methods},
            task_path=PATH_EXPERIMENT,
            results_n_train_path=results_n_train_path,
            methods=methods,
            test_stat_names=ALL_METRICS,
            compute_under_null=False,  # no type I error computation
            save_results=True,  # save results to disk
            seed=RANDOM_SEED,
        )
    torch.save(results_dict, result_path / f"results_{args.method}_dict_{args.observations}.pkl")

# plot identity line
plt.plot([0, 1], [0, 1], "k--")

# Plot results
for c, n_train in enumerate([100, 1000, 10000, 100000]):

    x = results_dict[n_train]["c2st"]["t_stat"]["mse"]
    y = results_dict[n_train][args.method]["t_stat"]["mse"]

    # pearson test statistic
    pearson_res = stats.pearsonr(x, y)
    print(
        f"Pearson correlation coeff. and p-value: {pearson_res.statistic}, {pearson_res.pvalue}"
    )
    print()

    # plot scatter plot for n_train
    plt.scatter(
        x,
        y,
        label=rf"${n_train}$",
        marker=METHODS_DICT[args.method]["marker"],
        color=METHODS_DICT[args.method]["colors"](c + 2),
        s=100,  # marker size
        alpha=0.7,  # marker transparency
    )
plt.xticks([0, 0.125, 0.25], ["0", "0.125", "0.25"])
plt.yticks([0, 0.125, 0.25], ["0", "0.125", "0.25"])
plt.xlim(-0.01, 0.26)
plt.ylim(-0.01, 0.26)
plt.xlabel(METHODS_DICT["c2st"])
plt.ylabel(METHODS_DICT[args.method]["name"])
plt.legend(title=r"$N_{\mathrm{train}}$ (for NPE)", loc="upper left")

if args.task == "two_moons":
    title = "Two Moons"
elif args.task == "slcp":
    title = "SLCP"
elif args.task == "gaussian_mixture":
    title = "Gaussian Mixture"
elif args.task == "gaussian_linear_uniform":
    title = "Gaussian Linear Uniform"
elif args.task == "bernoulli_glm":
    title = "Bernoulli GLM"
elif args.task == "bernoulli_glm_raw":
    title = "Bernoulli GLM Raw"
else:
    raise NotImplementedError(f"Task {args.task} not implemented.")
plt.title(title + "\n" + f"({TEST_SIZE} {args.observations} observations)")
plt.tight_layout()

plt.savefig(
    result_path
    / f"scatter_plot_mse_{args.task}_{args.observations}_c2st_{args.method}.pdf"
)
plt.show()
