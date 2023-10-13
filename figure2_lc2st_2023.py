# =============================================================================

#       SCRIPT TO REPRODUCE FIGURE 2 IN PAPER
#         (+ additional runtime experiment in Appendix A.5)

# =============================================================================

# DESCRIPTION: Compare different validation methods on SBIBM tasks for NPEs.
# > SBIBM tasks: two moons, slcp
#
# > Validation methods to compare:
#   >> REFERENCE for toy-examples when the true posterior is known (not amortized)
#       - oracle C2ST (vanilla) - permutation method
#       - oracle C2ST (MSE) - permutation method
#   >> OUR METHOD: when the true posterior is not known (amortized and single-class-eval)
#       - L-C2ST / LC2ST-NF (MSE) (permutation method / pre-computed null)
#       - local-HPD [Zhao et al. 2018] (pre-computed null)
# > Experiments to evaluate / compare the methods (on average over all observations x_0 from sbibm tasks):
#   - exp 1: t_stats / power as a function of N_train (at fixed N_cal)
#   - exp 2: power / type 1 error as a function of the number of n_cal (at fixed N_train)
#   - exp 3: run-time for computing the test statistic for one observation (at fixed N_cal and N_train)

# USAGE:
# >> python figure2_lc2st_2023.py --t_res_ntrain --n_train 100 1000 10000 100000
# >> python figure2_lc2st_2023.py --t_res_ntrain --n_train 100 1000 10000 100000 --power_ntrain
# >> python figure2_lc2st_2023.py --power_ncal --n_cal 100 500 1000 2000 5000 10000
# >> python figure2_lc2st_2023.py --runtime -nt 0 --n_cal 5000 10000 --task slcp
# >> python figure2_lc2st_2023.py --plot

# ====== Imports ======

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sbibm
import torch

from experiment_utils_sbibm import (
    l_c2st_results_n_train,
    compute_emp_power_l_c2st,
    compute_rejection_rates_from_pvalues_over_runs_and_observations,
    compute_average_rejection_rates,
)
from lc2st.c2st import sbibm_clf_kwargs  # , t_stats_c2st
from pathlib import Path
from plots_lc2st2023 import plot_sbibm_results_n_train_n_cal

# ====== GLOBAL PARAMETERS ======

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Path to save results
PATH_EXPERIMENT = Path("saved_experiments/lc2st_2023/exp_2")

# Methods to compare (labels for plots)
METHODS_ACC = [
    r"oracle C2ST ($\hat{t}_{Acc}$)",
    # r"$\ell$-C2ST ($\hat{t}_{Max0}$)",
    # r"$\ell$-C2ST-NF ($\hat{t}_{Max0}$)",
    # r"$\ell$-C2ST-NF-perm ($\hat{t}_{Max0}$)",
]
METHODS_L2 = [
    # r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)",
    r"$\ell$-C2ST ($\hat{t}_{\mathrm{MSE}_0}$)",
    r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)",
    # r"$\ell$-C2ST-NF-perm ($\hat{t}_{\mathrm{MSE}_0}$)",
    r"$local$-HPD",
]
METHODS_ALL = [
    # r"oracle C2ST ($\hat{t}_{Acc}$)",
    # r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)",
    # r"$\ell$-C2ST ($\hat{t}_{Max0}$)",
    # r"$\ell$-C2ST-NF ($\hat{t}_{Max0}$)",
    # r"$\ell$-C2ST-NF-perm ($\hat{t}_{Max0}$)",
    r"$\ell$-C2ST ($\hat{t}_{\mathrm{MSE}_0}$)",
    r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)",
    # r"$\ell$-C2ST-NF-perm ($\hat{t}_{\mathrm{MSE}_0}$)",
    r"$local$-HPD",
]

# Numbers of the observations x_0 from sbibm to evaluate the tests at
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Test parameters
ALPHA = 0.05
N_TRIALS_PRECOMPUTE = 100
N_RUNS = 50
NB_HPD_LEVELS = 11

# how to handle testing for multiple observations in empirical experiments
# ==> we compute rejection rates over runs for each observation seperately
# and then plot the mean/std over observations
BONFERONNI = False  # the goal here is to show how the `local` tests perform for a `fixed` observation ...
MEAN_RUNS = True  # no std over runs (naturally big for small n_cal or big n_train)
MEAN_OBS = False  # ... and the test result can be different for different observations

# Test statistics
ALL_METRICS = ["accuracy", "mse"]  # , "div"]

# Classifier parameters
CROSS_VAL = False
N_ENSEMBLE = 1

# Experiment parameters
N_CAL_EXP1 = 10000
N_TRAIN_EXP2 = 1000

# ====== Parse arguments======

parser = argparse.ArgumentParser()

# Data parameters
parser.add_argument(
    "--n_cal",
    nargs="+",
    type=int,
    default=[
        10000
    ],  # Use default for exp 1. Use [100, 500, 1000, 2000, 5000, 10000] for exp 2.
    help="Number of calibration samples to train (L)-C2ST on. Can be a list of integers.",
)

parser.add_argument(
    "--n_eval",
    type=int,
    default=10000,
    help="Number of evaluation samples for (L)-C2ST.",
)

# Test parameters
parser.add_argument(
    "--n_trials_null",
    "-nt",
    type=int,
    default=100,
    help="Number of trials to estimate the distribution of the test statistic under the null.",
)

# Experiment parameters
parser.add_argument(
    "--task",
    type=str,
    default="two_moons",
    choices=[
        "two_moons",
        "slcp",
        "gaussian_linear_uniform",
        "gaussian_mixture",
        "bernoulli_glm",
        "bernoulli_glm_raw",
    ],
    help="Task from sbibm to perform the experiment on.",
)

parser.add_argument(
    "--n_train",
    nargs="+",
    type=int,
    default=[
        1000
    ],  # Use default for exp 2. Use [100, 1_000, 10_000, 100_000] for exp 1.
    help="Number of training samples used to train the NPE. Can be a list of integers.",
)

parser.add_argument(
    "--t_res_ntrain",
    action="store_true",
    help="Exp 1a: Results as a function of N_train (at fixed N_cal=10_000).",
)

parser.add_argument(
    "--power_ntrain",
    action="store_true",
    help="Exp 1b: Plot the the empirical power / type 1 error as a function N_trian (at fixed N_cal=10_000).",
)

parser.add_argument(
    "--power_ncal",
    action="store_true",
    help="Exp 2: Plot the the empirical power / type 1 error as a function N_cal (at fixed N_train=100_000).",
)

parser.add_argument(
    "--runtime",
    "-r",
    action="store_true",
    help="Runtime for every method to compute the test statistic for one observation.",
)

parser.add_argument(
    "--plot", "-p", action="store_true", help="Plot results only.",
)

parser.add_argument(
    "--box_plots",
    "-b",
    action="store_true",
    help="Plot Box-plots for every observation.",
)

# ====== Experiments ======

# Parse arguments
args = parser.parse_args()

print()
print("=================================================")
print("  VALIDATION METHOD COMPARISON for sbibm-tasks")
print("=================================================")
print()

# Add oracle C2ST for all tasks
METHODS_L2.append(r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)")
METHODS_ALL.append(r"oracle C2ST ($\hat{t}_{\mathrm{MSE}}$)")
METHODS_ALL.append(r"oracle C2ST ($\hat{t}_{Acc}$)")

# Define task and path
task = sbibm.get_task(args.task)
task_path = PATH_EXPERIMENT / args.task

# SBI set-up for given task: prior, simulator, inference algorithm
prior = task.get_prior()
simulator = task.get_simulator()
algorithm = "npe"
if algorithm != "npe":
    raise NotImplementedError("Only NPE is supported for now.")
print(f"Task: {args.task} / Algorithm: {algorithm}")
print()

# Load observations
print(f"Loading observations {NUM_OBSERVATION_LIST}")
observation_list = [
    task.get_observation(num_observation=n_obs) for n_obs in NUM_OBSERVATION_LIST
]
print()
observation_dict = dict(zip(NUM_OBSERVATION_LIST, observation_list))

# Test set-up
# Dataset sizes
n_cal_list = args.n_cal
n_eval = args.n_eval

dim_theta = prior(num_samples=1).shape[-1]

# Path-params to save results for given test params
test_params = f"alpha_{ALPHA}_n_trials_null_{args.n_trials_null}"
eval_params = f"n_eval_{n_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"
results_path = (
    task_path
    / "results"
    / test_params
    / eval_params
    / f"bonferonni_{BONFERONNI}_mean_obs_{MEAN_OBS}_mean_runs_{MEAN_RUNS}"
)
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Classifier parameters
sbibm_kwargs = sbibm_clf_kwargs(ndim=dim_theta)

# kwargs for (l)c2st_scores function
kwargs_l_c2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
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

# pre-compute / load test statistics for the C2ST-NF null hypothesis
# they are independant of the estimator and the observation space (x)
# N.B> L-C2ST is still dependent on the observation space (x)
# as its trained on the joint samples (theta, x)
t_stats_null_c2st_nf = {ncal: None for ncal in n_cal_list}
# from lc2st.test_utils import precompute_t_stats_null
# if not args.plot:
#     for n_cal in n_cal_list:
#         t_stats_null_c2st_nf[n_cal] = precompute_t_stats_null(
#             metrics=ALL_METRICS,
#             n_cal=n_cal,
#             n_eval=n_eval,
#             dim_theta=dim_theta,
#             n_trials_null=N_TRIALS_PRECOMPUTE,
#             t_stats_null_path=task_path / "t_stats_null" / eval_params,
#             methods=["c2st_nf"],
#             t_stats_fn_c2st=t_stats_c2st,
#             kwargs_c2st=kwargs_c2st,
#             save_results=True,
#             load_results=True,
#             # args for lc2st only
#             kwargs_lc2st=None,
#             kwargs_lhpd=None,
#             x_cal=None,
#             observation_dict=None,
#         )["c2st_nf"]


# ====== EXP 1: test stats as a function of N_train (N_cal fixed) ======
if args.t_res_ntrain:
    # Get experiment parameters
    n_cal = N_CAL_EXP1
    n_train_list = args.n_train

    print()
    print(
        f"Experiment 1: test statistics as a function of N_train in {n_train_list} ..."
    )
    print(f"... for N_cal = {n_cal}")
    print()

    methods = ["lc2st", "lc2st_nf", "lhpd", "c2st"]

    # Compute test statistics for every n_train
    results_n_train, train_runtime = l_c2st_results_n_train(
        task,
        n_cal=n_cal,
        n_eval=n_eval,
        observation_dict=observation_dict,
        n_train_list=n_train_list,
        alpha=ALPHA,
        n_trials_null=args.n_trials_null,
        t_stats_null_c2st_nf=t_stats_null_c2st_nf[n_cal],
        n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
        kwargs_c2st=kwargs_l_c2st,
        kwargs_lc2st=kwargs_l_c2st,
        kwargs_lhpd=kwargs_lhpd,
        task_path=task_path,
        t_stats_null_path=task_path / "t_stats_null" / eval_params,
        results_n_train_path=Path(f"results") / test_params / eval_params,
        methods=methods,
        test_stat_names=ALL_METRICS,
        seed=RANDOM_SEED,
    )

    # Compute TPR for every n_train
    if args.power_ntrain:
        # Dictionary to load results from a given run
        # two moons
        if args.task == "two_moons":
            methods_dict = {
                "c2st": {n: 100 for n in n_train_list},
                "lc2st": {100: 65, 1000: 69, 10000: 56, 100000: 85},
                "lc2st_nf": {100: 56, 1000: 50, 10000: 67, 100000: 66},
                # "lc2st_nf_perm": {100: 56, 1000: 50, 10000: 35, 100000: 35},
                "lhpd": {100: 52, 1000: 54, 10000: 53, 100000: 65},
            }

        # slcp
        elif args.task == "slcp":
            methods_dict = {
                "c2st": {100: 59, 1000: 55, 10000: 76, 100000: 59,},
                "lc2st": {100: 52, 1000: 50, 10000: 60, 100000: 94},
                "lc2st_nf": {100: 52, 1000: 55, 10000: 54, 100000: 62},
                # "lc2st_nf_perm": {100: 27, 1000: 16, 10000: 35, 100000: 37},
                "lhpd": {100: 53, 1000: 50, 10000: 55, 100000: 50},
            }
        elif args.task == "gaussian_mixture":
            methods_dict = {
                "c2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st_nf": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lhpd": {100: 50, 1000: 50, 10000: 50, 100000: 50},
            }
        elif args.task == "gaussian_linear_uniform":
            methods_dict = {
                "c2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st_nf": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lhpd": {100: 50, 1000: 50, 10000: 50, 100000: 50},
            }
        elif args.task == "bernoulli_glm":
            methods_dict = {
                "c2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st_nf": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lhpd": {100: 50, 1000: 50, 10000: 50, 100000: 50},
            }
        elif args.task == "bernoulli_glm_raw":
            methods_dict = {
                "c2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lc2st_nf": {100: 50, 1000: 50, 10000: 50, 100000: 50},
                "lhpd": {100: 50, 1000: 50, 10000: 50, 100000: 50},
            }
        else:
            raise NotImplementedError(f"Task {args.task} not implemented.")

        # Number of runs to compute the empirical power over
        # n_runs = N_RUNS
        n_runs_dict = {
            "two_moons": 50,
            "slcp": 50,
            "gaussian_mixture": 50,
            "gausiian_linear_uniform": 50,
            "bernoulli_glm": 50,
            "bernoulli_glm_raw": 50,
        }
        n_runs = n_runs_dict[args.task]

        # Initialize dictionaries to store the results
        emp_power_dict, type_I_error_dict = (
            {
                n: {
                    m: {t_stat_name: [] for t_stat_name in ALL_METRICS}
                    for m in methods_dict.keys()
                }
                for n in n_train_list
            },
            {
                n: {
                    m: {t_stat_name: [] for t_stat_name in ALL_METRICS}
                    for m in methods_dict.keys()
                }
                for n in n_train_list
            },
        )
        p_values_dict, p_values_h0_dict = (
            {n: {m: None for m in methods_dict.keys()} for n in n_train_list},
            {n: {m: None for m in methods_dict.keys()} for n in n_train_list},
        )

        # Compute / Load p_values of every run for every n_train
        for m, n_train_run_dict in methods_dict.items():

            for n_train in n_train_list:
                (_, _, p_values, _,) = compute_emp_power_l_c2st(
                    n_runs=n_runs,
                    alpha=ALPHA,
                    task=task,
                    n_train=n_train,
                    observation_dict=observation_dict,
                    n_cal_list=[n_cal],
                    n_eval=n_eval,
                    n_trials_null=args.n_trials_null,
                    kwargs_c2st=kwargs_l_c2st,
                    kwargs_lc2st=kwargs_l_c2st,
                    kwargs_lhpd=kwargs_lhpd,
                    t_stats_null_c2st_nf=None,
                    n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
                    methods=[m],
                    test_stat_names=ALL_METRICS,
                    compute_emp_power=True,
                    compute_type_I_error=False,
                    task_path=task_path,
                    load_eval_data=True,
                    result_path=task_path
                    / f"npe_{n_train}"
                    / "results"
                    / test_params
                    / eval_params,
                    t_stats_null_path=task_path / "t_stats_null" / eval_params,
                    results_n_train_path=Path(f"results") / test_params / eval_params,
                    n_run_load_results=n_train_run_dict[n_train],
                    # save_every_n_runs=10,
                )
                p_values_dict[n_train][m] = p_values[n_cal][m]

                # (1) Compute test result (rejection or not) over runs for each observation seperately
                for t_stat_name in ALL_METRICS:
                    if m == "lhpd" and t_stat_name != "mse":
                        continue
                    (
                        emp_power_dict[n_train][m][t_stat_name],
                        _,
                    ) = compute_rejection_rates_from_pvalues_over_runs_and_observations(
                        p_values_dict=p_values_dict[n_train][m][t_stat_name],
                        alpha=ALPHA,
                        n_runs=n_runs,
                        num_observation_list=NUM_OBSERVATION_LIST,
                        compute_tpr=True,
                        compute_fpr=False,
                        p_values_h0_dict=None,
                        bonferonni_correction=BONFERONNI,
                    )

        # (2) Compute the mean/std ...
        for i, n_train in enumerate(n_train_list):
            for m in methods_dict.keys():
                if i == 0:
                    results_n_train[m]["TPR_mean"] = {
                        t_stat_name: [] for t_stat_name in ALL_METRICS
                    }
                    results_n_train[m]["TPR_std"] = {
                        t_stat_name: [] for t_stat_name in ALL_METRICS
                    }
                for t_stat_name in ALL_METRICS:
                    result_list = compute_average_rejection_rates(
                        emp_power_dict[n_train][m][t_stat_name],
                        mean_over_runs=MEAN_RUNS,
                        mean_over_observations=MEAN_OBS,
                    )
                    results_n_train[m]["TPR_mean"][t_stat_name].append(
                        np.mean(result_list)
                    )
                    results_n_train[m]["TPR_std"][t_stat_name].append(
                        np.std(result_list)
                    )

        # Save results
        torch.save(emp_power_dict, results_path / f"emp_power_dict_n_train.pkl")
        torch.save(results_n_train, results_path / f"avg_results_n_train.pkl")


if args.power_ncal:
    # Get experiment parameters
    n_train = N_TRAIN_EXP2
    n_cal_list = args.n_cal

    print()
    print(f"Experiment 2: Empirical Power as a function of N_cal in {n_cal_list} ...")

    print(f"... for N_train = {n_train}")
    print()

    # Path to save results
    npe_result_path = (
        task_path / f"npe_{n_train}" / "results" / test_params / eval_params
    )
    if not os.path.exists(npe_result_path):
        os.makedirs(npe_result_path)

    # Dictionary to load results from a given run
    # two moons
    if args.task == "two_moons":
        methods_dict = {
            "c2st": {n: 100 for n in [100, 500, 1000, 2000, 5000, 10000]},
            "lc2st": {100: 100, 500: 100, 1000: 100, 2000: 100, 5000: 100, 10000: 69},
            "lc2st_nf": {100: 67, 500: 67, 1000: 67, 2000: 100, 5000: 65, 10000: 50},
            # "lc2st_nf_perm": {100: 67, 500: 67, 1000: 67, 2000: 100, 5000: 65, 10000: 50},
            "lhpd": {100: 51, 500: 100, 1000: 61, 2000: 71, 5000: 53, 10000: 54,},
        }

    # slcp
    elif args.task == "slcp":
        methods_dict = {
            "c2st": {100: 77, 500: 77, 1000: 77, 2000: 52, 5000: 56, 10000: 55,},
            "lc2st": {100: 100, 500: 100, 1000: 100, 2000: 100, 5000: 100, 10000: 50},
            "lc2st_nf": {100: 64, 500: 64, 1000: 64, 2000: 100, 5000: 62, 10000: 55},
            # "lc2st_nf_perm": {
            #     100: 64,
            #     500: 64,
            #     1000: 64,
            #     2000: 100,
            #     5000: 40,
            #     10000: 16,
            # },
            "lhpd": {100: 88, 500: 52, 1000: 50, 2000: 50, 5000: 50, 10000: 50,},
        }
    elif args.task == "gaussian_mixture":
        methods_dict = {
            "c2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st_nf": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lhpd": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
        }
    elif args.task == "gaussian_linear_uniform":
        methods_dict = {
            "c2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st_nf": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lhpd": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
        }
    elif args.task == "bernoulli_glm":
        methods_dict = {
            "c2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st_nf": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lhpd": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
        }
    elif args.task == "bernoulli_glm_raw":
        methods_dict = {
            "c2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lc2st_nf": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
            "lhpd": {100: 50, 500: 50, 1000: 50, 2000: 50, 5000: 50, 10000: 50},
        }
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # Number of runs to compute the empirical power over
    # n_runs = N_RUNS
    n_runs_dict = {
        "two_moons": 50,
        "slcp": 50,
        "gaussian_mixture": 50,
        "gausiian_linear_uniform": 50,
        "bernoulli_glm": 50,
        "bernoulli_glm_raw": 50,
    }
    n_runs = n_runs_dict[args.task]

    # Initialize dictionaries to store the results
    emp_power_dict, type_I_error_dict = (
        {
            n: {
                m: {t_stat_name: [] for t_stat_name in ALL_METRICS}
                for m in methods_dict.keys()
            }
            for n in n_cal_list
        },
        {
            n: {
                m: {t_stat_name: [] for t_stat_name in ALL_METRICS}
                for m in methods_dict.keys()
            }
            for n in n_cal_list
        },
    )
    p_values_dict, p_values_h0_dict = (
        {n: {m: None for m in methods_dict.keys()} for n in n_cal_list},
        {n: {m: None for m in methods_dict.keys()} for n in n_cal_list},
    )

    # Compute / Load p_values of every run for every n_cal
    for m, n_cal_run_dict in methods_dict.items():
        for n_cal in n_cal_list:
            (_, _, p_values, p_values_h0,) = compute_emp_power_l_c2st(
                n_runs=n_runs,
                alpha=ALPHA,
                task=task,
                n_train=n_train,
                observation_dict=observation_dict,
                n_cal_list=[n_cal],
                n_eval=n_eval,
                n_trials_null=args.n_trials_null,
                kwargs_c2st=kwargs_l_c2st,
                kwargs_lc2st=kwargs_l_c2st,
                kwargs_lhpd=kwargs_lhpd,
                t_stats_null_c2st_nf=None,
                n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
                methods=[m],
                test_stat_names=ALL_METRICS,
                compute_emp_power=True,
                compute_type_I_error=True,
                task_path=task_path,
                load_eval_data=True,
                result_path=npe_result_path,
                t_stats_null_path=task_path / "t_stats_null" / eval_params,
                results_n_train_path=Path(f"results") / test_params / eval_params,
                n_run_load_results=n_cal_run_dict[n_cal],
                # save_every_n_runs=10,
            )
            p_values_dict[n_cal][m] = p_values[n_cal][m]
            p_values_h0_dict[n_cal][m] = p_values_h0[n_cal][m]

            # (1) Compute emp power over n_runs
            for t_stat_name in ALL_METRICS:
                if m == "lhpd" and t_stat_name != "mse":
                    continue
                (
                    emp_power_dict[n_cal][m][t_stat_name],
                    type_I_error_dict[n_cal][m][t_stat_name],
                ) = compute_rejection_rates_from_pvalues_over_runs_and_observations(
                    p_values_dict=p_values_dict[n_cal][m][t_stat_name],
                    p_values_h0_dict=p_values_h0_dict[n_cal][m][t_stat_name],
                    alpha=ALPHA,
                    n_runs=n_runs,
                    num_observation_list=NUM_OBSERVATION_LIST,
                    compute_tpr=True,
                    compute_fpr=True,
                    bonferonni_correction=BONFERONNI,
                )

    results_n_cal = {m: {} for m in methods_dict.keys()}
    # (2) Compute the mean/std ...
    for i, n_cal in enumerate(n_cal_list):
        for m in methods_dict.keys():
            for result_name, result_dict in zip(
                ["TPR", "FPR"], [emp_power_dict, type_I_error_dict]
            ):
                if i == 0:
                    results_n_cal[m][result_name + "_mean"] = {
                        t_stat_name: [] for t_stat_name in ALL_METRICS
                    }
                    results_n_cal[m][result_name + "_std"] = {
                        t_stat_name: [] for t_stat_name in ALL_METRICS
                    }
                for t_stat_name in ALL_METRICS:
                    result_list = compute_average_rejection_rates(
                        result_dict[n_cal][m][t_stat_name],
                        mean_over_runs=MEAN_RUNS,
                        mean_over_observations=MEAN_OBS,
                    )
                    results_n_cal[m][result_name + "_mean"][t_stat_name].append(
                        np.mean(result_list)
                    )
                    results_n_cal[m][result_name + "_std"][t_stat_name].append(
                        np.std(result_list)
                    )

    # Save results
    torch.save(emp_power_dict, results_path / f"emp_power_dict_n_cal.pkl")
    torch.save(type_I_error_dict, results_path / f"type_I_error_dict_n_cal.pkl")
    torch.save(results_n_cal, results_path / f"avg_results_n_cal.pkl")

# ====== Exp 3: RUNTIME ======
if args.runtime:
    try:
        dict_runtimes = torch.load(task_path / "results" / f"runtimes_appendix.pkl")
    except FileNotFoundError:
        methods_dict = {
            "c2st": "accuracy",
            "lc2st": "mse",
            "lc2st_nf": "mse",
            "lhpd": "mse",
        }
        n_train_list = [100, 1000, 10000, 100000]
        dict_runtimes = {
            m: {n_cal: [] for n_cal in args.n_cal} for m in methods_dict.keys()
        }
        for n_cal in args.n_cal:
            # Compute test statistics for every n_train
            results_n_train, train_runtime = l_c2st_results_n_train(
                task,
                n_cal=n_cal,
                n_eval=n_eval,
                observation_dict=observation_dict,
                n_train_list=n_train_list,
                alpha=ALPHA,
                n_trials_null=0,  # just look at runtime to compute test statistics (no null hypothesis)
                t_stats_null_c2st_nf=t_stats_null_c2st_nf[n_cal],
                n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
                kwargs_c2st=kwargs_l_c2st,
                kwargs_lc2st=kwargs_l_c2st,
                kwargs_lhpd=kwargs_lhpd,
                task_path=task_path,
                t_stats_null_path=task_path / "t_stats_null" / eval_params,
                results_n_train_path=Path(f"results") / test_params / eval_params,
                methods=methods_dict.keys(),
                test_stat_names=ALL_METRICS,
                seed=RANDOM_SEED,
            )

            # Add results to dictionary
            for m, t in methods_dict.items():
                print(f"Method: {m}")
                if "l" in m:
                    dict_runtimes[m][n_cal] = np.array(train_runtime[m])
                else:
                    dict_runtimes[m][n_cal] = np.array(
                        results_n_train[m]["run_time_mean"][t]
                    )

        # Save results
        torch.save(dict_runtimes, task_path / "results" / f"runtimes_appendix.pkl")
    print(dict_runtimes)

# ====== PLOTS ONLY ======
if args.plot:
    # Path to save figures
    fig_path = (
        task_path
        / "figures"
        / eval_params
        / test_params
        / f"bonferonni_{BONFERONNI}_mean_obs_{MEAN_OBS}_mean_runs_{MEAN_RUNS}"
    )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Load results
    results_n_train = torch.load(results_path / f"avg_results_n_train.pkl")
    results_n_cal = torch.load(results_path / f"avg_results_n_cal.pkl")

    n_train_list = [100, 1000, 10000, 100000]
    n_cal_list = [100, 500, 1000, 2000, 5000, 10000]

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

    # Plot results
    fig = plot_sbibm_results_n_train_n_cal(
        results_n_train=results_n_train,
        results_n_cal=results_n_cal,
        n_train_list=n_train_list,
        n_cal_list=n_cal_list,
        methods_mse=METHODS_L2,
        methods_all=METHODS_ALL,
        title=title,
    )
    # Save figure
    plt.savefig(
        fig_path
        / f"results_ntrain_1000_n_cal_10000_bonferonni_{BONFERONNI}_mean_over_obs_{MEAN_OBS}_mean_over_runs_{MEAN_RUNS}.pdf"
    )
    plt.show()

    if args.box_plots:
        # Box plots to show variability over runs for each observation seperately
        import seaborn as sns
        import pandas as pd

        # Dictionary to load results from a given run
        # two moons
        if args.task == "two_moons":
            methods_dict = {
                "c2st": {n: 100 for n in n_train_list},
                "lc2st": {100: 65, 1000: 69, 10000: 56, 100000: 85},
                "lc2st_nf": {100: 56, 1000: 50, 10000: 67, 100000: 66},
                # "lc2st_nf_perm": {100: 56, 1000: 50, 10000: 35, 100000: 35},
                "lhpd": {100: 52, 1000: 54, 10000: 53, 100000: 65},
            }

        # slcp
        elif args.task == "slcp":
            methods_dict = {
                "c2st": {100: 59, 1000: 55, 10000: 76, 100000: 59,},
                "lc2st": {100: 52, 1000: 50, 10000: 60, 100000: 94},
                "lc2st_nf": {100: 52, 1000: 55, 10000: 54, 100000: 62},
                # "lc2st_nf_perm": {100: 27, 1000: 16, 10000: 35, 100000: 37},
                "lhpd": {100: 53, 1000: 50, 10000: 55, 100000: 50},
            }
        else:
            raise NotImplementedError("Only two_moons and slcp are supported for now.")

        n_runs = min(m[n] for m in methods_dict.values() for n in n_train_list)

        emp_power_dict = torch.load(results_path / f"emp_power_dict_n_train.pkl")

        sns.set_theme(style="ticks", palette="pastel")

        for m in methods_dict.keys():
            for t_stat_name in ALL_METRICS:
                if "lc2st" in m and t_stat_name == "accuracy":
                    continue
                if "lhpd" in m and t_stat_name != "mse":
                    continue
                if t_stat_name == "div":
                    continue
                df = pd.DataFrame()
                list_n_train = []
                list_n_obs = []
                list_tpr = []
                for n_train in n_train_list:
                    for n_obs in NUM_OBSERVATION_LIST:
                        for n_r in range(n_runs):
                            print(emp_power_dict[n_train][m][t_stat_name][n_r])
                            list_n_train.append(n_train)
                            list_n_obs.append(n_obs)
                            list_tpr.append(
                                emp_power_dict[n_train][m][t_stat_name][n_r][n_obs - 1]
                            )
                df["n_train"] = list_n_train
                df["observation"] = list_n_obs
                df["TPR"] = list_tpr

                sns.boxplot(
                    x="n_train",
                    y="TPR",
                    hue="observation",
                    data=df,
                    showmeans=True,
                    meanline=True,
                    meanprops={"linestyle": "--", "linewidth": 2, "color": "black"},
                )
                sns.despine(offset=10, trim=True)
                plt.title(m + " " + t_stat_name)
                plt.ylim(0, 1)
                plt.savefig(task_path / "figures" / f"boxplot_{m}_{t_stat_name}.png")
                plt.show()
