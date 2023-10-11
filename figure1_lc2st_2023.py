# =============================================================================

#       SCRIPT TO REPRODUCE FIGURE 1 PAPER

# =============================================================================

# DESCRIPTION: Single class evaluation of the C2ST test statistics under distribution shift.
# > Tasks: Guassian (mean-shift), Gaussian (variance-shift), Student t (df-shift)
# > Experiments:
#   - exp 1: Compute the test statistics under distribution shift for optimal Bayes classifier
#   - exp 2: Compute the empirical power under distribution shift for estimated classifier

# USAGE:
# >> python figure1_lc2st_2023.py --opt_bayes --t_shift
# >> python figure1_lc2st_2023.py --power_shift
# >> python figure1_lc2st_2023.py --plot


# ====== IMPORTS ======

import argparse
import os
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.stats import multivariate_normal as mvn
from scipy.stats import t

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from lc2st.c2st import c2st_scores, t_stats_c2st

from classifiers.optimal_bayes import (
    opt_bayes_scores,
    AnalyticGaussianLQDA,
    AnalyticStudentClassifier,
)

from c2st_p_values_roc import c2st_p_values_tfpr
from lc2st.test_utils import eval_htest

from plots_lc2st2023 import plot_plot_c2st_single_eval_shift

# ====== GLOBAL PARAMETERS ======

# Path to save/load the results
PATH_EXPERIMENT = "saved_experiments/lc2st_2023/exp_1/"

# Data parameters
N_SAMPLES_EVAL = 10_000  # N_v (validation set size - used to compute the test statistics for a trained classifier)

# Test parameters
ALPHA = 0.05  # significance level
N_RUNS = 100  # number of runs to compute the power
N_TRIALS_NULL = 1000  # number of trials to estimate the null distribution
USE_PERMUTATION = (
    False  # whether to use the permutation method to estimate the null distribution
)

# Test statistics
METRICS = {
    "accuracy": [
        "acc_ref",
        "acc_single_class",
    ],
    "mse": [
        "mse_ref",
        "mse_single_class",
    ],
    "div": [
        "max_ref",
        "max_single_class",
    ],
}

# ====== Parse arguments ======

parser = argparse.ArgumentParser()

# Data parameters
parser.add_argument(
    "--dim",
    type=int,
    default=2,
    help="Dimension of the data (number of features).",
)

parser.add_argument(
    "--q_dist",
    "-q",
    type=str,
    default="variance",
    choices=["mean", "variance", "df"],
    help="Variable / shifted parameter in the distribution of Q.",
)

# Whether to use the optimal bayes classifier or not
parser.add_argument(
    "--opt_bayes",
    action="store_true",
    help="Whether to use the Optimal Bayes Classifier.",
)

# Experiment parameters

parser.add_argument(
    "--t_shift",
    action="store_true",
    help="Compute test statistics over distribution shifts.",
)

parser.add_argument(
    "--power_shift",
    action="store_true",
    help="Compute empirical power over distribution shifts.",
)

parser.add_argument(
    "--plot",
    "-p",
    action="store_true",
    help="Only plot the results of the experiments.",
)

args = parser.parse_args()

# ====== EXPERIMENT SETUP ======

# Dimension of the data (number of features)
dim = args.dim

# P - class 0 distribution: Standard Gaussian (fixed)
P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))

# Parameters for different experiments
if args.q_dist == "mean":
    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = \mathcal{N}(m, I)$"

    # N_cal (training set size)
    n_samples_list = [2000]

    # Mean-shift
    shifts = np.concatenate(
        [
            [-1, -0.5, -0.3],
            np.arange(-0.1, 0.0, 0.02),
            [0.0],
            np.arange(0.02, 0.12),
            [0.3, 0.5, 1],
        ]
    )

    # Q - class 1 distrribution: standard Gaussian with shifted mean
    Q_dist_list = [mvn(mean=np.array([mu] * dim), cov=np.eye(dim)) for mu in shifts]

    # Classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticGaussianLQDA(dim=dim, mu=mu) for mu in shifts]
    else:
        # estimated classifier
        clf_name = "LDA"
        clf_class = LinearDiscriminantAnalysis
        clf_kwargs = {"solver": "eigen", "priors": [0.5, 0.5]}


elif args.q_dist == "variance":
    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = \mathcal{N}(0,\sigma^2I)$"

    # N_cal (training set size)
    n_samples_list = [2000]

    # Variance-shift
    shifts = np.concatenate([[0.01], np.arange(0.1, 1.6, 0.1)])
    # shifts = np.array([0.01, 0.1, 0.5, 1, 1.5, 3])

    # Q - class 1 distrribution: centered Gaussian with shifted variance
    Q_dist_list = [mvn(mean=np.zeros(dim), cov=s * np.eye(dim)) for s in shifts]

    # Classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticGaussianLQDA(dim=dim, sigma=s) for s in shifts]
    else:
        # estimated classifier
        clf_name = "QDA"
        clf_class = QuadraticDiscriminantAnalysis
        clf_kwargs = {"priors": [0.5, 0.5]}

elif args.q_dist == "df":
    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = t(df)$"

    # N_cal (training set size)
    n_samples_list = [2000]

    # df-shift (degrees of freedom)
    shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35]

    # Q - class 1 distrribution: standard Student with shifted degrees of freedom
    Q_dist_list = [t(df=df, loc=0, scale=1) for df in shifts]

    # Classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticStudentClassifier(df=df) for df in shifts]
    else:
        # estimated classifier
        clf_name = "MLP"
        clf_class = MLPClassifier
        clf_kwargs = {"alpha": 0, "max_iter": 25000}
else:
    raise NotImplementedError

# List of test statistics names
test_stat_names = [item for sublist in list(METRICS.values()) for item in sublist]

# ====== EXP 1: T_STATS UNDER DISTRIBUTION SHIFT ======

if args.t_shift and not args.plot:
    print()
    print("=================================================")
    print("EXP 1: COMPUTE T_STATS UNDER DISTRIBUTION SHIFT")
    print("       ", h0_label)
    print("       Classifier: ", clf_name)
    print("=================================================")
    print()
    test_stats = dict(zip(test_stat_names, [[] for _ in test_stat_names]))
    cal_curves = {"oracle": [], "single_class": []}

    # Generate data from P
    P_eval = P_dist.rvs(size=N_SAMPLES_EVAL)

    # Loop over shifts
    for i, (s, Q_dist) in enumerate(zip(shifts, Q_dist_list)):
        # Compute test statistics
        for b in [True, False]:  # single class eval vs. not
            # Generate data from Q
            Q_eval = Q_dist.rvs(size=N_SAMPLES_EVAL)
            if args.opt_bayes:
                # Evaluate the *optimal Bayes* classifier (no training)
                scores = opt_bayes_scores(
                    P=P_eval,
                    Q=Q_eval,
                    clf=clf_list[i],
                    metrics=list(METRICS.keys()),
                    single_class_eval=b,
                )
            else:
                # Generate training data
                P = P_dist.rvs(size=n_samples_list[0])
                Q = Q_dist.rvs(size=n_samples_list[0])
                if dim == 1:
                    P = P.reshape(-1, 1)
                    Q = Q.reshape(-1, 1)
                    P_eval = P_eval.reshape(-1, 1)
                    Q_eval = Q_eval.reshape(-1, 1)
                # Train and evaluate the *extimated* classifier
                scores, probas = t_stats_c2st(
                    P=P,
                    Q=Q,
                    cross_val=False,
                    P_eval=P_eval,
                    Q_eval=Q_eval,
                    clf_class=clf_class,
                    clf_kwargs=clf_kwargs,
                    metrics=list(METRICS.keys()),
                    single_class_eval=b,
                    null_hypothesis=False,
                    return_probas=True,
                )
            # Append scores to dict
            for metric, t_names in zip(METRICS.keys(), METRICS.values()):
                if b:
                    name = t_names[1]
                else:
                    name = t_names[0]
                test_stats[name].append(scores[metric])

    # Save computed test statistics
    torch.save(
        test_stats,
        PATH_EXPERIMENT + f"test_stats_{clf_name}_shift_{args.q_dist}_dim_{dim}.pkl",
    )


# ====== EXP 2: EMPIRICAL POWER UNDER DISTRIBUTION SHIFT  ======

if args.power_shift and not args.plot:
    print()
    print("========================================================")
    print("EXP 2: COMPUTE EMPIRICAL POWER UNDER DISTRIBUTION SHIFT")
    print("       ", h0_label)
    print("       Classifier: ", clf_name)
    print("========================================================")
    print()

    # Initialize test statistics function
    t_stats_c2st_custom = partial(
        t_stats_c2st,
        scores_fn=c2st_scores,
        metrics=list(METRICS.keys()),
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        # args for scores_fn
        cross_val=False,
    )

    # Pre-compute test statistics under the null distribution
    scores_null_list = []
    for n in n_samples_list:
        print()
        print(f"N_cal = {n}")
        if not USE_PERMUTATION:
            print()
            print(
                "Pre-computing or loading the test statistics under the null distribution."
                + "\n They will be reused at every test-run. The permutation method is not needed."
            )
            print()

            scores_null = dict(zip([True, False], [[], []]))

            filename = f"nt_{N_TRIALS_NULL}_N_{n}_dim_{dim}_{clf_name}.npy"
            if os.path.exists(PATH_EXPERIMENT + "t_stats_null/" + filename):
                # Load null scores if they exist ...
                scores_null = np.load(
                    PATH_EXPERIMENT + "t_stats_null/" + filename,
                    allow_pickle=True,
                ).item()
            else:
                # ... otherwise, compute them
                # Generate training and evalaluation data from P for every trial
                list_P_null = [P_dist.rvs(n) for _ in range(2 * N_TRIALS_NULL)]
                list_P_eval_null = [
                    P_dist.rvs(N_SAMPLES_EVAL) for _ in range(2 * N_TRIALS_NULL)
                ]
                if dim == 1:
                    for i in range(2 * N_TRIALS_NULL):
                        list_P_null[i] = list_P_null[i].reshape(-1, 1)
                        list_P_eval_null[i] = list_P_eval_null[i].reshape(-1, 1)

                # Compute test statistics under the null
                for b in [True, False]:  # single class eval vs. not
                    t_stats_null = t_stats_c2st_custom(
                        null_hypothesis=True,
                        n_trials_null=N_TRIALS_NULL,
                        list_P_null=list_P_null,
                        list_P_eval_null=list_P_eval_null,
                        use_permutation=False,
                        # args for scores_fn
                        single_class_eval=b,
                        # unnecessary, but needed inside `t_stats_c2st`
                        P=list_P_null[0],
                        Q=list_P_null[1],
                        P_eval=list_P_eval_null[0],
                        Q_eval=list_P_eval_null[1],
                    )
                    scores_null[b] = t_stats_null

                # Save null scores
                if not os.path.exists(PATH_EXPERIMENT + "t_stats_null/"):
                    os.makedirs(PATH_EXPERIMENT + "t_stats_null/")
                np.save(
                    PATH_EXPERIMENT + "t_stats_null/" + filename,
                    scores_null,
                )

        else:
            print()
            print(
                f"Not pre-computing the test-statistics under the null."
                + "\n Using the permutation method to estimate them at each test run."
            )
            print()
            scores_null = {True: None, False: None}
        scores_null_list.append(scores_null)

    # Define function to evaluate the test
    eval_c2st = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st_custom,
        verbose=False,
        metrics=list(METRICS.keys()),
    )

    # Get the number of samples for training
    n = n_samples_list[0]
    # Get the null scores
    scores_null = scores_null_list[0]

    try:
        # Load TPR dicts if they exist ...
        TPR_list = torch.load(
            PATH_EXPERIMENT
            + f"TPR_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl",
        )
        TPR_std_list = torch.load(
            PATH_EXPERIMENT
            + f"TPR_std_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl",
        )
        print("Loaded Results.")
        print()
    except FileNotFoundError:
        # ... otherwise, compute them
        # Initialize TPR dicts
        TPR_list, TPR_std_list = (
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
        )

        # Loop over shifts
        for i, (s, Q_dist) in enumerate(zip(shifts, Q_dist_list)):
            print()
            print(f"{args.q_dist} shift: {np.round(s,2)}")
            print()

            # Compute empirical power (TPR)
            for b in [True, False]:  # single class eval vs. not
                TPR, _, TPR_std, _, _, _ = c2st_p_values_tfpr(
                    eval_c2st_fn=partial(
                        eval_c2st, single_class_eval=b, n_trials_null=N_TRIALS_NULL
                    ),
                    n_runs=N_RUNS,
                    n_samples={"train": n, "eval": N_SAMPLES_EVAL},
                    alpha_list=[ALPHA],
                    P_dist=P_dist,
                    Q_dist=Q_dist,
                    metrics=list(METRICS.keys()),
                    metrics_cv=[],
                    scores_null={
                        False: scores_null[b],
                        True: None,  # no cross val metrics
                    },
                    compute_FPR=False,
                    return_std=True,
                )
                # Append TPR and std to dict
                for metric, t_names in zip(METRICS.keys(), METRICS.values()):
                    if b:
                        name = t_names[1]
                    else:
                        name = t_names[0]
                    TPR_list[name].append(TPR[metric][0])
                    TPR_std_list[name].append(TPR_std[metric][0])

        # Save TPR dicts
        torch.save(
            TPR_list,
            PATH_EXPERIMENT
            + f"TPR_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl",
        )
        torch.save(
            TPR_std_list,
            PATH_EXPERIMENT
            + f"TPR_std_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl",
        )

# ====== ONLY PLOT THE RESULTS ======

if args.plot:
    # set-up path to save figures:
    fig_path = PATH_EXPERIMENT + "figures/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Test statistics for *optimal Bayes* classifier
    t_stats_dict = torch.load(
        PATH_EXPERIMENT + f"test_stats_OptBayes_shift_{args.q_dist}_dim_{dim}.pkl"
    )
    # TPR for *estimated* classifier (e.g. QDA)
    TPR_dict = torch.load(
        PATH_EXPERIMENT
        + f"TPR_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl"
    )
    TPR_std_dict = torch.load(
        PATH_EXPERIMENT
        + f"TPR_std_{clf_name}_shift_{args.q_dist}_dim_{dim}_n_runs_{N_RUNS}.pkl"
    )

    # Plot the results of both experiments
    plot_plot_c2st_single_eval_shift(
        shift_list=shifts,
        t_stats_dict=t_stats_dict,
        TPR_dict=TPR_dict,
        TPR_std_dict=TPR_std_dict,
        shift_name=args.q_dist,
        clf_name=clf_name,
    )
    plt.savefig(fig_path + f"optbayes_{clf_name}_shift_{args.q_dist}_dim_{dim}.pdf")
    plt.show()
