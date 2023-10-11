# This is a general script to perform evaluations on the Classifier Two Sample Test (C2ST).
# =======================
# We define a utility function that computes the p-values, TPR and FPR over several runs of the C2ST
# and depending on different parameters of the experiment (e.g. the data distributions P and Q,
# the number of samples, the test statistics, whether to use only one class during evalaution, etc.).
# =======================
# In the main function, we define the parameters of the experiment and call the function to compute the p-values, TPR and FPR.
# The implemented experiments that can be run with this script are:
#   - plot the p-values/TPR/FPR/ROC curves for each metric over a grid of alpha values (significance levels).
#   - evaluate the type 1 error rate (FPR) for a given significance level alpha as a function of the sample size.
# Other experiments can be added.

import numpy as np

from tqdm import tqdm


def c2st_p_values_tfpr(
    eval_c2st_fn,
    n_runs,
    alpha_list,
    P_dist,
    Q_dist,
    n_samples,
    metrics,
    metrics_cv=[],
    n_folds=2,
    compute_FPR=True,
    compute_TPR=True,
    scores_null=None,
    return_std=False,
):
    """Computes the p-values, TPR and FPR over several runs of the Classifier Two Sample Test (C2ST)
    between two distributions P and Q:

                                         (H0): P = Q   (H1): P != Q
    for different metrics (test statistics).

    The p-value of a test-run is defined as the probability of falsely rejecting the null hypothesis (H0).
    For a given significance level alpha, we reject the null hypothesis (H0) if p-value < alpha.
    - TPR is the average number of times we correctly reject the null hypothesis (H0): power of the test.
    - FPR is the average number of times we incorrectly reject the null hypothesis (H0).
    We compute them for a range of significance levels alpha in (0,1), so that we can plot the ROC curve.

    Args:
        eval_c2st_fn (function): function that evaluates the C2ST test
        n_runs (int): number of test runs to compute FPR and TPR. Each time with new samples from P and Q.
        alpha_list (list): list of significance levels alpha in (0,1) to compute FPR and TPR at
        P_dist (scipy.stats.rv_continuous): distribution of P
        Q_dist (scipy.stats.rv_continuous): distribution of Q
        n_samples (dict): dict of number of samples from P and Q (for training and evaluation).
            keys: "train" and "eval".
            values: int.
        metrics (list): list of metrics to be used for the test (test statistics)
        metrics_cv (list): list of metrics to be used for the cross-validation.
            Defauts to None.
        compute_FPR (bool): whether to compute FPR or not.
            Defaults to True.
        compute_TPR (bool): whether to compute TPR or not.
            Defaults to True.
        scores_null (dict): dict of test statistics under the null.
            keys: True (cross-val) and False (no cross-val).
            values: second output of t_stats_c2st function.
            If None, use_permuation should be True.
            Defaults to None.

    Returns:
        p_values_H1 (dict): dict of p-values for each metric under (H1)
        p_values_H0 (dict): dict of p-values for each metric under (H0)
        TPR (dict): dict of TPR for each metric at each alpha in alpha_list
        FPR (dict): dict of FPR for each metric at each alpha in alpha_list
    """
    # extract number of samples
    n_train, n_eval = n_samples["train"], n_samples["eval"]

    # combine metrics and metrics_cv
    all_metrics = metrics + metrics_cv

    if scores_null is None:
        t_stats_null = None
        t_stats_null_cv = None
    else:
        t_stats_null = scores_null[False]
        t_stats_null_cv = scores_null[True]

    # initialize dict with empty lists
    p_values_H1 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    p_values_H0 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    # loop over test runs
    for _ in tqdm(range(n_runs), desc="Test runs"):
        # generate samples from P and Q
        P = P_dist.rvs(n_train)
        Q = Q_dist.rvs(n_train)
        Q_H0 = P_dist.rvs(n_train)

        P_eval = P_dist.rvs(n_eval)
        Q_eval = Q_dist.rvs(n_eval)
        Q_H0_eval = P_dist.rvs(n_eval)

        if P.ndim == 1:
            P = P.reshape(-1, 1)
            Q = Q.reshape(-1, 1)
            Q_H0 = Q_H0.reshape(-1, 1)

            P_eval = P_eval.reshape(-1, 1)
            Q_eval = Q_eval.reshape(-1, 1)
            Q_H0_eval = Q_H0_eval.reshape(-1, 1)

        if compute_TPR:
            # evaluate test under (H1)
            _, p_value, _, _ = eval_c2st_fn(
                metrics=metrics,
                # args for t_stats_c2st
                P=P,
                Q=Q,
                P_eval=P_eval,
                Q_eval=Q_eval,
                cross_val=False,
                t_stats_null=t_stats_null,
            )
            # update the empirical power at alpha for each metric
            for m in metrics:
                p_values_H1[m].append(p_value[m])

        if compute_FPR:
            # evaluate test under (H0)
            _, p_value, _, _ = eval_c2st_fn(
                metrics=metrics,
                P=P,
                Q=Q_H0,
                P_eval=P_eval,
                Q_eval=Q_H0_eval,
                cross_val=False,
                t_stats_null=t_stats_null,
            )
            # update the FPR at alpha for each metric
            for m in metrics:
                p_values_H0[m].append(p_value[m])

        if len(metrics_cv) > 0:
            if compute_TPR:
                # evaluate test under (H1) over several cross-val folds
                _, p_value_cv, _, _ = eval_c2st_fn(
                    metrics=metrics_cv,
                    P=P,
                    Q=Q,
                    cross_val=True,
                    n_folds=n_folds,
                    t_stats_null=t_stats_null_cv,
                )
                # update the empirical power at alpha for each cv-metric
                for m in metrics_cv:
                    p_values_H1[m].append(p_value_cv[m])

            if compute_FPR:
                # evaluate test under (H0) over several cross-val folds
                _, p_value_cv, _, _ = eval_c2st_fn(
                    metrics=metrics_cv,
                    P=P,
                    Q=Q_H0,
                    cross_val=True,
                    n_folds=n_folds,
                    t_stats_null=t_stats_null_cv,
                )
                # update the FPR at alpha for each cv-metric
                for m in metrics_cv:
                    p_values_H0[m].append(p_value_cv[m])

    # compute TPR and FPR at every alpha
    TPR = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    TPR_std = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    FPR = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    FPR_std = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    for alpha in alpha_list:
        # append TPR/TPF at alpha for each metric
        for m in all_metrics:
            if compute_TPR and alpha != 0:
                TPR[m].append(np.mean(np.array(p_values_H1[m]) <= alpha))
                TPR_std[m].append(np.std(np.array(p_values_H1[m]) <= alpha))
            else:
                TPR[m].append(0)
                TPR_std[m].append(0)
            if compute_FPR and alpha != 0:
                FPR[m].append(np.mean(np.array(p_values_H0[m]) <= alpha))
                FPR_std[m].append(np.std(np.array(p_values_H0[m]) <= alpha))
            else:
                FPR[m].append(0)
                FPR_std[m].append(0)

    if return_std:
        return TPR, FPR, TPR_std, FPR_std, p_values_H1, p_values_H0
    else:
        return TPR, FPR, p_values_H1, p_values_H0


if __name__ == "__main__":
    import argparse
    import os

    from functools import partial
    import matplotlib.pyplot as plt

    from lc2st.test_utils import eval_htest
    from lc2st.c2st import t_stats_c2st

    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import t

    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.neural_network import MLPClassifier

    # experiment parameters that need to be defined and cannot be passed in the argparser

    PATH_EXPERIMENT = "saved_experiments/c2st_evaluation/"

    # metrics / test statistics
    metrics = ["accuracy", "mse", "div"]
    # metrics_cv = ["accuracy_cv"]
    metrics_cv = []
    all_metrics = metrics + metrics_cv

    cross_val_folds = 2

    # Parse arguments
    # default values according to [Lee et al. 2018](https://arxiv.org/abs/1812.08927)
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument(
        "--n_samples",
        "-ns",
        nargs="+",
        type=int,
        default=[100],  # [25, 50, 100, 200, 500, 1000, 1500, 2000]
        help="Number of samples to draw from P and Q. Can be a list of values.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="Dimension of the data (number of features).",
    )

    parser.add_argument(
        "--q_dist",
        "-q",
        type=str,
        default=None,
        choices=["mean", "scale", "df"],
        help="Variable parameter in the distribution of Q.",
    )

    parser.add_argument(
        "--shifts",
        "-s",
        nargs="+",
        type=float,
        default=[0],
        help="List of shifts to apply to Q (mean/scale if gaussian, df if Student).",
    )

    # test parameters
    parser.add_argument(
        "--n_runs", "-nr", type=int, default=300, help="Number of test runs.",
    )
    parser.add_argument(
        "-alphas",
        "-a",
        nargs="+",
        type=float,
        default=np.linspace(0, 1, 20),
        help="List of significance levels to evaluate the test at.",
    )

    # null distribution parameters
    parser.add_argument(
        "--n_trials_null",
        "-nt",
        type=int,
        default=100,
        help="Number of trials to estimate the distribution of the test statistic under the null.",
    )
    parser.add_argument(
        "--use_permutation",
        "-p",
        action="store_true",
        help="Use permutations to estimate the null distribution. \
            If False, approximate the true null distribution with samples from P.",
    )

    # classifier parameters
    parser.add_argument(
        "--clf_name",  # make a list?
        "-c",
        type=str,
        default="LDA",
        choices=["LDA", "QDA", "MLP"],
        help="Classifier to use.",
    )
    parser.add_argument(
        "--single_class_eval",
        "-1c",
        action="store_true",
        help="Evaluate the classifier on one class only.",
    )
    parser.add_argument(
        "--in_sample",
        "-in",
        action="store_true",
        help="In-sample evaluation of the classifier (on training data).",
    )

    # experiment parameters
    parser.add_argument(
        "--roc",
        action="store_true",
        help="Compute and Plot ROC curve for the test. In this case `alphas` should be a grid in (0,1).",
    )
    parser.add_argument(
        "--err_ns",
        action="store_true",
        help="Compute and Plot Type 1 error/Power for the test over multiple sample sizes.",
    )

    parser.add_argument(
        "--err_shift",
        action="store_true",
        help="Compute and Plot Type 1 error/Power for the test over multiple shifts.",
    )

    parser.add_argument(
        "--perm_exp",
        action="store_true",
        help="Compute and Plot Type 1 error/Power for the test with or without permutation method.",
    )

    args = parser.parse_args()

    # ==== GLOBAL TEST PARAMETERS ====
    N_RUNS = args.n_runs
    N_TRIALS_NULL = args.n_trials_null

    # list of sample sizes to use for training the classifier
    N_SAMPLES_LIST = args.n_samples

    # list of sample sizes to evaluate the test statistics on
    # fixed at high value as Q = normal (NF) or estimator,
    # that we can sample arbitrarily many samples from
    N_EVAL_LIST = [10_000] * len(N_SAMPLES_LIST)

    # ==== EXPERIMENT SETUP ====
    # Define data distributions P and Q
    dim = args.dim  # data dimension
    P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))  # P is a standard Gaussian

    SHIFT_LIST = args.shifts
    if args.q_dist == "mean":
        s = 1
        Q_dist_list = [
            mvn(mean=np.array([mu] * dim), cov=s * np.eye(dim)) for mu in SHIFT_LIST
        ]  # Q is a reduced Gaussian with different means
    elif args.q_dist == "scale":
        mu = 0
        Q_dist_list = [
            mvn(mean=np.array([mu] * dim), cov=s * np.eye(dim)) for s in SHIFT_LIST
        ]  # Q is a centered Gaussian with different scales
    elif args.q_dist == "df":
        mu = 0
        s = 1
        Q_dist_list = [
            t(df=df, loc=mu, scale=s) for df in SHIFT_LIST
        ]  # Q is a standard Student with different degrees of freedom
    else:
        raise NotImplementedError

    # Initialize classifier
    if args.clf_name == "LDA":
        clf_class = LinearDiscriminantAnalysis
        clf_kwargs = {"solver": "eigen", "priors": [0.5, 0.5]}
    elif args.clf_name == "QDA":
        clf_class = QuadraticDiscriminantAnalysis
        clf_kwargs = {"priors": [0.5, 0.5]}
    elif args.clf_name == "MLP":
        clf_class = MLPClassifier
        clf_kwargs = {"alpha": 0, "max_iter": 25000}
    else:
        raise NotImplementedError

    # Initialise function to compute the test statistics
    t_stats_c2st_custom = partial(
        t_stats_c2st,
        n_trials_null=N_TRIALS_NULL,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        in_sample=args.in_sample,
        single_class_eval=args.single_class_eval,
        metrics=metrics,
    )

    # Pre-compute test statistics under the null distribution
    scores_null_list = []
    for k, n in enumerate(N_SAMPLES_LIST):
        n_eval = N_EVAL_LIST[k]
        print()
        print(f"N = {n}")
        if not args.use_permutation:
            # Not using the permutation method to simulate the null distribution
            # Using data from P to compute the scores/test statistics under the true null distribution
            print()
            print(
                "Pre-computing or loading the test statistics under the null distribution."
                + "\n They will be reused at every test-run. The permutation method is not needed."
            )
            print()
            scores_null = dict(zip([True, False], [None, None]))
            for cross_val, metric_list in zip([True, False], [metrics_cv, metrics]):
                filename = f"nt_{N_TRIALS_NULL}_N_{n}_dim_{dim}_{args.clf_name}_single_class_{args.single_class_eval}_in_sample_{args.in_sample}"
                if cross_val:
                    filename += f"_cross_val_nfolds_{cross_val_folds}.npy"
                else:
                    filename += ".npy"
                if os.path.exists(PATH_EXPERIMENT + "t_stats_null/" + filename):
                    # load null scores if they exist
                    t_stats_null = np.load(
                        PATH_EXPERIMENT + "t_stats_null/" + filename, allow_pickle=True,
                    ).item()
                else:
                    # otherwise, compute them
                    # generate data from P
                    list_P_null = [P_dist.rvs(n) for _ in range(2 * N_TRIALS_NULL)]
                    list_P_eval_null = [
                        P_dist.rvs(n_eval) for _ in range(2 * N_TRIALS_NULL)
                    ]
                    if dim == 1:
                        for i in range(2 * N_TRIALS_NULL):
                            list_P_null[i] = list_P_null[i].reshape(-1, 1)
                            list_P_eval_null[i] = list_P_eval_null[i].reshape(-1, 1)
                    t_stats_null = t_stats_c2st_custom(
                        null_hypothesis=True,
                        use_permutation=False,
                        metrics=metric_list,
                        cross_val=cross_val,
                        n_folds=cross_val_folds,
                        list_P_null=list_P_null,
                        list_P_eval_null=list_P_eval_null,
                        # required args inside `t_stats_c2st`
                        P=list_P_null[0],
                        Q=list_P_null[1],
                        P_eval=list_P_eval_null[0],
                        Q_eval=list_P_eval_null[1],
                    )
                    np.save(
                        PATH_EXPERIMENT + "t_stats_null/" + filename, t_stats_null,
                    )
                scores_null[cross_val] = t_stats_null
        else:
            print()
            print(
                f"Not pre-computing the test-statistics under the null."
                + "\n Using the permutation method to estimate them at each test run."
            )
            print()
            scores_null = None
        scores_null_list.append(scores_null)

    # Define function to evaluate the test
    eval_c2st = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st_custom,
        # use_permutation=args.use_permutation,
        verbose=False,
        metrics=metrics,
    )

    test_params = f"nruns_{N_RUNS}_n_trials_null_{N_TRIALS_NULL}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}"

    # ==== EXPERIMENTS 1 and 2:  ROC curves and TPR/FPR for fixed Q over different sample sizes ====
    if args.roc or args.err_ns:
        # define Q
        Q_dist = Q_dist_list[0]
        if args.q_dist == "mean":
            s = 1
            mu = SHIFT_LIST[0]
            q_params = f"gaussian_mean_{np.round(mu,2)}_dim_{dim}"
            H1_label = f"(H1): N(m={np.round(mu,2)}, {np.round(s,2)})"
            H1_label_shift = f"(H1): N(m, {np.round(s,2)})"
        elif args.q_dist == "scale":
            mu = 0
            s = SHIFT_LIST[0]
            q_params = f"gaussian_scale_{np.round(s,2)}_dim_{dim}"
            H1_label = f"(H1): N({np.round(mu,2)}, s={np.round(s,2)})"
            H1_label_shift = f"(H1): N({np.round(mu,2)}, s)"
        elif args.q_dist == "df":
            mu = 0
            s = 1
            df = SHIFT_LIST[0]
            q_params = f"student_df_{np.round(df,2)}"
            H1_label = (
                f"(H1): t({np.round(mu,2)}, {np.round(s,2)}, df={np.round(df,2)})"
            )
            H1_label_shift = f"(H1): t({np.round(mu,2)}, {np.round(s,2)}, df)"
        else:
            raise NotImplementedError

        # For each sample size, compute the test results for each metric:
        # p-values, TPR and FPR (at given alphas)
        TPR_list, FPR_list, p_values_H0_list, p_values_H1_list = [], [], [], []
        TPR_list_perm, FPR_list_perm, p_values_H0_list_perm, p_values_H1_list_perm = (
            [],
            [],
            [],
            [],
        )
        for i, n in enumerate(N_SAMPLES_LIST):
            print()
            print(f"N = {n}")
            print()
            TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
                eval_c2st_fn=eval_c2st,
                n_runs=N_RUNS,
                n_samples={"train": n, "eval": N_EVAL_LIST[i]},
                alpha_list=args.alphas,
                P_dist=P_dist,
                Q_dist=Q_dist,
                metrics=metrics,
                metrics_cv=metrics_cv,
                n_folds=cross_val_folds,
                scores_null=scores_null_list[i],
            )
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            p_values_H1_list.append(p_values_H1)
            p_values_H0_list.append(p_values_H0)

            if args.perm_exp:
                print("Running permutation test")
                (
                    TPR_perm,
                    FPR_perm,
                    p_values_H1_perm,
                    p_values_H0_perm,
                ) = c2st_p_values_tfpr(
                    eval_c2st_fn=partial(
                        eval_c2st, use_permutation=True, n_trials_null=100
                    ),
                    n_runs=N_RUNS,
                    n_samples={"train": n, "eval": N_EVAL_LIST[i]},
                    alpha_list=args.alphas,
                    P_dist=P_dist,
                    Q_dist=Q_dist,
                    metrics=metrics,
                    metrics_cv=metrics_cv,
                    n_folds=cross_val_folds,
                    scores_null=None,
                )
                TPR_list_perm.append(TPR_perm)
                FPR_list_perm.append(FPR_perm)
                p_values_H1_list_perm.append(p_values_H1_perm)
                p_values_H0_list_perm.append(p_values_H0_perm)

        # ==== EXP 1: plot ROC curves comparing test statistics for a given sample size ====

        if args.roc:
            print(f"EXP 1: ROC curves for {H1_label}, dim={dim}, n={n}")
            for i, n in enumerate(N_SAMPLES_LIST):
                # Plot p-values for each metric
                p_values_H0, p_values_H1 = p_values_H0_list[i], p_values_H1_list[i]
                for m in all_metrics:
                    p_values = np.concatenate(
                        [p_values_H1[m], p_values_H0[m]]
                    )  # concatenate H1 and H0 p-values
                    index = np.concatenate(
                        [np.ones(N_RUNS), np.zeros(N_RUNS)]
                    )  # 1 for H1, 0 for H0
                    sorter = np.argsort(p_values)  # sort p-values
                    sorted_index = index[sorter]  # sort index
                    idx_0 = np.where(sorted_index == 0)[0]  # find index of H0 p-values
                    idx_1 = np.where(sorted_index == 1)[0]  # find index of H1 p-values

                    plt.plot(np.sort(p_values), color="blue", label="p-values")

                    plt.scatter(
                        np.arange(2 * N_RUNS)[idx_1],
                        np.sort(p_values)[idx_1],
                        c="g",
                        label=H1_label,
                        alpha=0.3,
                    )
                    plt.scatter(
                        np.arange(2 * N_RUNS)[idx_0],
                        np.sort(p_values)[idx_0],
                        c="r",
                        label="H0",
                        alpha=0.3,
                    )
                    plt.legend()
                    plt.title(f"C2ST-{m}, (N={n}, dim={dim})")
                    plt.savefig(
                        PATH_EXPERIMENT
                        + f"p_values_{m}_{args.clf_name}"
                        + f"_{q_params}"
                        + f"_{test_params}"
                        + f"_N_{n}.pdf"
                    )
                    plt.show()

                TPR, FPR = TPR_list[i], FPR_list[i]
                # Plot TPR for each metric
                for m in all_metrics:
                    plt.plot(args.alphas, TPR[m], label=m)
                plt.legend()
                plt.title(f"TPR for C2ST, {H1_label}, (N={n}, dim={dim})")
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"tpr_{args.clf_name}"
                    + f"_{q_params}"
                    + f"_{test_params}"
                    + f"_N_{n}.pdf"
                )
                plt.show()

                # Plot FPR for each metric
                for m in all_metrics:
                    plt.plot(args.alphas, FPR[m], label=m)
                plt.legend()
                plt.title(f"FPR for C2ST, {H1_label}, (N={n}, dim={dim})")
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"fpr_{args.clf_name}"
                    + f"_{q_params}"
                    + f"_{test_params}"
                    + f"_N_{n}.pdf"
                )
                plt.show()

                # Plot ROC
                for m in all_metrics:
                    plt.plot(FPR[m], TPR[m], label=m)
                plt.legend()
                plt.title(f"ROC for C2ST, {H1_label}, (N={n}, dim={dim})")
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"roc_{args.clf_name}"
                    + f"_{q_params}"
                    + f"_{test_params}"
                    + f"_N_{n}.pdf"
                )
                plt.show()

        # ==== EXP 2: FPR and TPR as a function of n_samples at alpha =====
        # (as in [Lopez-Paz et al. 2016](https://arxiv.org/abs/1610.06545))

        if args.err_ns:
            print(
                f"EXP 2: FPR and TPR as a function of n_samples for {H1_label}, dim={dim}"
            )
            for k, alpha in enumerate(args.alphas):
                FPR_a = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
                TPR_a = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
                for m in all_metrics:
                    for i in range(len(N_SAMPLES_LIST)):
                        FPR_a[m].append(FPR_list[i][m][k])
                        TPR_a[m].append(TPR_list[i][m][k])
                if args.perm_exp:
                    FPR_a_perm = dict(
                        zip(all_metrics, [[] for _ in range(len(all_metrics))])
                    )
                    TPR_a_perm = dict(
                        zip(all_metrics, [[] for _ in range(len(all_metrics))])
                    )
                    for m in all_metrics:
                        for i in range(len(N_SAMPLES_LIST)):
                            FPR_a_perm[m].append(FPR_list_perm[i][m][k])
                            TPR_a_perm[m].append(TPR_list_perm[i][m][k])
                    for m in all_metrics:
                        plt.plot(N_SAMPLES_LIST, FPR_a[m], label=m)
                        plt.plot(
                            N_SAMPLES_LIST,
                            FPR_a_perm[m],
                            label=m + " (perm)",
                            linestyle="--",
                        )
                    plt.legend()
                    plt.title(
                        f"C2ST Type I error / FPR (alpha = {np.round(alpha,2)})"
                        + f"\n dim={dim}"
                    )
                    plt.savefig(
                        PATH_EXPERIMENT
                        + f"perm_type_I_error_ns_alpha_{np.round(alpha,2)}_{args.clf_name}_dim_{dim}"
                        + f"_{test_params}.pdf"
                    )
                    plt.show()

                    for m in all_metrics:
                        plt.plot(N_SAMPLES_LIST, TPR_a[m], label=m)
                        plt.plot(
                            N_SAMPLES_LIST,
                            TPR_a_perm[m],
                            label=m + " (perm)",
                            linestyle="--",
                        )
                    plt.legend()
                    plt.title(
                        f"C2ST Power / TPR (alpha = {np.round(alpha,2)})"
                        + f"\n {H1_label}, dim={dim}"
                    )
                    plt.savefig(
                        PATH_EXPERIMENT
                        + f"perm_power_ns_alpha_{np.round(alpha,2)}_{args.clf_name}"
                        + f"_{q_params}"
                        + f"_{test_params}.pdf"
                    )
                    plt.show()

                else:
                    for m in all_metrics:
                        plt.plot(N_SAMPLES_LIST, FPR_a[m], label=m)
                    plt.legend()
                    plt.title(
                        f"C2ST Type I error / FPR (alpha = {np.round(alpha,2)})"
                        + f"\n dim={dim}"
                    )
                    plt.savefig(
                        PATH_EXPERIMENT
                        + f"type_I_error_ns_alpha_{np.round(alpha,2)}_{args.clf_name}_dim_{dim}"
                        + f"_{test_params}.pdf"
                    )
                    plt.show()

                    for m in all_metrics:
                        plt.plot(N_SAMPLES_LIST, TPR_a[m], label=m)
                    plt.legend()
                    plt.title(
                        f"C2ST Power / TPR (alpha = {np.round(alpha,2)})"
                        + f"\n {H1_label}, dim={dim}"
                    )
                    plt.savefig(
                        PATH_EXPERIMENT
                        + f"power_ns_alpha_{np.round(alpha,2)}_{args.clf_name}"
                        + f"_{q_params}"
                        + f"_{test_params}.pdf"
                    )
                    plt.show()

    # ==== EXP 3: FPR and TPR as a function of shifts at fixed N and alpha=====
    if args.err_shift:
        # define case labels
        if args.q_dist == "mean":
            s = 1
            case = f"gaussian_mean"
            H1_label = f"(H1): N(m, {np.round(s,2)})"
        elif args.q_dist == "scale":
            mu = 0
            case = f"gaussian_scale"
            H1_label = f"(H1): N({np.round(mu,2)}, s)"
        elif args.q_dist == "df":
            mu = 0
            s = 1
            case = f"student_df"
            H1_label = f"(H1): t({np.round(mu,2)}, {np.round(s,2)}, df)"
        else:
            raise NotImplementedError

        # fix N
        n = N_SAMPLES_LIST[0]
        n_eval = N_EVAL_LIST[0]
        scores_null = scores_null_list[0]

        print(
            f"Exp 3: FPR and TPR as a function of {case} shifts at fixed N={n}, dim={dim}"
        )

        # compute TPR and FPR for each shift
        TPR_list, FPR_list, p_values_H0_list, p_values_H1_list = [], [], [], []
        TPR_list_perm, FPR_list_perm, p_values_H0_list_perm, p_values_H1_list_perm = (
            [],
            [],
            [],
            [],
        )
        for i, Q_dist in enumerate(Q_dist_list):
            print()
            print(f"{case}: {H1_label}, shift = {SHIFT_LIST[i]}")
            print()
            TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
                eval_c2st_fn=eval_c2st,
                n_runs=N_RUNS,
                n_samples={"train": n, "eval": n_eval},
                alpha_list=args.alphas,
                P_dist=P_dist,
                Q_dist=Q_dist,
                metrics=metrics,
                metrics_cv=metrics_cv,
                n_folds=cross_val_folds,
                scores_null=scores_null,
            )
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            p_values_H1_list.append(p_values_H1)
            p_values_H0_list.append(p_values_H0)

            if args.perm_exp:
                print("Running permutation test")
                (
                    TPR_perm,
                    FPR_perm,
                    p_values_H1_perm,
                    p_values_H0_perm,
                ) = c2st_p_values_tfpr(
                    eval_c2st_fn=partial(
                        eval_c2st, use_permutation=True, n_trials_null=100
                    ),
                    n_runs=N_RUNS,
                    n_samples={"train": n, "eval": n_eval},
                    alpha_list=args.alphas,
                    P_dist=P_dist,
                    Q_dist=Q_dist,
                    metrics=metrics,
                    metrics_cv=metrics_cv,
                    n_folds=cross_val_folds,
                    scores_null=None,
                )
                TPR_list_perm.append(TPR_perm)
                FPR_list_perm.append(FPR_perm)
                p_values_H1_list_perm.append(p_values_H1_perm)
                p_values_H0_list_perm.append(p_values_H0_perm)

        # For each alpha, plot TPR and FPR for each metric as a function of shift
        for k, alpha in enumerate(args.alphas):
            FPR_a = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
            TPR_a = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
            for m in all_metrics:
                for i in range(len(SHIFT_LIST)):
                    FPR_a[m].append(FPR_list[i][m][k])
                    TPR_a[m].append(TPR_list[i][m][k])

            if args.perm_exp:
                FPR_a_perm = dict(
                    zip(all_metrics, [[] for _ in range(len(all_metrics))])
                )
                TPR_a_perm = dict(
                    zip(all_metrics, [[] for _ in range(len(all_metrics))])
                )
                for m in all_metrics:
                    for i in range(len(SHIFT_LIST)):
                        FPR_a_perm[m].append(FPR_list_perm[i][m][k])
                        TPR_a_perm[m].append(TPR_list_perm[i][m][k])

                for m in all_metrics:
                    plt.plot(SHIFT_LIST, FPR_a[m], label=m)
                    plt.plot(
                        SHIFT_LIST, FPR_a_perm[m], label=m + " (perm)", linestyle="--"
                    )
                plt.legend()
                plt.title(
                    f"C2ST Type I error / FPR (alpha = {np.round(alpha,2)})"
                    + f"\n dim={dim}"
                )
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"perm_type_I_error_{case}_shift_dim_{dim}_N_{n}_alpha_{np.round(alpha,2)}_{args.clf_name}"
                    + f"_{test_params}.pdf"
                )
                plt.show()

                for m in all_metrics:
                    plt.plot(SHIFT_LIST, TPR_a[m], label=m)
                    plt.plot(
                        SHIFT_LIST, TPR_a_perm[m], label=m + " (perm)", linestyle="--"
                    )
                plt.legend()
                plt.title(f"C2ST Power / TPR (alpha = {np.round(alpha,2)})")
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"perm_power_{case}_shift_dim_{dim}_N_{n}_alpha_{np.round(alpha,2)}_{args.clf_name}"
                    + f"_{test_params}.pdf"
                )
                plt.show()

            else:
                # FPR
                for m in all_metrics:
                    plt.plot(SHIFT_LIST, FPR_a[m], label=m)
                plt.legend()
                plt.title(
                    f"C2ST Type I error / FPR (alpha = {np.round(alpha,2)})"
                    + f"\n dim={dim}"
                )
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"type_I_error_{case}_shift_dim_{dim}_N_{n}_alpha_{np.round(alpha,2)}_{args.clf_name}"
                    + f"_{test_params}.pdf"
                )
                plt.show()

                # TPR
                for m in all_metrics:
                    plt.plot(SHIFT_LIST, TPR_a[m], label=m)
                plt.legend()
                plt.title(
                    f"C2ST Power / TPR (alpha = {np.round(alpha,2)})"
                    + f"\n {H1_label}, dim={dim}"
                )
                plt.savefig(
                    PATH_EXPERIMENT
                    + f"power_{case}_shift_dim_{dim}_N_{n}_alpha_{np.round(alpha,2)}_{args.clf_name}"
                    + f"_{test_params}.pdf"
                )
                plt.show()
