# Utils for general hypothesis testing:
# - compute_pvalue
# - eval_htest

# Utils for implemented validation diagnostics:
#  - permute_data (for permutation method to simulate null hypothesis)
#  - precompute_test_statistics_null

import numpy as np
import os
import torch


def compute_pvalue(t_stat_est, t_stats_null):
    """Computes the p-value of a hypothesis test as the empirical estimate of:

        p = Prob(T > \hat{T} | H0)

        which represents the probability of making a type 1 error, i.e. the probability
        of falsly rejecting the null hypothesis (H0).

    Args:
        t_stat_est (float): test statistic \hat{T} estimated on observed data.
        t_stats_null (list or array): a sample {ti} of the test statistic drawn under (H0):
            --> t_i ~ T|(H0).

    Returns:
        float: empirical p-value: 1/n * \sum_{i=1}^n 1_{t_i > \hat{T}}, ti ~ T|(H0).
    """
    return (t_stat_est < np.array(t_stats_null)).mean()


def eval_htest(
    t_stats_estimator, metrics, conf_alpha=0.05, t_stats_null=None, **kwargs
):
    """Evaluates a hypothesis test at a given significance level.

    Args:
        t_stats_estimator (function):
            - takes as input a list of metrics that will give an estimate
            of the corresponding test statistics when computed on an observed data sample.
            - returns objects taken as inputs in `compute_pvalue` (i.e. test statistic
            estimated on observed data and drawn under the null hypothesis)
        metrics (list of str): contains the names of the metrics used in `t_stats_estimator`
        conf_alpha (float, optional): significance level of the test, yielding a confidence level
            of 1 - conf_alpha. Defaults to 0.05.
        t_stats_null (list or array, optional): precomputed samples {t_i} of the test statistic drawn under (H0):
            --> t_i ~ T|(H0). Defaults to None.
        kwargs: additional inputs to `t_stats_estimator`: True rejected, False otherwise.

    Returns:
        reject (dict): dictionary of booleans indicating whether the null hypothesis is rejected
            for each metric.
        p_value (dict): dictionary of p-values for each metric.
        t_stat_data (dict): dictionary of test statistics estimated on observed data for each metric.
        t_stats_null (dict): dictionary of test statistics drawn under the null hypothesis for each metric.
    """
    reject = {}
    p_value = {}
    t_stat_data = t_stats_estimator(metrics=metrics, **kwargs)
    if t_stats_null is None:
        t_stats_null = t_stats_estimator(
            metrics=metrics, null_hypothesis=True, **kwargs
        )

    for m in metrics:
        p_value[m] = compute_pvalue(t_stat_data[m], t_stats_null[m])
        reject[m] = p_value[m] < conf_alpha  # True = reject

    return reject, p_value, t_stat_data, t_stats_null


def permute_data(P, Q, seed=42):
    """Permute the concatenated data [P,Q] to create null-hyp samples.

    Args:
        P (torch.Tensor): data of shape (n_samples, dim)
        Q (torch.Tensor): data of shape (n_samples, dim)
        seed (int, optional): random seed. Defaults to 42.
    """
    # set seed
    torch.manual_seed(seed)
    # check inputs
    assert P.shape[0] == Q.shape[0]

    n_samples = P.shape[0]
    X = torch.cat([P, Q], dim=0)
    X_perm = X[torch.randperm(n_samples * 2)]
    return X_perm[:n_samples], X_perm[n_samples:]


def precompute_t_stats_null(
    metrics,
    n_cal,
    n_eval,
    dim_theta,
    n_trials_null,
    t_stats_null_path,
    observation_dict,
    x_cal,
    t_stats_fn_c2st,
    t_stats_fn_lc2st,
    t_stats_fn_lhpd,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    methods=["c2st_nf", "lc2st_nf", "lhpd"],
    save_results=True,
    load_results=True,
    return_predicted_probas=False,
):
    # fixed distribution for null hypothesis (base distribution)
    from scipy.stats import multivariate_normal as mvn

    if save_results and not os.path.exists(t_stats_null_path):
        os.makedirs(t_stats_null_path)

    P_dist_null = mvn(mean=torch.zeros(dim_theta), cov=torch.eye(dim_theta))
    list_P_null = [
        P_dist_null.rvs(n_cal, random_state=t) for t in range(2 * n_trials_null)
    ]
    list_P_eval_null = [
        P_dist_null.rvs(n_eval, random_state=t) for t in range(2 * n_trials_null)
    ]

    t_stats_null_dict = dict(zip(methods, [{} for _ in methods]))
    probas_null_dict = dict(zip(methods, [{} for _ in methods]))

    for m in methods:
        try:
            if not load_results:
                raise FileNotFoundError
            t_stats_null = torch.load(
                t_stats_null_path
                / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl"
            )
            if return_predicted_probas:
                name = "probas"
                if m == "lhpd":
                    name = "r_alphas"
                probas_null = torch.load(
                    t_stats_null_path
                    / f"{m}_{name}_nt_{n_trials_null}_n_cal_{n_cal}.pkl"
                )
            else:
                probas_null = None
            print()
            print(f"Loaded pre-computed test statistics for {m}-H_0")
        except FileNotFoundError:
            print()
            print(
                f"Pre-compute test statistics for {m}-H_0 (N_cal={n_cal}, n_trials={n_trials_null})"
            )
            if m == "c2st_nf":
                print()
                print("C2ST: TRAIN / EVAL CLASSIFIERS ...")
                print()
                t_stats_null = t_stats_fn_c2st(
                    null_hypothesis=True,
                    metrics=metrics,
                    list_P_null=list_P_null,
                    list_P_eval_null=list_P_eval_null,
                    use_permutation=False,
                    n_trials_null=n_trials_null,
                    # required kwargs for t_stats_c2st
                    P=None,
                    Q=None,
                    # kwargs for c2st_scores
                    **kwargs_c2st,
                )
            elif m == "lc2st_nf":
                # train clfs on joint samples
                print()
                print("L-C2ST: TRAINING CLASSIFIERS on the joint ...")
                print()
                _, _, trained_clfs_null = t_stats_fn_lc2st(
                    null_hypothesis=True,
                    metrics=metrics,
                    list_P_null=list_P_null,
                    list_x_P_null=[x_cal] * len(list_P_null),
                    use_permutation=False,
                    n_trials_null=n_trials_null,
                    return_clfs_null=True,
                    # required kwargs for t_stats_lc2st
                    P=None,
                    Q=None,
                    x_P=None,
                    x_Q=None,
                    P_eval=None,
                    list_P_eval_null=list_P_eval_null,
                    x_eval=None,
                    # kwargs for lc2st_scores
                    eval=False,
                    **kwargs_lc2st,
                )
                print()
                print("L-C2ST: Evaluate for every observation ...")
                t_stats_null = {}
                probas_null = {}
                for key_obs, observation in observation_dict.items():
                    t_stats_null[key_obs], probas_null[key_obs] = t_stats_fn_lc2st(
                        null_hypothesis=True,
                        metrics=metrics,
                        list_P_null=list_P_null,
                        list_P_eval_null=list_P_eval_null,
                        # ==== added for LC2ST ====
                        list_x_P_null=[x_cal] * len(list_P_null),
                        x_eval=observation,
                        return_probas=True,
                        # =========================
                        use_permutation=False,
                        n_trials_null=n_trials_null,
                        trained_clfs_null=trained_clfs_null,
                        # required kwargs for t_stats_lc2st
                        P=None,
                        Q=None,
                        x_P=None,
                        x_Q=None,
                        P_eval=None,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
            elif m == "lhpd":
                # train clfs on joint samples
                print()
                print("L-HPD: TRAINING CLASSIFIERS on the joint ...")
                print()
                _, _, trained_clfs_null = t_stats_fn_lhpd(
                    metrics=["mse"],
                    Y=list_P_null[0],  # for dim inside lhpd_scores
                    X=x_cal,
                    null_hypothesis=True,
                    n_trials_null=n_trials_null,
                    return_clfs_null=True,
                    # required kwargs for t_stats_lhpd
                    x_eval=None,
                    # kwargs for lhpd_scores
                    eval=False,
                    est_log_prob_fn=None,
                    est_sample_fn=None,
                    **kwargs_lhpd,
                )
                print()
                print("L-HPD: Evaluate for every observation ...")
                t_stats_null = {}
                probas_null = {}
                for key_obs, observation in observation_dict.items():
                    t_stats_null[key_obs], probas_null[key_obs] = t_stats_fn_lhpd(
                        metrics=["mse"],
                        Y=list_P_null[0],  # for dim inside lhpd_scores
                        X=x_cal,
                        null_hypothesis=True,
                        n_trials_null=n_trials_null,
                        trained_clfs_null=trained_clfs_null,
                        return_clfs_null=False,
                        return_r_alphas=True,
                        # required kwargs for t_stats_lhpd
                        x_eval=observation,
                        # kwargs for lhpd_scores
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        **kwargs_lhpd,
                    )

            if save_results:
                torch.save(
                    t_stats_null,
                    t_stats_null_path
                    / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl",
                )
                if return_predicted_probas:
                    name = "probas"
                    if m == "lhpd":
                        name = "r_alphas"
                    torch.save(
                        probas_null,
                        t_stats_null_path
                        / f"{m}_{name}_nt_{n_trials_null}_n_cal_{n_cal}.pkl",
                    )
        t_stats_null_dict[m] = t_stats_null
        probas_null_dict[m] = probas_null
    if return_predicted_probas:
        return t_stats_null_dict, probas_null_dict
    else:
        return t_stats_null_dict
