import os

import torch

from lc2st.vanillaC2ST import t_stats_c2st
from lc2st.localC2ST import t_stats_lc2st
from lc2st.localHPD import t_stats_lhpd


def precompute_t_stats_null(
    metrics,
    n_cal,
    n_eval,
    dim_theta,
    n_trials_null,
    t_stats_null_path,
    observation_dict,
    x_cal,
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
                t_stats_null = t_stats_c2st(
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
                _, _, trained_clfs_null = t_stats_lc2st(
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
                    t_stats_null[key_obs], probas_null[key_obs] = t_stats_lc2st(
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
                _, _, trained_clfs_null = t_stats_lhpd(
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
                    t_stats_null[key_obs], probas_null[key_obs] = t_stats_lhpd(
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
