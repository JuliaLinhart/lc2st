# Functions to run experiments on sbibm tasks with npe-flows

# IMPORTS
import copy
import numpy as np
import os
import time
import torch

from lc2st.test_utils import eval_htest, permute_data, precompute_t_stats_null
from lc2st.c2st import t_stats_c2st
from lc2st.lc2st import t_stats_lc2st, lc2st_scores
from lc2st.lhpd import t_stats_lhpd, lhpd_scores
from tasks.sbibm.data_generators import (
    generate_task_data,
    generate_npe_data_for_c2st,
    generate_npe_data_for_lc2st,
)
from tasks.sbibm.npe_utils import sample_from_npe_obs
from tqdm import tqdm


def l_c2st_results_n_train(
    task,
    n_cal,
    n_eval,
    observation_dict,
    n_train_list,
    alpha,
    n_trials_null,
    t_stats_null_c2st_nf,
    n_trials_null_precompute,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    task_path,
    t_stats_null_path,
    results_n_train_path="",
    methods=["c2st", "lc2st", "lc2st_nf"],
    test_stat_names=["accuracy", "mse", "div"],
    seed=42,
):
    """Compute the test results (for one run) for a given task and multiple npes
    (corresponding to n_train_list). All methods use the permutation method,
    except for (l)c2st_nf and lhpd that precomputes the test statistics under the null
    hypothesis and reuses them for every npe.

    Args:
        task (str): sbibm task name
        n_cal (int): Number of samples to use for calibration (train classifier).
        n_eval (int): Number of samples to use for evaluation (evaluate classifier).
        observation_dict (dict): dict of observations over which we average the results.
            keys are observation numbers
            values are torch tensors of shape (1, dim_x)
        n_train_list (List[int]): list of number of training samples for the npe.
        alpha (float): significance level of the test.
        n_trials_null (int): number of trials to compute the null distribution of
            the test statistic.
        t_stats_null_c2st_nf (dict): dict of precomputed test statistics for c2st_nf.
        n_trials_null_precompute (int): number of trials to precompute the null distribution
            of the test statistic.
        kwargs_c2st (dict): dict of kwargs for c2st.
        kwargs_lc2st (dict): dict of kwargs for lc2st.
        kwargs_lhpd (dict): dict of kwargs for lhpd.
        task_path (str): path to the task folder.
        t_stats_null_path (str): path to the folder where the precomputed test statistics under
            the null hypothesis are saved.
        results_n_train_path (str): path to the folder where the results are saved.
        methods (List[str]): list of methods to use for the test.
            Defaults to ['c2st', 'lc2st', 'lc2st-nf'].
        test_stat_names (List[str]): list of test statistic names to compute empirical power for.
            Must be compatible with the test_stat_estimator from `lc2st.test_utils.eval_htest`.
            Defaults to ['accuracy', 'mse', 'div'].
        seed (int): seed for reproducibility.

    Returns:
        avg_results (dict): dict of average results for every method and test statistic
            whose values are lists of average results for every n_train.
        train_runtime (dict): dict of average runtime for every method
            whose values are lists of average runtime for every n_train.
    """
    # Generate data
    if "c2st" in methods:
        generate_c2st_data = True
    else:
        generate_c2st_data = False

    data_samples = generate_data_one_run(
        n_cal=n_cal,
        n_eval=n_eval,
        task=task,
        observation_dict=observation_dict,
        n_train_list=n_train_list,
        task_path=task_path,
        save_data=True,  # save data to disk
        load_cal_data=True,  # load calibration data from disk
        load_eval_data=True,  # load evaluation data from disk
        seed=seed,  # fixed seed for reproducibility
        generate_c2st_data=generate_c2st_data,
    )

    t_stats_null_lc2st_nf = None
    t_stats_null_lhpd = None

    # Precompute test statistics under null hypothesis for lc2st_nf
    # same for every estimator (no need to recompute for every n_train)
    if "lc2st_nf" in methods:
        x_cal = data_samples["joint_cal"]["x"]
        dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
        t_stats_null_lc2st_nf = precompute_t_stats_null(
            n_cal=n_cal,
            n_eval=n_eval,
            dim_theta=dim_theta,
            n_trials_null=n_trials_null_precompute,
            t_stats_fn_lc2st=t_stats_lc2st,
            kwargs_lc2st=kwargs_lc2st,
            x_cal=x_cal,
            observation_dict=observation_dict,
            methods=["lc2st_nf"],
            metrics=test_stat_names,
            t_stats_null_path=t_stats_null_path,
            save_results=True,
            load_results=True,
            # args only for lc2st
            t_stats_fn_c2st=None,
            t_stats_fn_lhpd=None,
            kwargs_c2st=None,
            kwargs_lhpd=None,
        )["lc2st_nf"]

    if "lhpd" in methods:
        x_cal = data_samples["joint_cal"]["x"]
        dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
        t_stats_null_lhpd = precompute_t_stats_null(
            n_cal=n_cal,
            n_eval=n_eval,
            dim_theta=dim_theta,
            n_trials_null=n_trials_null_precompute,
            t_stats_fn_lhpd=t_stats_lhpd,
            kwargs_lhpd=kwargs_lhpd,
            x_cal=x_cal,
            observation_dict=observation_dict,
            methods=["lhpd"],
            metrics=["mse"],
            t_stats_null_path=t_stats_null_path,
            save_results=True,
            load_results=True,
            # args only for lhpd
            t_stats_fn_c2st=None,
            t_stats_fn_lc2st=None,
            kwargs_c2st=None,
            kwargs_lc2st=None,
        )["lhpd"]

    # Dictionary of average result keys
    avg_result_keys = {
        "TPR": "reject",
        "p_value_mean": "p_value",
        "p_value_std": "p_value",
        "p_value_min": "p_value",
        "p_value_max": "p_value",
        "t_stat_mean": "t_stat",
        "t_stat_std": "t_stat",
        "t_stat_min": "t_stat",
        "t_stat_max": "t_stat",
        "run_time_mean": "run_time",
        "run_time_std": "run_time",
    }
    # Initialize dict of average results
    avg_results = dict(zip(methods, [dict() for _ in methods]))
    for m in methods:
        avg_results[m] = dict(
            zip(
                avg_result_keys.keys(),
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in avg_result_keys
                ],
            )
        )
    train_runtime = dict(zip(methods, [[] for _ in methods]))

    # Compute test results for every n_train (one run)
    # Loop over n_train
    for n_train in n_train_list:
        results_dict, train_runtime_n = compute_test_results_npe_one_run(
            alpha=alpha,
            data_samples=data_samples,
            n_train=n_train,
            observation_dict=observation_dict,
            kwargs_c2st=kwargs_c2st,
            kwargs_lc2st=kwargs_lc2st,
            kwargs_lhpd=kwargs_lhpd,
            n_trials_null=n_trials_null,
            t_stats_null_c2st_nf=t_stats_null_c2st_nf,
            t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
            t_stats_null_lhpd=t_stats_null_lhpd,
            t_stats_null_dict_npe={m: None for m in methods},
            task_path=task_path,
            results_n_train_path=results_n_train_path,
            methods=methods,
            test_stat_names=test_stat_names,
            compute_under_null=False,  # no type I error computation
            save_results=True,  # save results to disk
            seed=seed,
        )

        # Append train runtime
        for method in methods:
            train_runtime[method].append(train_runtime_n[method])

        # Append average results
        for method, results in results_dict.items():
            if method in methods:
                for k, v in avg_result_keys.items():
                    for t_stat_name in test_stat_names:
                        if method == "lhpd" and t_stat_name != "mse":
                            continue
                        # compute std
                        if "std" in k:
                            if "run_time" in k:
                                avg_results[method][k][t_stat_name].append(
                                    np.std(
                                        np.array(results[v][t_stat_name])
                                        # / n_trials_null
                                    )
                                )
                            else:
                                avg_results[method][k][t_stat_name].append(
                                    np.std(results[v][t_stat_name])
                                )
                        # compute min
                        elif "min" in k:
                            avg_results[method][k][t_stat_name].append(
                                np.min(results[v][t_stat_name])
                            )
                        # compute max
                        elif "max" in k:
                            avg_results[method][k][t_stat_name].append(
                                np.max(results[v][t_stat_name])
                            )
                        else:
                            # compute mean
                            if "run_time" in k:
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(
                                        np.array(results[v][t_stat_name])
                                        # / n_trials_null
                                    )
                                )
                            else:
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(results[v][t_stat_name])
                                )
    return avg_results, train_runtime


def compute_emp_power_l_c2st(
    n_runs,
    alpha,
    task,
    n_train,
    observation_dict,
    n_cal_list,
    n_eval,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    n_trials_null,
    n_trials_null_precompute,
    t_stats_null_c2st_nf,
    task_path,
    methods=["c2st", "lc2st", "lc2st_nf", "lhpd"],
    test_stat_names=["accuracy", "mse", "div"],
    compute_emp_power=True,
    compute_type_I_error=False,
    n_run_load_results=0,
    result_path="",
    t_stats_null_path="",
    results_n_train_path="",
    load_eval_data=True,
    save_every_n_runs=5,
):
    """Compute the empirical power of all methods for a given task and npe-flow
    (corresponding to n_train). We also compute the type I error if specified.

    All methods will use the permutation method, except for (l)-c2st_nf and lhpd that for which
    an asymptotic approximation of the null distribution of the test statistic can be computed
    and used for both, power and type I error computation.

    This function enables doing experiments where the results are obtained for all methods
    on the same data, by computing them all in the same run.

    Args:
        n_runs (int): number of test runs to compute empirical power / type I error.
        alpha (float): significance level of the test.
        task (str): sbibm task name
        n_train (int): number of training samples for the npe.
        observation_dict (dict): dict of observations over which we average the results.
            keys are observation numbers
            values are torch tensors of shape (1, dim_x)
        n_cal_list (List[int]): Number of samples to use for calibration (train classifier).
        n_eval (int): Number of samples to use for evaluation (evaluate classifier).
        kwargs_c2st (dict): dict of kwargs for c2st.
        kwargs_lc2st (dict): dict of kwargs for lc2st.
        kwargs_lhpd (dict): dict of kwargs for lhpd.
        n_trials_null (int): number of trials to compute the null distribution of the test statistic.
        n_trials_null_precompute (int): number of trials to precompute the null distribution of
            the test statistic.
        t_stats_null_c2st_nf (dict): dict of precomputed test statistics for c2st_nf.
        task_path (str): path to the task folder.
        methods (List[str]): list of methods to use for the test.
            Defaults to ['c2st', 'lc2st', 'lc2st_nf', 'lhpd'].
        test_stat_names (List[str]): list of test statistic names.
            Must be compatible with the test_stat_estimator from `lc2st.test_utils.eval_htest`.
            Defaults to ['accuracy', 'mse', 'div'].
        compute_emp_power (bool): whether to compute the empirical power of the test.
            Defaults to True.
        compute_type_I_error (bool): whether to compute the type I error of the test.
            Defaults to False.
        n_run_load_results (int): number of the run from which to load the results.
            Defaults to 0.
        result_path (str): path to the folder where the results are saved.
        t_stats_null_path (str): path to the folder where the precomputed test statistics under
            the null hypothesis are saved.
        results_n_train_path (str): path to the folder where the results are saved.
        load_eval_data (bool): whether to load the evaluation data from disk.
            Defaults to True.
        save_every_n_runs (int): frequency at which to save the results.
            Defaults to 5.

    Returns:
        emp_power (dict): dict of empirical power for every n_cal, method and test statistic
            whose values are lists of empirical power for every observation.
        type_I_error (dict): dict of type I error for every n_cal, method and test statistic
            whose values are lists of type I error for every observation.
        p_values (dict): dict of p-values for every n_cal, method, test statistic and observation
            whose values are lists of p-values for every run.
        p_values_h0 (dict): dict of p-values under the null hypothesis for every n_cal, method,
            test statistic and observation, whose values are lists of p-values under the null hypothesis
            for every run.
    """
    # ==== Precompute test statistics under null hypothesis for methods that don't need the permutation method ====

    # Initialize dict of precomputed test statistics under null hypothesis
    all_methods = ["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm", "lhpd"]
    t_stats_null_dict = {n_cal: {m: None for m in all_methods} for n_cal in n_cal_list}

    # c2st:
    # independent of x_cal so it's possible to compute them once and use them for all test runs
    if "c2st" in methods:
        try:
            for n_cal in n_cal_list:
                t_stats_null_dict[n_cal]["c2st"] = torch.load(
                    task_path
                    / f"npe_{n_train}"
                    / "t_stats_null"
                    / f"t_stats_null_c2st_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
        except FileNotFoundError:
            # Generate data for a maximum n_cal
            data_samples = generate_data_one_run(
                n_cal=max(n_cal_list),
                n_eval=n_eval,
                task=task,
                observation_dict=observation_dict,
                n_train_list=[n_train],
                task_path=task_path,
                save_data=True,
                load_cal_data=True,
                load_eval_data=True,
            )
            for n_cal in n_cal_list:
                # Reduce number of calibration samples to n_cal
                data_samples_n = copy.deepcopy(data_samples)
                data_samples_n["base_dist"]["cal"] = data_samples_n["base_dist"]["cal"][
                    :n_cal
                ]
                data_samples_n["ref_posterior"]["cal"] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["ref_posterior"]["cal"].items()
                }
                data_samples_n["npe_obs"]["cal"][n_train] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["npe_obs"]["cal"][n_train].items()
                }
                data_samples_n["ref_inv_transform"]["cal"][n_train] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["ref_inv_transform"]["cal"][
                        n_train
                    ].items()
                }
                data_samples_n["joint_cal"]["x"] = data_samples_n["joint_cal"]["x"][
                    :n_cal
                ]
                data_samples_n["joint_cal"]["theta"] = data_samples_n["joint_cal"][
                    "theta"
                ][:n_cal]
                data_samples_n["npe_x_cal"][n_train] = data_samples_n["npe_x_cal"][
                    n_train
                ][:n_cal]
                data_samples_n["inv_transform_theta_cal"][n_train] = data_samples_n[
                    "inv_transform_theta_cal"
                ][n_train][:n_cal]

                # Compute test statistics for c2st-H_0
                print()
                print(
                    f"Pre-compute test statistics for c2st-H_0 (N_cal={n_cal}, n_trials={n_trials_null})"
                )
                print()
                _, _, t_stats_null = compute_test_results_npe_one_run(
                    alpha=alpha,
                    data_samples=data_samples_n,
                    n_train=n_train,
                    observation_dict=observation_dict,
                    kwargs_c2st=kwargs_c2st,
                    kwargs_lc2st=kwargs_lc2st,
                    kwargs_lhpd=kwargs_lhpd,
                    n_trials_null=n_trials_null,  # low number of trials as they need to be recomputed for every estimator
                    t_stats_null_lc2st_nf=None,
                    t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                    t_stats_null_lhpd=None,
                    t_stats_null_dict_npe=t_stats_null_dict[n_cal],
                    test_stat_names=test_stat_names,
                    methods=["c2st"],
                    compute_under_null=False,
                    task_path=task_path,
                    results_n_train_path=results_n_train_path,
                    save_results=True,
                    load_results=False,
                    return_t_stats_null=True,
                )
                t_stats_null_dict[n_cal]["c2st"] = t_stats_null["c2st"]

    # Initialize dicts
    emp_power = {}
    type_I_error = {}
    p_values = {}
    p_values_h0 = {}

    # Loop over n_cal
    for n_cal in n_cal_list:
        # Initialize dicts at n_cal
        emp_power[n_cal], p_values[n_cal] = {}, {}
        type_I_error[n_cal], p_values_h0[n_cal] = {}, {}

        for result_dict, p_values_dict, name, name_p, compute in zip(
            [emp_power, type_I_error],
            [p_values, p_values_h0],
            ["emp_power", "type_I_error"],
            ["p_values", "p_values_h0_"],
            [compute_emp_power, compute_type_I_error],
        ):
            if not compute:
                continue
            try:
                for method in methods:
                    # Load result if it exists and start from run n_run_load_results + 1 ...
                    result_dict[n_cal][method] = torch.load(
                        result_path
                        / f"n_runs_{n_run_load_results}"
                        / f"{name}_{method}_n_runs_{n_run_load_results}_n_cal_{n_cal}.pkl"
                    )
                    p_values_dict[n_cal][method] = torch.load(
                        result_path
                        / f"n_runs_{n_run_load_results}"
                        / f"{name_p}_obs_per_run_{method}_n_runs_{n_run_load_results}_n_cal_{n_cal}.pkl"
                    )
                start_run = n_run_load_results + 1
                print(
                    f"Loaded {name} results for N_cal = {n_cal} from run {n_run_load_results} ..."
                )
            except FileNotFoundError:
                # ... otherwise initialize them to empty lists and start from run 1
                start_run = 1
                for method in methods:
                    result_dict[n_cal][method] = dict(
                        zip(
                            test_stat_names,
                            [np.zeros(len(observation_dict)) for _ in test_stat_names],
                        )
                    )
                    p_values_dict[n_cal][method] = dict(
                        zip(
                            test_stat_names,
                            [
                                dict(
                                    zip(
                                        observation_dict.keys(),
                                        [[] for _ in observation_dict.keys()],
                                    )
                                )
                                for _ in test_stat_names
                            ],
                        )
                    )
    # Compute results for every run
    # Loop over runs
    for n in range(start_run, n_runs + 1):
        print()
        print("====> RUN: ", n, "/", n_runs, f", N_cal = {n_cal_list} <====")
        # Generate data for a maximum n_cal
        generate_c2st_data = True
        if "c2st" not in methods:
            generate_c2st_data = False
        data_samples = generate_data_one_run(
            n_cal=max(n_cal_list),
            n_eval=n_eval,
            task=task,
            observation_dict=observation_dict,
            n_train_list=[n_train],
            task_path=task_path,
            save_data=False,
            load_cal_data=False,
            load_eval_data=load_eval_data,
            generate_c2st_data=generate_c2st_data,
            seed=n,  # different seed for every run
        )

        # Loop over n_cal
        for n_cal in n_cal_list:
            print()
            print("================")
            print("N_cal = ", n_cal)
            print("================")
            print()
            # Reduce number of calibration samples to n_cal
            data_samples_n = copy.deepcopy(data_samples)
            data_samples_n["base_dist"]["cal"] = data_samples_n["base_dist"]["cal"][
                :n_cal
            ]
            data_samples_n["joint_cal"]["x"] = data_samples_n["joint_cal"]["x"][:n_cal]
            data_samples_n["joint_cal"]["theta"] = data_samples_n["joint_cal"]["theta"][
                :n_cal
            ]

            x_cal = data_samples_n["joint_cal"]["x"]
            dim_theta = data_samples_n["joint_cal"]["theta"].shape[-1]

            data_samples_n["npe_x_cal"][n_train] = data_samples_n["npe_x_cal"][n_train][
                :n_cal
            ]
            data_samples_n["inv_transform_theta_cal"][n_train] = data_samples_n[
                "inv_transform_theta_cal"
            ][n_train][:n_cal]

            # Only generate c2st data if needed (can be expensive to sample from the true posterior)
            if generate_c2st_data:
                data_samples_n["ref_posterior"]["cal"] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["ref_posterior"]["cal"].items()
                }
                data_samples_n["npe_obs"]["cal"][n_train] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["npe_obs"]["cal"][n_train].items()
                }
                data_samples_n["ref_inv_transform"]["cal"][n_train] = {
                    k: v[:n_cal]
                    for k, v in data_samples_n["ref_inv_transform"]["cal"][
                        n_train
                    ].items()
                }

            # Compute test statistics under null for methods dependent of x_cal
            for m, metrics in zip(["lc2st_nf", "lhpd"], [test_stat_names, ["mse"]]):
                if m in methods:
                    if m == "lc2st_nf":
                        kwargs_lc2st_temp = kwargs_lc2st
                        kwargs_lhpd_temp = {}
                    else:
                        kwargs_lc2st_temp = {}
                        kwargs_lhpd_temp = kwargs_lhpd

                    t_stats_null_dict[n_cal][m] = precompute_t_stats_null(
                        n_cal=n_cal,
                        n_eval=n_eval,
                        dim_theta=dim_theta,
                        n_trials_null=n_trials_null_precompute,
                        t_stats_fn_lc2st=t_stats_lc2st,
                        t_stats_fn_lhpd=t_stats_lhpd,
                        kwargs_lc2st=kwargs_lc2st_temp,
                        kwargs_lhpd=kwargs_lhpd_temp,
                        x_cal=x_cal,
                        observation_dict=observation_dict,
                        methods=[m],
                        metrics=metrics,
                        t_stats_null_path=t_stats_null_path,
                        save_results=False,
                        load_results=False,
                        # args only for c2st
                        t_stats_fn_c2st=None,
                        kwargs_c2st=None,
                    )[m]

            # Empirical Power = True Positive Rate (TPR)
            # count rejection of H0 under H1 (p_value <= alpha) for every run
            # and for every observation: [reject(obs1), reject(obs2), ...]
            if compute_emp_power:
                print()
                print("Computing empirical power...")

                # Compute test results for H1
                H1_results_dict, _ = compute_test_results_npe_one_run(
                    alpha=alpha,
                    data_samples=data_samples_n,
                    n_train=n_train,
                    observation_dict=observation_dict,
                    kwargs_c2st=kwargs_c2st,
                    kwargs_lc2st=kwargs_lc2st,
                    kwargs_lhpd=kwargs_lhpd,
                    n_trials_null=n_trials_null,
                    t_stats_null_lc2st_nf=t_stats_null_dict[n_cal]["lc2st_nf"],
                    t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                    t_stats_null_lhpd=t_stats_null_dict[n_cal]["lhpd"],
                    t_stats_null_dict_npe=t_stats_null_dict[n_cal],
                    test_stat_names=test_stat_names,
                    methods=methods,
                    compute_under_null=False,
                    task_path=task_path,
                    results_n_train_path="",
                    save_results=False,
                    seed=n,  # different seed for every run
                )
                # Add results to dicts
                for m in methods:
                    for t_stat_name in test_stat_names:
                        if m == "lhpd" and t_stat_name != "mse":
                            continue
                        p_value_t = H1_results_dict[m]["p_value"][t_stat_name]
                        # Increment list of average rejections of H0 under H1
                        emp_power[n_cal][m][t_stat_name] += (
                            (np.array(p_value_t) <= alpha) * 1 / n_runs
                        )
                        # Increment p_values for every observation
                        for num_obs in observation_dict.keys():
                            p_values[n_cal][m][t_stat_name][num_obs].append(
                                p_value_t[num_obs - 1]
                            )

                    if n % save_every_n_runs == 0:
                        # Set up path to save results
                        result_path_n = result_path / f"n_runs_{n}"
                        if not os.path.exists(result_path_n):
                            os.makedirs(result_path_n)

                        # Save results
                        torch.save(
                            emp_power[n_cal][m],
                            result_path_n
                            / f"emp_power_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                        )
                        torch.save(
                            p_values[n_cal][m],
                            result_path_n
                            / f"p_values_obs_per_run_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                        )

                        # Remove previous results
                        result_path_previous = (
                            result_path / f"n_runs_{n-save_every_n_runs}"
                        )
                        if os.path.exists(result_path_previous):
                            os.remove(
                                result_path_previous
                                / f"emp_power_{m}_n_runs_{n-save_every_n_runs}_n_cal_{n_cal}.pkl"
                            )
                            os.remove(
                                result_path_previous
                                / f"p_values_obs_per_run_{m}_n_runs_{n-save_every_n_runs}_n_cal_{n_cal}.pkl"
                            )

            else:
                emp_power[n_cal], p_values[n_cal] = None, None

            # Type I error = False Positive Rate (FPR)
            # count rejection of H0 under H0 (p_value <= alpha) for every run
            # and for every observation: [reject(obs1), reject(obs2), ...]
            if compute_type_I_error:
                print()
                print("Computing Type I error...")
                print()

                # Fixed distribution for null hypothesis (base distribution)
                from scipy.stats import multivariate_normal as mvn

                # Generate data for L-C2ST-NF (Q_h0 = P_h0 = N(0,1))
                base_dist_samples_null = mvn(
                    mean=torch.zeros(dim_theta), cov=torch.eye(dim_theta)
                ).rvs(
                    n_cal, random_state=n + 1
                )  # not same random state as for other data generation (we dont want same data for P and Q)

                # Compatible with torch data
                base_dist_samples_null = torch.FloatTensor(base_dist_samples_null)

                # Compute test results for H0
                H0_results_dict, _ = compute_test_results_npe_one_run(
                    alpha=alpha,
                    data_samples=data_samples_n,
                    n_train=n_train,
                    observation_dict=observation_dict,
                    kwargs_c2st=kwargs_c2st,
                    kwargs_lc2st=kwargs_lc2st,
                    kwargs_lhpd=kwargs_lhpd,
                    n_trials_null=n_trials_null,
                    t_stats_null_lc2st_nf=t_stats_null_dict[n_cal]["lc2st_nf"],
                    t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                    t_stats_null_lhpd=t_stats_null_dict[n_cal]["lhpd"],
                    t_stats_null_dict_npe=t_stats_null_dict[n_cal],
                    test_stat_names=test_stat_names,
                    methods=methods,
                    compute_under_null=True,
                    base_dist_samples_null=base_dist_samples_null,
                    task_path=task_path,
                    results_n_train_path="",
                    save_results=False,
                    seed=n,  # different seed for every run
                )
                # Add results to dicts
                for m in methods:
                    for t_stat_name in test_stat_names:
                        if m == "lhpd" and t_stat_name != "mse":
                            continue

                        p_value_t = H0_results_dict[m]["p_value"][t_stat_name]
                        # Increment list of average rejections of H0 under H0
                        type_I_error[n_cal][m][t_stat_name] += (
                            (np.array(p_value_t) <= alpha) * 1 / n_runs
                        )
                        # Increment p_value for every observation
                        for num_obs in observation_dict.keys():
                            p_values_h0[n_cal][m][t_stat_name][num_obs].append(
                                p_value_t[num_obs - 1]
                            )
                    if n % save_every_n_runs == 0:
                        # Set up path to save results
                        result_path_n = result_path / f"n_runs_{n}"
                        if not os.path.exists(result_path_n):
                            os.makedirs(result_path_n)
                        # Save results
                        torch.save(
                            type_I_error[n_cal][m],
                            result_path_n
                            / f"type_I_error_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                        )
                        torch.save(
                            p_values_h0[n_cal][m],
                            result_path_n
                            / f"p_values_h0__obs_per_run_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                        )

                        # Remove previous results
                        result_path_previous = (
                            result_path / f"n_runs_{n-save_every_n_runs}"
                        )
                        if os.path.exists(result_path_previous):
                            os.remove(
                                result_path_previous
                                / f"type_I_error_{m}_n_runs_{n-save_every_n_runs}_n_cal_{n_cal}.pkl"
                            )
                            os.remove(
                                result_path_previous
                                / f"p_values_h0__obs_per_run_{m}_n_runs_{n-save_every_n_runs}_n_cal_{n_cal}.pkl"
                            )

            else:
                type_I_error[n_cal] = None
                p_values_h0[n_cal] = None

    return emp_power, type_I_error, p_values, p_values_h0


def generate_data_one_run(
    n_cal,
    n_eval,
    task,
    observation_dict,
    task_path,
    n_train_list,
    save_data=True,
    load_cal_data=True,
    load_eval_data=True,
    generate_c2st_data=True,
    seed=42,  # fixed seed for reproducibility
    task_observations=True,
):
    """Generate data for one run of the test.

    Args:
        n_cal (int): number of calibration samples.
        n_eval (int): number of evaluation samples.
        task (str): sbibm task name.
        observation_dict (dict): dict of observations.
            keys are observation numbers
            values are torch tensors of shape (1, dim_x)
        task_path (str): path to the task folder.
        n_train_list (List[int]): number of training samples for the npe.
        save_data (bool): whether to save the generated data.
            Defaults to True.
        load_cal_data (bool): whether to load the calibration data from disk.
            Defaults to True.
        load_eval_data (bool): whether to load the evaluation data from disk.
            Defaults to True.
        generate_c2st_data (bool): whether to generate the data for c2st.
            Defaults to True.
        seed (int): seed for reproducibility.

    Returns:
        data_samples (dict): dict of data samples.
    """

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load posterior estimator (NF)
    # trained using the code from https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi
    # >>> python run.py --multirun task={task} task.num_simulations={n_train_list} algorithm=npe
    npe = {}
    for N_train in n_train_list:
        npe[N_train] = torch.load(
            task_path / f"npe_{N_train}" / f"posterior_estimator.pkl"
        ).flow
    print(f"Loaded npe posterior estimator trained on {n_train_list} samples.")
    print()

    # Get base distribution (same for all npes)
    base_dist = npe[n_train_list[0]].posterior_estimator._distribution

    # Generate / load calibration and evaluation datasets
    print(" ==========================================")
    print("     Generating / loading datasets")
    print(" ==========================================")
    print()
    # Calibration set for fixed task data
    print(f"Calibration set for fixed task data (n_cal={n_cal})")
    try:
        if not load_cal_data:
            raise FileNotFoundError
        # Base distribution samples for NF methods
        base_dist_samples_cal = torch.load(
            task_path / f"base_dist_samples_n_cal_{n_cal}.pkl"
        )
        # Joint samples
        joint_samples_cal = torch.load(task_path / f"joint_samples_n_cal_{n_cal}.pkl")
        theta_cal = joint_samples_cal["theta"]
        x_cal = joint_samples_cal["x"]
        # Only load reference posterior samples if needed
        if generate_c2st_data:
            reference_posterior_samples_cal = torch.load(
                task_path / f"reference_posterior_samples_n_cal_{n_cal}.pkl"
            )
        else:
            reference_posterior_samples_cal = None
    except FileNotFoundError:
        # Generate data
        base_dist_samples_cal = base_dist.sample(n_cal).detach()

        if task_observations:
            num_observation_list = list(observation_dict.keys())
        else:
            num_observation_list = None
        reference_posterior_samples_cal, theta_cal, x_cal = generate_task_data(
            n_cal,
            task,
            num_observation_list=num_observation_list,
            observation_list=list(observation_dict.values()),
            sample_from_reference=generate_c2st_data,  # only sample from reference if needed
        )
        joint_samples_cal = {"theta": theta_cal, "x": x_cal}
        # Save data
        if save_data:
            torch.save(
                base_dist_samples_cal,
                task_path / f"base_dist_samples_n_cal_{n_cal}.pkl",
            )
            if generate_c2st_data:
                torch.save(
                    reference_posterior_samples_cal,
                    task_path / f"reference_posterior_samples_n_cal_{n_cal}.pkl",
                )
            torch.save(
                joint_samples_cal, task_path / f"joint_samples_n_cal_{n_cal}.pkl",
            )

    # Eval set for fixed task data (no joint samples)
    print()
    print(f"Evaluation set for fixed task data (n_eval={n_eval})")
    try:
        if not load_eval_data:
            raise FileNotFoundError
        # Base distribution samples for NF methods
        base_dist_samples_eval = torch.load(
            task_path / f"base_dist_samples_n_eval_{n_eval}.pkl"
        )
        # Only load reference posterior samples if needed
        if generate_c2st_data:
            reference_posterior_samples_eval = torch.load(
                task_path / f"reference_posterior_samples_n_eval_{n_eval}.pkl"
            )
        else:
            reference_posterior_samples_eval = None
    except FileNotFoundError:
        # Generate data
        if task_observations:
            num_observation_list = list(observation_dict.keys())
        else:
            num_observation_list = None
        reference_posterior_samples_eval, _, _ = generate_task_data(
            n_eval,
            task,
            num_observation_list=num_observation_list,
            observation_list=list(observation_dict.values()),
            sample_from_joint=False,
            sample_from_reference=generate_c2st_data,  # only sample from reference if needed
        )
        base_dist_samples_eval = base_dist.sample(n_eval).detach()
        # Save data
        if save_data:
            if generate_c2st_data:
                torch.save(
                    reference_posterior_samples_eval,
                    task_path / f"reference_posterior_samples_n_eval_{n_eval}.pkl",
                )
            torch.save(
                base_dist_samples_eval,
                task_path / f"base_dist_samples_n_eval_{n_eval}.pkl",
            )

    # Calibration and eval set for every npe
    print()
    print(
        f"Calibration and evaluation sets for every estimator (n_cal={n_cal}, n_eval={n_eval})"
    )
    npe_samples_obs = {"cal": {}, "eval": {}}
    reference_inv_transform_samples_cal = {}
    reference_inv_transform_samples_eval = {}
    npe_samples_x_cal = {}
    inv_transform_samples_theta_cal = {}
    # Loop over n_train
    for N_train in n_train_list:
        print()
        print(f"Data for npe with N_train = {N_train}:")
        npe_samples_obs["cal"][N_train] = {}
        reference_inv_transform_samples_cal[N_train] = {}

        npe_path = task_path / f"npe_{N_train}"
        # ==== C2ST calibration dataset ==== #
        print("     1. C2ST: at fixed observation x_0")
        if generate_c2st_data:  # only generate if needed
            try:
                if not load_cal_data:
                    raise FileNotFoundError
                # NPE samples for at fixed observation x_0
                npe_samples_obs["cal"][N_train] = torch.load(
                    npe_path / f"npe_samples_obs_n_cal_{n_cal}.pkl"
                )
                # Inverse transform samples at fixed observation x_0
                reference_inv_transform_samples_cal[N_train] = torch.load(
                    npe_path / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl"
                )
            except FileNotFoundError:
                # Generate data
                (
                    npe_samples_obs["cal"][N_train],
                    reference_inv_transform_samples_cal[N_train],
                ) = generate_npe_data_for_c2st(
                    npe[N_train],
                    base_dist_samples_cal,
                    reference_posterior_samples_cal,
                    list(observation_dict.values()),
                    list(observation_dict.keys()),
                )
                # Save data
                if save_data:
                    torch.save(
                        npe_samples_obs["cal"][N_train],
                        npe_path / f"npe_samples_obs_n_cal_{n_cal}.pkl",
                    )
                    torch.save(
                        reference_inv_transform_samples_cal[N_train],
                        npe_path / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl",
                    )
        else:
            npe_samples_obs["cal"][N_train] = {
                num_obs: None for num_obs in observation_dict.keys()
            }
            reference_inv_transform_samples_cal[N_train] = {
                num_obs: None for num_obs in observation_dict.keys()
            }

        try:
            if not load_eval_data:
                raise FileNotFoundError
            # NPE samples for every observation x_0 (used in L-C2ST)
            npe_samples_obs["eval"][N_train] = torch.load(
                npe_path / f"npe_samples_obs_n_eval_{n_eval}.pkl"
            )
            # Inverse transform samples for every observation x_0
            if generate_c2st_data:  # only generate if needed
                reference_inv_transform_samples_eval[N_train] = torch.load(
                    npe_path / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl"
                )
            else:
                reference_inv_transform_samples_eval[N_train] = {
                    num_obs: None for num_obs in observation_dict.keys()
                }
        except FileNotFoundError:
            # Generate data
            (
                npe_samples_obs["eval"][N_train],
                reference_inv_transform_samples_eval[N_train],
            ) = generate_npe_data_for_c2st(
                npe[N_train],
                base_dist_samples_eval,
                reference_posterior_samples_eval,
                list(observation_dict.values()),
                list(observation_dict.keys()),
                nf_case=(not generate_c2st_data),  # only generate if needed
            )
            # Save data
            if save_data:
                torch.save(
                    npe_samples_obs["eval"][N_train],
                    npe_path / f"npe_samples_obs_n_eval_{n_eval}.pkl",
                )
                if generate_c2st_data:
                    torch.save(
                        reference_inv_transform_samples_eval[N_train],
                        npe_path
                        / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl",
                    )

        # ==== L-C2ST calibration dataset ==== #
        print("     2. L-C2ST: for every x in x_cal")
        try:
            if not load_cal_data:
                raise FileNotFoundError
            # NPE samples for every x in x_cal
            npe_samples_x_cal[N_train] = torch.load(
                npe_path / f"npe_samples_x_cal_{n_cal}.pkl"
            )
            # Inverse transform samples for every x in x_cal
            inv_transform_samples_theta_cal[N_train] = torch.load(
                npe_path / f"inv_transform_samples_theta_cal_{n_cal}.pkl"
            )
        except FileNotFoundError:
            # Generate data
            (
                npe_samples_x_cal[N_train],
                inv_transform_samples_theta_cal[N_train],
            ) = generate_npe_data_for_lc2st(
                npe[N_train], base_dist_samples_cal, joint_samples_cal
            )
            # Save data
            if save_data:
                torch.save(
                    npe_samples_x_cal[N_train],
                    npe_path / f"npe_samples_x_cal_{n_cal}.pkl",
                )
                torch.save(
                    inv_transform_samples_theta_cal[N_train],
                    npe_path / f"inv_transform_samples_theta_cal_{n_cal}.pkl",
                )

    # Add generated data to dict
    base_dist_samples = {"cal": base_dist_samples_cal, "eval": base_dist_samples_eval}
    reference_posterior_samples = {
        "cal": reference_posterior_samples_cal,
        "eval": reference_posterior_samples_eval,
    }
    npe_samples_obs = {"cal": npe_samples_obs["cal"], "eval": npe_samples_obs["eval"]}
    reference_inv_transform_samples = {
        "cal": reference_inv_transform_samples_cal,
        "eval": reference_inv_transform_samples_eval,
    }

    data_dict = {
        "base_dist": base_dist_samples,
        "ref_posterior": reference_posterior_samples,
        "npe_obs": npe_samples_obs,
        "ref_inv_transform": reference_inv_transform_samples,
        "joint_cal": joint_samples_cal,
        "npe_x_cal": npe_samples_x_cal,
        "inv_transform_theta_cal": inv_transform_samples_theta_cal,
    }

    return data_dict


def compute_test_results_npe_one_run(
    data_samples,
    n_train,
    observation_dict,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    n_trials_null,
    t_stats_null_c2st_nf,
    t_stats_null_lc2st_nf,
    t_stats_null_lhpd,
    t_stats_null_dict_npe,
    task_path,
    results_n_train_path,
    test_stat_names=["accuracy", "mse", "div"],
    methods=["c2st", "lc2st", "lc2st_nf", "lhpd"],
    alpha=0.05,
    compute_under_null=False,
    base_dist_samples_null=None,
    return_t_stats_null=False,
    save_results=True,
    load_results=True,
    seed=42,  # fix seed for reproducibility
):
    """Compute test results for one run of the test.
    All methods use the permutation method, except for L-C2ST-NF and local-HPD, which
        use precomputed asymtotic approximations of the null distributions that can be used for any
        new NPE.

    Args:
        data_samples (dict): dict of data samples as returned by generate_data_one_run.
        n_train (int): number of training samples for the npe.
        observation_dict (dict): dict of observations.
            keys are observation numbers
            values are torch tensors of shape (1, dim_x)
        kwargs_c2st (dict): kwargs for c2st.
        kwargs_lc2st (dict): kwargs for lc2st.
        kwargs_lhpd (dict): kwargs for lhpd.
        n_trials_null (int): number of trials for the permutation test.
        t_stats_null_c2st_nf (dict): dict of null test statistics for c2st_nf.
        t_stats_null_lc2st_nf (dict): dict of null test statistics for lc2st_nf.
        t_stats_null_lhpd (dict): dict of null test statistics for lhpd.
        t_stats_null_dict_npe (dict): dict of null test statistics for npe dependent methods.
        task_path (str): path to the task folder.
        results_n_train_path (str): path to the results folder for the considered n_train.
        test_stat_names (List[str]): list of test statistic names.
            Must be compatible with scores functions of the considered methods.
            Defaults to ["accuracy", "mse", "div"].
        methods (List[str]): list of methods.
            Defaults to ["c2st", "lc2st", "lc2st_nf", "lhpd"].
        alpha (float): significance level.
            Defaults to 0.05.
        compute_under_null (bool): whether to compute the test under the null hypothesis.
            Defaults to False.
        base_dist_samples_null (torch tensor): samples from the base distribution
            for the NF null hypothesis.
            Defaults to None.
        return_t_stats_null (bool): whether to return the null test statistics.
            Defaults to False.
        save_results (bool): whether to save the results.
            Defaults to True.
        load_results (bool): whether to load the results.
            Defaults to True.
        seed (int): seed for reproducibility.

    Returns:
        results_dict (dict): dict of test results for every method, target and test statistic
            whose values are lists of length n_obs.
        train_runtime (dict): dict of training runtimes for every method
            whose values are floats.
        t_stats_null_dict (dict): dict of null test statistics for every method, test statistics and observation
            whose values are lists of length n_trials_null(_precompute).
    """

    # Extract data samples independent from estimator
    base_dist_samples = data_samples["base_dist"]
    reference_posterior_samples = data_samples["ref_posterior"]
    theta_cal, x_cal = data_samples["joint_cal"].values()
    # Extract data samples for the considered "n_train"-estimator (npe)
    npe_samples_obs = {k: v[n_train] for k, v in data_samples["npe_obs"].items()}
    reference_inv_transform_samples = {
        k: v[n_train] for k, v in data_samples["ref_inv_transform"].items()
    }
    npe_samples_x_cal = data_samples["npe_x_cal"][n_train]
    inv_transform_samples_theta_cal = data_samples["inv_transform_theta_cal"][n_train]

    # Get dataset sizes
    n_cal = len(base_dist_samples["cal"])
    n_eval = len(base_dist_samples["eval"])

    if list(observation_dict.keys())[0] is None:
        observation_dict = {i + 1: v for i, v in enumerate(observation_dict.values())}

    print()
    print(" ==========================================")
    print("     COMPUTING TEST RESULTS")
    print(" ==========================================")
    print()
    print(f"N_train = {n_train}")

    # Set up paths to save results
    result_path = task_path / f"npe_{n_train}" / results_n_train_path
    if compute_under_null:
        result_path = result_path / "null"
    if save_results and not os.path.exists(result_path):
        os.makedirs(result_path)

    t_stats_null_path = task_path / f"npe_{n_train}" / "t_stats_null"

    # Initialize dicts
    train_runtime = dict(zip(methods, [0 for _ in methods]))
    results_dict = dict(zip(methods, [{} for _ in methods]))
    t_stats_null_dict = {
        m: {num_obs: {} for num_obs in observation_dict.keys()} for m in methods
    }
    # Define result keys
    result_keys = ["reject", "p_value", "t_stat", "t_stats_null", "run_time"]
    trained_clfs_lc2st_nf = None
    runtime_lc2st_nf = None
    # Loop over methods
    for m in methods:
        # Initialize results dict
        results_dict[m] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )
        try:
            # Load results if available ...
            if load_results:
                results_dict[m] = torch.load(
                    result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
                if "l" in m:
                    train_runtime[m] = torch.load(
                        result_path / f"runtime_{m}_n_cal_{n_cal}.pkl"
                    )
            if return_t_stats_null:
                t_stats_null_dict[m] = torch.load(
                    t_stats_null_path
                    / f"t_stats_null_{m}_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
            print()
            print(f"Loaded {m} results for n_eval={n_eval}, n_cal={n_cal}")
            print()
        except FileNotFoundError:
            # ... otherwise compute results
            if m == "c2st" or m == "c2st_nf":
                print()
                print("     C2ST: train for every observation x_0")
                print()
                # Loop over observations x_0
                for n_obs in tqdm(
                    observation_dict.keys(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    if m == "c2st":
                        # Class 0 vs. class 1: T ~ q(theta | x_0) vs. p(theta | x_0)
                        P, Q = (
                            npe_samples_obs["cal"][n_obs],
                            reference_posterior_samples["cal"][n_obs],
                        )
                        P_eval, Q_eval = (
                            npe_samples_obs["eval"][n_obs],
                            reference_posterior_samples["eval"][n_obs],
                        )
                        # If computing under null, permute data
                        if compute_under_null:
                            P, Q = permute_data(P, Q, seed=seed)
                            P_eval, Q_eval = permute_data(P_eval, Q_eval, seed=seed)
                        # Get precomputed test stats under null if available
                        if t_stats_null_dict_npe["c2st"] is None:
                            t_stats_null = None
                        else:
                            t_stats_null = t_stats_null_dict_npe["c2st"][n_obs]

                    elif m == "c2st_nf":
                        # Class 0 vs. class 1: Z ~ N(0,I) vs. Z ~ p(T^{-1}(theta,x_0) | x_0)
                        P, Q = (
                            base_dist_samples["cal"],
                            reference_inv_transform_samples["cal"][n_obs],
                        )
                        P_eval, Q_eval = (
                            base_dist_samples["eval"],
                            reference_inv_transform_samples["eval"][n_obs],
                        )
                        # No permuation method and precomputed test stats under null
                        if compute_under_null:
                            Q = base_dist_samples_null  # the null hypothesis is known (P=Q independent of x)
                        t_stats_null = t_stats_null_c2st_nf

                    # Evaluate the test
                    t0 = time.time()
                    c2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_c2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null,
                        # kwargs for t_stats_c2st
                        P=P,
                        Q=Q,
                        P_eval=P_eval,
                        Q_eval=Q_eval,
                        use_permutation=True,
                        n_trials_null=n_trials_null,
                        # kwargs for c2st_scores
                        **kwargs_c2st,
                    )
                    runtime = time.time() - t0

                    # Add results to dict
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict[m][result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                results_dict[m][result_name][t_stat_name].append(
                                    c2st_results_obs[i][t_stat_name]
                                )

            elif "lc2st" in m:
                print()
                print("     L-C2ST: amortized")
                print()

                x_P, x_Q = x_cal, x_cal  # same context datafor both classes

                if m == "lc2st":
                    # Class 0 vs. class 1: T,X ~ q(theta, x) vs. p(theta, x)
                    P, Q = npe_samples_x_cal, theta_cal
                    P_eval_obs = npe_samples_obs["eval"]

                if m == "lc2st_nf" or m == "lc2st_nf_perm":
                    # Class 0 vs. class 1: Z,X ~ N(0,I)p(x) vs. p(T^{-1}(theta,x), x)
                    P, Q = base_dist_samples["cal"], inv_transform_samples_theta_cal
                    P_eval_obs = {
                        n_obs: base_dist_samples["eval"]
                        for n_obs in observation_dict.keys()
                    }
                if m == "lc2st" or m == "lc2st_nf_perm":
                    # Permutation method and no precomputed test stats under null
                    t_stats_null = t_stats_null_dict_npe[m]
                    if compute_under_null:
                        joint_P_x = torch.cat([P, x_P], dim=1)
                        joint_Q_x = torch.cat([Q, x_Q], dim=1)
                        joint_P_x, joint_Q_x = permute_data(
                            joint_P_x, joint_Q_x, seed=seed
                        )
                        P, x_P = (
                            joint_P_x[:, : P.shape[-1]],
                            joint_P_x[:, P.shape[-1] :],
                        )
                        Q, x_Q = (
                            joint_Q_x[:, : Q.shape[-1]],
                            joint_Q_x[:, Q.shape[-1] :],
                        )
                else:
                    # No permutation method and precomputed test stats under null
                    t_stats_null = t_stats_null_lc2st_nf
                    if compute_under_null:
                        Q = base_dist_samples_null

                # Train classifier on the joint
                print(f"{m}: TRAINING CLASSIFIER on the joint ...")
                print()
                print("... for the observed data")
                if m == "lc2st" or compute_under_null or trained_clfs_lc2st_nf is None:
                    t0 = time.time()
                    _, _, trained_clfs_lc2st = lc2st_scores(
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        x_eval=None,
                        eval=False,
                        **kwargs_lc2st,
                    )
                    runtime = time.time() - t0
                    train_runtime[m] = runtime
                    # Use same classifier trained on observed data for lc2st_nf and lc2st_nf_perm
                    # --> they only differ in the computation of the null hypothesis
                    if "lc2st_nf" in m and not compute_under_null:
                        trained_clfs_lc2st_nf = trained_clfs_lc2st
                        runtime_lc2st_nf = runtime
                    train_runtime[m] = runtime
                else:
                    print("     Using classifier trained for lc2st_nf method")
                    trained_clfs_lc2st = trained_clfs_lc2st_nf
                    runtime = runtime_lc2st_nf
                    train_runtime[m] = runtime
                if save_results:
                    torch.save(runtime, result_path / f"runtime_{m}_n_cal_{n_cal}.pkl")

                if t_stats_null is None:
                    print("... under the null hypothesis")
                    # Train classifier on the joint under null
                    _, _, trained_clfs_null_lc2st = t_stats_lc2st(
                        null_hypothesis=True,
                        n_trials_null=n_trials_null,
                        use_permutation=True,
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        x_eval=None,
                        P_eval=None,
                        return_clfs_null=True,
                        # kwargs for lc2st_sores
                        eval=False,
                        **kwargs_lc2st,
                    )
                    t_stats_null = {n_obs: None for n_obs in observation_dict.keys()}
                else:
                    trained_clfs_null_lc2st = None

                # Evaluate the test
                # Loop over observations x_0
                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    # Evaluate the test
                    t0 = time.time()
                    lc2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lc2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null[num_observation],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        P_eval=P_eval_obs[num_observation],
                        use_permutation=True,
                        n_trials_null=n_trials_null,
                        return_probas=False,
                        # unnessary args as we have pretrained clfs
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lc2st,
                        trained_clfs_null=trained_clfs_null_lc2st,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
                    runtime = time.time() - t0  # / n_trials_null

                    # Add results to dict
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict[m][result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                results_dict[m][result_name][t_stat_name].append(
                                    lc2st_results_obs[i][t_stat_name]
                                )

            elif m == "lhpd":
                print()
                print("     Local HPD: amortized")
                print()

                # Get estimator (needed to compute HPD values)
                npe = torch.load(
                    task_path / f"npe_{n_train}" / "posterior_estimator.pkl"
                ).flow

                # Define sample and log prob functions for the estimator (needed to compute HPD values)
                def npe_sample_fn(n_samples, x):
                    npe.set_default_x(x)
                    return sample_from_npe_obs(npe, x, n_samples=n_samples)

                def npe_log_prob_fn(theta, x):
                    npe.set_default_x(x)
                    return npe.log_prob(theta, x)

                print(f"{m}: TRAINING CLASSIFIER on the joint ...")
                print()
                print("... for the observed data")
                t0 = time.time()
                if compute_under_null:
                    _, _, trained_clfs_lhpd = t_stats_lhpd(
                        null_hypothesis=True,
                        n_trials_null=1,
                        Y=theta_cal,
                        X=x_cal,
                        x_eval=None,
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        eval=False,
                        return_clfs_null=True,
                        **kwargs_lhpd,
                    )
                    trained_clfs_lhpd = trained_clfs_lhpd[0]
                else:
                    _, _, trained_clfs_lhpd = lhpd_scores(
                        Y=theta_cal,
                        X=x_cal,
                        est_log_prob_fn=npe_log_prob_fn,
                        est_sample_fn=npe_sample_fn,
                        return_clfs=True,
                        x_eval=None,
                        eval=False,
                        **kwargs_lhpd,
                    )
                runtime = time.time() - t0
                train_runtime[m] = runtime

                # Evaluate the test
                # Loop over observations x_0
                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    # Evaluate the test
                    t0 = time.time()
                    lhpd_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lhpd,
                        metrics=["mse"],
                        t_stats_null=t_stats_null_lhpd[num_observation],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        Y=theta_cal,
                        X=x_cal,
                        n_trials_null=n_trials_null,
                        return_r_alphas=False,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lhpd,
                        # kwargs for lhpd_scores
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        **kwargs_lhpd,
                    )
                    runtime = time.time() - t0  # / n_trials_null

                    # Add results to dict
                    for i, result_name in enumerate(result_keys):
                        if result_name == "run_time":
                            results_dict[m][result_name]["mse"].append(runtime)
                        else:
                            results_dict[m][result_name]["mse"].append(
                                lhpd_results_obs[i]["mse"]
                            )

            # Save results for the given method
            if save_results:
                torch.save(
                    results_dict[m],
                    result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    train_runtime[m], result_path / f"runtime_{m}_n_cal_{n_cal}.pkl"
                )

            # Add null test statistics to dict
            for t_stat_name in test_stat_names:
                if m == "lhpd" and t_stat_name != "mse":
                    continue
                for i, num_obs in enumerate(observation_dict.keys()):
                    t_stats_null_dict[m][num_obs][t_stat_name] = results_dict[m][
                        "t_stats_null"
                    ][t_stat_name][i]

            # Save null test statistics for the given method
            if save_results:
                if not os.path.exists(t_stats_null_path):
                    os.makedirs(t_stats_null_path)
                torch.save(
                    t_stats_null_dict[m],
                    t_stats_null_path
                    / f"t_stats_null_{m}_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

    if return_t_stats_null:
        return results_dict, train_runtime, t_stats_null_dict
    else:
        return results_dict, train_runtime


def compute_rejection_rates_from_pvalues_over_runs_and_observations(
    n_runs,
    alpha,
    num_observation_list,
    p_values_dict,
    p_values_h0_dict=None,
    compute_tpr=True,
    compute_fpr=False,
    bonferonni_correction=False,
):
    """Compute rejection rates from p-values over runs and observations for a given method and test statistic.

    Args:
        n_runs (int): number of runs.
        alpha (float): significance level.
        num_observation_list (List[int]): list of observation numbers.
        p_values_dict (dict): dict of p-values for each observation number and run.
        p_values_h0_dict (dict): dict of p-values under the null hypothesis for each observation number and run.
            Defaults to None.
        compute_tpr (bool): whether to compute the true positive rate (empirical power).
            Defaults to True.
        compute_fpr (bool): whether to compute the false positive rate (type I error).
            Defaults to False.
        bonferonni_correction (bool): whether to apply the Bonferonni correction.
            Defaults to False.

    Returns:
        emp_power_list (List[List[float]]): list of lists of empirical power values.
            axis 0: runs
            axis 1: observations (size 1 if Bonferonni correction is applied)
        type_I_error_list (List[List[float]]): list of lists of type I error values.
            axis 0: runs
            axis 1: observations (size 1 if Bonferonni correction is applied)
    """
    emp_power_list = []
    type_I_error_list = []
    # Loop over runs
    for n_r in range(n_runs):
        for result_list, p_values, compute in zip(
            [emp_power_list, type_I_error_list],
            [p_values_dict, p_values_h0_dict],
            [compute_tpr, compute_fpr],
        ):
            if compute and p_values is not None:
                # Get p-values for the considered run
                p_value_n_r = np.array(
                    [p_values[n_obs][n_r] for n_obs in num_observation_list]
                )
                # Apply Bonferonni correction and append result...
                if bonferonni_correction:
                    result_list.append(
                        [np.any(p_value_n_r <= alpha / len(p_value_n_r))]
                    )
                # ... or just append the result
                else:
                    result_list.append((np.array(p_value_n_r) <= alpha) * 1)

    return emp_power_list, type_I_error_list


def compute_average_rejection_rates(
    result_dict, mean_over_runs, mean_over_observations
):
    """Compute average rejection rates from a dict of results for a given method and test statistic.

    Args:
        result_dict (dict): dict of results (rejected or not).
            axis 0: runs
            axis 1: observations

    Returns:
        result_list (np.array): array of average rejection rates.

    """
    # Over runs (for each observation)
    if mean_over_runs:
        result_list = np.mean(result_dict, axis=0)
    # Over observations (for each run)
    elif mean_over_observations:
        result_list = np.mean(result_dict, axis=1)
    else:
        result_list = result_dict
    return result_list


if __name__ == "__main__":
    import torch
    import sbibm
    from lc2st.lhpd import hpd_values, t_stats_lhpd
    from tasks.sbibm.npe_utils import sample_from_npe_obs
    from lc2st.lc2st import sbibm_clf_kwargs

    import matplotlib.pyplot as plt

    task = sbibm.get_task("slcp")
    npe = torch.load(
        "saved_experiments/neurips2023/exp_2/slcp/npe_100/posterior_estimator.pkl"
    ).flow
    joint_samples = torch.load(
        "saved_experiments/neurips2023/exp_2/slcp/joint_samples_n_cal_10000.pkl"
    )
    x, theta = joint_samples["x"][:100], joint_samples["theta"][:100]
    observation = task.get_observation(1)

    def sample_fn(n_samples, x):
        npe.set_default_x(x)
        return sample_from_npe_obs(npe, x, n_samples)

    def log_prob_fn(theta, x):
        npe.set_default_x(x)
        return npe.log_prob(theta, x)

    joint_hpd_values = hpd_values(theta, log_prob_fn, sample_fn, x)
    # hpd_values[-1] = 1
    # alphas = np.linspace(0.1, 0.9, 11)
    # for alpha in alphas:
    #     print((joint_hpd_values <= alpha).sum())

    kwargs = sbibm_clf_kwargs(theta.shape[-1])
    kwargs["early_stopping"] = False

    t_stat, r_alphas = t_stats_lhpd(
        Y=theta,
        X=x,
        n_alphas=11,
        x_eval=observation,
        est_log_prob_fn=npe.log_prob,
        est_sample_fn=sample_fn,
        return_r_alphas=True,
        joint_hpd_values=joint_hpd_values,
        clf_kwargs=kwargs,
    )

    t_stats_null, r_alphas_null = t_stats_lhpd(
        null_hypothesis=True,
        X=x[:100],
        Y=theta[:100],
        n_trials_null=10,
        x_eval=observation,
        return_r_alphas=True,
        est_log_prob_fn=None,
        est_sample_fn=None,
        **{"clf_kwargs": kwargs, "n_alphas": 11},
    )

    from lc2st.test_utils import compute_pvalue

    pvalue = compute_pvalue(t_stat["mse"], t_stats_null["mse"])
    print(t_stat)
    print(pvalue)

    alphas = list(r_alphas.keys())
    alphas = np.concatenate([alphas, [1]])
    r_alphas = {**r_alphas, **{1.0: 1.0}}
    plt.plot(alphas, r_alphas.values())

    import pandas as pd

    r_alphas_null = {**r_alphas_null, **{len(alphas): [1.0] * 10}}
    lower_band = pd.DataFrame(r_alphas_null).quantile(q=0.05 / 2, axis=0)
    upper_band = pd.DataFrame(r_alphas_null).quantile(q=1 - 0.05 / 2, axis=0)

    plt.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)
    plt.show()
