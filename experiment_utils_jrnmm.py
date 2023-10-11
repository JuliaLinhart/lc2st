# JRNMM experiment set-up and functions
#
# Using code from the hnpe folder copied from https://github.com/plcrodrigues/HNPE
# and posterior.py is taken from https://github.com/plcrodrigues/HNPE/Ex2-JRNMM/
#
# Prior, simulator and summary extractor are defined in the tasks.jrnmm folder
# also taken from https://github.com/plcrodrigues/HNPE
# --> using the simulator (in training the flow and generating observations)
# requires a special R-environment.

# IMPORTS
import numpy as np
import os
import time
import torch

from functools import partial
from hnpe.misc import make_label
from hnpe.inference import run_inference
from hnpe.posterior import build_flow, IdentityJRNMM, get_posterior
from lc2st.lhpd import hpd_values, lhpd_scores, t_stats_lhpd
from lc2st.lc2st import lc2st_scores, t_stats_lc2st
from lc2st.test_utils import eval_htest
from sbi.analysis.sbc import run_sbc
from tasks.jrnmm.summary import summary_JRNMM
from tqdm import tqdm

# GLOBAL VARIABLES

# We train for one round : amortized
NB_ROUNDS = 1

# Choose the naive or factorized flow
NAIVE = True
N_EXTRA = 0

# List of (trec, nextra)
LIST_TREC_NEXTRA = [(8, 0)]  # , (8,9)] #[(2,3), (8,10)]   #(8,3), (2,10) #(8,4) (2,4)

# Extra obs have all parameters in common with x_0 or only the global one (gain)
# during inference only
LIST_SINGLE_REC = [False]


# Target folder path inside results folder for saving
PATH_EXPERIMENT = "saved_experiments/JR-NMM/normal_4d/"

# Setup the parameters for the example
meta_parameters = {}
meta_parameters["n_extra"] = N_EXTRA  # how many extra observations to consider
meta_parameters["summary"] = "Fourier"  # what kind of summary features to use
meta_parameters["naive"] = NAIVE  # whether to do naive implementation
meta_parameters["n_rd"] = NB_ROUNDS  # number of rounds to use in the SNPE procedure
meta_parameters["n_sf"] = 33  # number of summary features to consider
meta_parameters["t_recording"] = 8  # seconds for the simulations (fs = 128 Hz)
meta_parameters["n_ss"] = int(128 * meta_parameters["t_recording"])

# which example case we are considering here
meta_parameters["case"] = (
    PATH_EXPERIMENT + "JRNMM_nextra_{:02}_trec_{}"
    "naive_{}_"
    "single_rec_False".format(
        meta_parameters["n_extra"],
        meta_parameters["t_recording"],
        meta_parameters["naive"],
    )
)

# Choose how to get the summary features
summary_extractor = summary_JRNMM(
    n_extra=meta_parameters["n_extra"],
    d_embedding=meta_parameters["n_sf"],
    n_time_samples=meta_parameters["n_ss"],
    type_embedding=meta_parameters["summary"],
)

# We use the log power spectral density as summary features
summary_extractor.embedding.net.logscale = True


def train_posterior_jrnmm(n_train):
    """Train a posterior (NPE) for the JRNMM model.

    Args:
        n_train (int): number of simulations to use for training

    Returns:
        npe_jrnmm (DirectPosterior): trained posterior via `sbi`-package
    """
    # IMPORTS (require R-environment)
    from tasks.jrnmm.simulator import prior_JRNMM, simulator_JRNMM

    # Set prior distribution for the parameters
    input_parameters = ["C", "mu", "sigma", "gain"]
    prior = prior_JRNMM(
        parameters=[
            ("C", 10.0, 250.0),
            ("mu", 50.0, 500.0),
            ("sigma", 100.0, 5000.0),
            ("gain", -20.0, +20.0),
        ]
    )

    # Choose how to setup the simulator for training
    simulator = partial(
        simulator_JRNMM,
        input_parameters=input_parameters,
        t_recording=meta_parameters["t_recording"],
        n_extra=meta_parameters["n_extra"],
        p_gain=prior,
    )

    # Number of simulations per training round
    meta_parameters["n_sr"] = n_train

    # Label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)

    # Device to use for training
    device = "cpu"

    # Define function which creates a neural network density estimator
    build_nn_posterior = partial(
        build_flow,
        embedding_net=IdentityJRNMM(),
        naive=meta_parameters["naive"],
        aggregate=True,
        z_score_theta=True,
        z_score_x=True,
        n_layers=10,
    )

    # Run the SNPE procedure over 1 round
    _ = run_inference(
        simulator=simulator,
        prior=prior,
        build_nn_posterior=build_nn_posterior,
        ground_truth=None,
        meta_parameters=meta_parameters,
        summary_extractor=summary_extractor,
        save_rounds=True,
        device=device,
        num_workers=1,
        max_num_epochs=100000,
    )

    # Get posterior
    npe_jrnmm = get_posterior(
        simulator,
        prior,
        summary_extractor,
        build_nn_posterior,
        meta_parameters,
        round_=0,
        path=PATH_EXPERIMENT
        + "Flows_amortized/JRNMM_nextra_00_naive_True_single_rec_False/Fourier_n_rd_1_n_sr_50000_n_sf_33/",
    )
    return npe_jrnmm


def generate_observations(c, mu, sigma, gain_list):
    """Generate observations for the JRNMM model.

    Args:
        c (float): ground truth parameter 1 (common to all observations)
        mu (float): ground truth parameter 2 (common to all observations)
        sigma (float): ground truth parameter 3 (common to all observations)
        gain_list (list): list of ground truth parameters 4

    Returns:
        x_obs_list (torch.Tensor): list of observations
            of shape (len(gain_list), 33, n_extra+1)
    """
    # IMPORTS (require R-environment)
    from tasks.jrnmm.simulator import get_ground_truth, prior_JRNMM

    # Set prior distribution for the parameters
    prior = prior_JRNMM(
        parameters=[
            ("C", 10.0, 250.0),
            ("mu", 50.0, 500.0),
            ("sigma", 100.0, 5000.0),
            ("gain", -20.0, +20.0),
        ]
    )

    # Compute observations
    theta_true_list = []
    x_obs_list = []
    for g in gain_list:
        theta_true = torch.FloatTensor([c, mu, sigma, g])
        meta_parameters["theta"] = theta_true
        theta_true_list.append(theta_true)

        ground_truth = get_ground_truth(
            meta_parameters,
            input_parameters=["C", "mu", "sigma", "gain"],
            p_gain=prior,
            single_recording=False,
        )
        ground_truth["observation"] = summary_extractor(ground_truth["observation"])
        x_obs = ground_truth["observation"]  # torch.Size([1, 33, n_extra+1])
        x_obs_list.append(x_obs)
    x_obs_list = torch.stack(x_obs_list)[:, 0, :, :]

    return x_obs_list


def global_coverage_tests(
    npe, prior, theta_cal, x_cal, save_path, methods=["sbc", "hpd"]
):
    """Compute global coverage tests for the JRNMM model.

    Args:
        npe (hnpe.posterior.JRNMMFlow_nflows_base): trained posterior estimator
        prior (Distribution): prior distribution
        theta_cal (torch.Tensor): calibration parameters from prior
            of shape (n_cal, dim_theta)
        x_cal (torch.Tensor): calibration observations simulated from theta_cal
            of shape (n_cal, dim_x)
        save_path (str): path to save the results
        methods (list, optional): list of methods to use for global coverage tests.
            Defaults to ["sbc", "hpd"].

    Returns:
        global_rank_stats (dict): dictionary containing the global rank statistics
            for each method (there are n_cal values per dim).
            (dim=4 for "sbc", dim=1 for "hpd")
    """
    # create folder for saving
    save_path_global = save_path / "global_tests"
    if not os.path.exists(save_path_global):
        os.makedirs(save_path_global)

    global_rank_stats = {}
    # 1. SBC
    if "sbc" in methods:
        try:
            # Load sbc ranks if already computed ...
            global_rank_stats["sbc"] = np.array(
                torch.load(save_path_global / f"sbc_ranks_n_cal_{x_cal.shape[0]}.pkl")
            )
        except FileNotFoundError:
            # ... otherwise compute them

            print("     Simulation Based Calibration (SBC):")
            print()
            # Define posterior estimator compatible with sbi implementation of SBC
            from sbi.inference.posteriors.direct_posterior import DirectPosterior

            posterior_sbc = DirectPosterior(
                posterior_estimator=npe, prior=prior, x_shape=x_cal[0][None, :].shape,
            )
            # Run SBC
            sbc = run_sbc(theta_cal, x_cal, posterior=posterior_sbc)
            global_rank_stats["sbc"] = sbc[0]
            # Save SBC ranks
            torch.save(
                global_rank_stats["sbc"],
                save_path_global / f"sbc_ranks_n_cal_{x_cal.shape[0]}.pkl",
            )

    # 2. HPD
    if "hpd" in methods:
        try:
            # Load hpd values if already computed ...
            global_rank_stats["hpd"] = torch.load(
                save_path_global / f"hpd_ranks_n_cal_{x_cal.shape[0]}.pkl"
            )
        except FileNotFoundError:
            # ... otherwise compute them

            print("     HPD:")
            print()

            # Define npe-log-prob function compatible with `hpd_values` function
            from sbi.utils import match_theta_and_x_batch_shapes

            def posterior_log_prob_fn(theta, x):
                theta, x = match_theta_and_x_batch_shapes(theta, x)
                return npe.log_prob(theta, x)

            # Compute HPD values
            joint_hpd_values_cal = hpd_values(
                theta_cal,
                X=x_cal,
                est_log_prob_fn=posterior_log_prob_fn,
                est_sample_fn=npe.sample,
            )

            # Save HPD values for later use in local method
            torch.save(
                joint_hpd_values_cal,
                save_path / f"joint_hpd_values_n_cal_{x_cal.shape[0]}.pkl",
            )

            # Compute HPD ranks
            joint_hpd_ranks = torch.cat(
                (joint_hpd_values_cal, torch.tensor([0.0, 1.0]))
            )
            global_rank_stats["hpd"] = torch.sort(joint_hpd_ranks).values
            # Save HPD ranks
            torch.save(
                global_rank_stats["hpd"],
                save_path_global / f"hpd_ranks_n_cal_{x_cal.shape[0]}.pkl",
            )

    return global_rank_stats


def local_coverage_tests(
    alpha,
    npe,
    theta_cal,
    x_cal,
    n_eval,
    observation_dict,
    t_stats_null_lc2st,
    t_stats_null_lhpd,
    kwargs_lc2st,
    kwargs_lhpd,
    data_path,
    result_path,
    return_predicted_probas=False,
    return_trained_clfs=False,
    methods=["lhpd", "lc2st_nf"],
    test_stat_names=["mse", "div"],
):
    # Create folder for saving
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Initialize results dictionaries
    train_runtime = {}
    trained_clfs_dict = {"lc2st_nf": None, "lhpd": None}
    probas_obs_dict = dict(zip(methods, [{} for _ in methods]))
    results_dict = dict(zip(methods, [{} for _ in methods]))
    result_keys = ["reject", "p_value", "t_stat", "t_stats_null"]

    # Loop over methods
    for m in methods:
        try:
            # Load results if already computed ...
            results_dict[m] = torch.load(
                result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl"
            )
            train_runtime[m] = torch.load(
                result_path / f"runtime_{m}_n_cal_{x_cal.shape[0]}.pkl"
            )
            if return_predicted_probas:
                name = "probas"
                if m == "lhpd":
                    name = "r_alphas"
                probas_obs_dict[m] = torch.load(
                    result_path
                    / f"{m}_{name}_obs_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl"
                )
            if return_trained_clfs:
                trained_clfs_dict[m] = torch.load(
                    result_path / f"trained_clfs_{m}_n_cal_{x_cal.shape[0]}.pkl"
                )

        except FileNotFoundError:
            # ... otherwise compute them
            # Initialize results dictionary
            results_dict[m] = dict(
                zip(
                    result_keys,
                    [
                        dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                        for _ in result_keys
                    ],
                )
            )

            # L-C2ST-NF
            if m == "lc2st_nf":
                print()
                print("     Local C2ST-NF:")
                print()
                try:
                    # Load data if available ...
                    inv_transform_samples_cal = torch.load(
                        data_path / f"inv_transform_samples_n_cal_{x_cal.shape[0]}.pkl"
                    )
                except FileNotFoundError:
                    # ... otherwise generate it
                    print(
                        "Computing the inverse transformation of the flow on samples of the joint ..."
                    )
                    print()
                    inv_transform_samples_cal = npe._transform(
                        theta_cal, context=x_cal
                    )[0].detach()
                    # save data
                    torch.save(
                        inv_transform_samples_cal,
                        data_path / f"inv_transform_samples_n_cal_{x_cal.shape[0]}.pkl",
                    )
                try:
                    # Load data if available ...
                    base_dist_samples_cal = torch.load(
                        data_path / f"base_dist_samples_n_cal_{x_cal.shape[0]}.pkl"
                    )
                    base_dist_samples_eval = torch.load(
                        data_path / f"base_dist_samples_n_eval_{n_eval}.pkl"
                    )
                except FileNotFoundError:
                    # ... otherwise generate it
                    print("Sampling from the base distribution ...")
                    print()
                    base_dist_samples_cal = npe._flow._distribution.sample(
                        x_cal.shape[0]
                    ).detach()
                    base_dist_samples_eval = npe._flow._distribution.sample(
                        n_eval
                    ).detach()
                    # save data
                    torch.save(
                        base_dist_samples_cal,
                        data_path / f"base_dist_samples_n_cal_{x_cal.shape[0]}.pkl",
                    )
                    torch.save(
                        base_dist_samples_eval,
                        data_path / f"base_dist_samples_n_eval_{n_eval}.pkl",
                    )

                print("TRAINING CLASSIFIERS on the joint ...")
                print()
                t0 = time.time()
                _, _, trained_clfs_lc2st = lc2st_scores(
                    P=base_dist_samples_cal,
                    Q=inv_transform_samples_cal,
                    x_P=x_cal[:, :, 0],
                    x_Q=x_cal[:, :, 0],
                    x_eval=None,
                    eval=False,
                    return_clfs=True,
                    **kwargs_lc2st,
                )
                train_runtime["lc2st_nf"] = time.time() - t0
                trained_clfs_dict["lc2st_nf"] = trained_clfs_lc2st

                # Compute results for every observation x_0
                # Loop over observations
                probas_obs = {}
                for key_obs, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    # Evaluate the test
                    lc2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lc2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null_lc2st[key_obs],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        P_eval=base_dist_samples_eval,
                        n_trials_null=len(t_stats_null_lc2st[key_obs]["mse"]),
                        P=None,
                        Q=None,
                        x_P=None,
                        x_Q=None,
                        return_probas=False,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lc2st,
                        **kwargs_lc2st,
                    )
                    # Get predicted probabilities
                    if return_predicted_probas:
                        _, probas_obs[key_obs] = t_stats_lc2st(
                            P=None,
                            Q=None,
                            x_P=None,
                            x_Q=None,
                            x_eval=observation,
                            P_eval=base_dist_samples_eval,
                            metrics=test_stat_names,
                            trained_clfs=trained_clfs_lc2st,
                            return_probas=True,
                            **kwargs_lc2st,
                        )
                    # Add results to results dictionary
                    for i, result_name in enumerate(result_keys):
                        for test_stat_name in test_stat_names:
                            results_dict[m][result_name][test_stat_name].append(
                                lc2st_results_obs[i][test_stat_name]
                            )
                probas_obs_dict[m] = probas_obs

            # L-HPD
            if m == "lhpd":
                print()
                print("     Local HPD:")
                print()

                try:
                    # Load data if available ...
                    joint_hpd_values_cal = torch.load(
                        data_path / f"joint_hpd_values_n_cal_{x_cal.shape[0]}.pkl"
                    )
                except FileNotFoundError:
                    # ... otherwise compute it
                    print("Computing the joint HPD values ...")

                    # Define npe-log-prob function compatible with `hpd_values` function
                    from sbi.utils import match_theta_and_x_batch_shapes

                    def posterior_log_prob_fn(theta, x):
                        theta, x = match_theta_and_x_batch_shapes(theta, x)
                        return npe.log_prob(theta, x)

                    # Compute HPD values
                    joint_hpd_values_cal = hpd_values(
                        theta_cal,
                        X=x_cal,
                        est_log_prob_fn=posterior_log_prob_fn,
                        est_sample_fn=npe.sample,
                    )

                    # Save HPD values
                    torch.save(
                        joint_hpd_values_cal,
                        data_path / f"joint_hpd_values_n_cal_{x_cal.shape[0]}.pkl",
                    )

                print("TRAINING CLASSIFIERS on the joint ...")
                print()
                t0 = time.time()
                _, _, trained_clfs_lhpd = lhpd_scores(
                    Y=theta_cal,
                    X=x_cal[:, :, 0],
                    joint_hpd_values=joint_hpd_values_cal,
                    est_log_prob_fn=None,
                    est_sample_fn=None,
                    x_eval=None,
                    # kwargs for lhpd_scores
                    eval=False,
                    **kwargs_lhpd,
                )
                train_runtime["lhpd"] = time.time() - t0
                trained_clfs_dict["lhpd"] = trained_clfs_lhpd

                # Compute results for every observation x_0
                # Loop over observations
                probas_obs = {}
                for key_obs, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    lhpd_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lhpd,
                        metrics=["mse"],
                        t_stats_null=t_stats_null_lhpd[key_obs],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        Y=theta_cal,
                        X=x_cal[:, :, 0],
                        n_trials_null=len(t_stats_null_lhpd[key_obs]["mse"]),
                        return_r_alphas=False,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lhpd,
                        # kwargs for lhpd_scores
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        **kwargs_lhpd,
                    )
                    # Get predicted probabilities
                    if return_predicted_probas:
                        _, probas_obs[key_obs] = t_stats_lhpd(
                            Y=theta_cal,
                            X=x_cal[:, :, 0],
                            x_eval=observation,
                            trained_clfs=trained_clfs_lhpd,
                            return_r_alphas=True,
                            # kwargs for lhpd_scores
                            est_log_prob_fn=None,
                            est_sample_fn=None,
                            **kwargs_lhpd,
                        )
                    # Add results to results dictionary
                    for i, result_name in enumerate(result_keys):
                        results_dict[m][result_name]["mse"].append(
                            lhpd_results_obs[i]["mse"]
                        )
                probas_obs_dict[m] = probas_obs

            # Save results
            torch.save(
                results_dict[m],
                result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl",
            )
            torch.save(
                train_runtime[m],
                result_path / f"runtime_{m}_n_cal_{x_cal.shape[0]}.pkl",
            )
            if return_predicted_probas:
                name = "probas"
                if m == "lhpd":
                    name = "r_alphas"
                torch.save(
                    probas_obs_dict[m],
                    result_path
                    / f"{m}_{name}_obs_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl",
                )

            if return_trained_clfs:
                torch.save(
                    trained_clfs_dict[m],
                    result_path / f"trained_clfs_{m}_n_cal_{x_cal.shape[0]}.pkl",
                )
    if return_trained_clfs:
        return results_dict, train_runtime, probas_obs_dict, trained_clfs_dict

    elif return_predicted_probas:
        return results_dict, train_runtime, probas_obs_dict

    else:
        return results_dict, train_runtime

