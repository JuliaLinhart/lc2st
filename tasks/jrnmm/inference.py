from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter

import numpy as np
import torch

from nde.train import train_lampe_npe

from sbi import inference as sbi_inference
from sbi.utils import get_log_root
from sbi.utils.sbiutils import standardizing_net


def run_inference_lampe(
    simulator,
    prior,
    estimator,
    meta_parameters,
    ground_truth=None,
    summary_extractor=None,
    save_rounds=False,
    seed=42,
    max_num_epochs=10_000,
    training_batch_size=100,
    dataset_train=None,
    optimizer=torch.optim.AdamW,
    lr=5e-4,  # default learning rate from sbi training function
    epochs_until_convergence=20,
):
    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / meta_parameters["label"]
        folderpath.mkdir(exist_ok=True, parents=True)
        if ground_truth is not None:
            path = folderpath / "ground_truth.pkl"
            torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # loop over rounds
    posteriors = []
    proposal = prior
    if ground_truth is not None:
        ground_truth_obs = ground_truth["x"]

    for round_ in range(meta_parameters["n_rd"]):
        # simulate / load the necessary data
        if dataset_train is not None:
            # load the necessary data
            theta, x = dataset_train["theta"], dataset_train["x"]
        else:
            # simulate the necessary data
            theta = proposal.sample((meta_parameters["n_sr"],))
            x = simulator(theta)

            # extract summary features
            if summary_extractor is not None:
                x = summary_extractor(x)

        # define inference object
        inference = estimator(theta, x)

        # train the neural posterior with the loaded data
        _, epochs = train_lampe_npe(
            inference,
            theta,
            x,
            num_epochs=max_num_epochs,
            batch_size=training_batch_size,
            lr=lr,
            clip=5.0,  # default clip from sbi training function
            optimizer=optimizer,
            validation=True,
            epochs_until_converge=epochs_until_convergence,
        )
        print(f"inference done in {epochs} epochs")

        inference.eval()
        posteriors.append(inference)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            torch.save(inference, path)
            print("saved")

        # define proposal for next round
        if meta_parameters["n_rd"] > 1:
            assert ground_truth is not None
            # set the proposal prior for the next round
            proposal = inference.flow(ground_truth_obs)

    return posteriors


def summary_plcr(prefix):
    logdir = Path(
        get_log_root(),
        prefix,
        datetime.now().isoformat().replace(":", "_"),
    )
    return SummaryWriter(logdir)


def run_inference_sbi(
    simulator,
    prior,
    build_nn_posterior,
    ground_truth,
    meta_parameters,
    summary_extractor=None,
    save_rounds=False,
    seed=42,
    device="cpu",
    num_workers=1,
    max_num_epochs=None,
    stop_after_epochs=20,
    training_batch_size=100,
    build_aggregate_before=None,
):
    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a SBI-wrapper on the simulator object for compatibility
    simulator, prior = sbi_inference.prepare_for_sbi(simulator, prior)

    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / meta_parameters["label"]
        print(folderpath)
        folderpath.mkdir(exist_ok=True, parents=True)
        if ground_truth is not None:
            path = folderpath / "ground_truth.pkl"
            torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # setup the inference procedure
    inference = sbi_inference.SNPE(
        prior=prior,
        density_estimator=build_nn_posterior,
        show_progress_bars=True,
        device=device,
        summary_writer=summary_plcr(meta_parameters["label"]),
    )

    # loop over rounds
    posteriors = []
    proposal = prior
    if ground_truth is not None:
        ground_truth_obs = ground_truth["observation"]

    for round_ in range(meta_parameters["n_rd"]):
        # simulate the necessary data
        theta, x = sbi_inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=meta_parameters["n_sr"],
            num_workers=num_workers,
        )

        if "cuda" in device:
            torch.cuda.empty_cache()

        # extract summary features
        if summary_extractor is not None:
            x = summary_extractor(x)

        if (x[0].shape[0] == 3) and meta_parameters["norm_before"]:
            print("norm_before")
            x0_n = torch.cat([x[:, 0].reshape(-1, 1), x[:, 2].reshape(-1, 1)], dim=1)
            stand = standardizing_net(x0_n)
            x0_n = stand(x0_n)

            x[:, 0] = x0_n[:, 0]
            x[:, 2] = x0_n[:, 1]

            path = folderpath / f"stand_net_round_{round_:02}.pkl"
            torch.save(stand.state_dict(), path)

        ## ------- added --------- ##
        # standardize data wrt x and aggregate extra observations
        if build_aggregate_before is not None:
            aggregate_before = build_aggregate_before(x_ref=x)  # standardize data wrt x
            x = aggregate_before(x)
            if ground_truth is not None:
                ground_truth_obs = aggregate_before(ground_truth["observation"])
        else:
            aggregate_before = None
        ## ----------------------- ##
        # train the neural posterior with the loaded data
        nn_posterior = inference.append_simulations(theta, x).train(
            num_atoms=10,
            training_batch_size=training_batch_size,
            use_combined_loss=True,
            discard_prior_samples=True,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True,
        )
        nn_posterior.zero_grad()
        posterior = inference.build_posterior(nn_posterior)
        posteriors.append(posterior)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            posterior.posterior_estimator.save_state(path)
            ## --------------- added -------------- ##
            # save aggregate net parameters: mean and std based on training simulations
            if aggregate_before is not None:
                path = folderpath / f"norm_agg_before_net_round_{round_:02}.pkl"
                torch.save(aggregate_before.state_dict(), path)
            ## ------------------------------------ ##

        if meta_parameters["n_rd"] > 1:
            assert ground_truth is not None
            # set the proposal prior for the next round
            proposal = posterior.set_default_x(ground_truth_obs)

    return posteriors


if __name__ == "__main__":
    from .simulator import prior_JRNMM, simulator_JRNMM
    from hnpe.summary import summary_JRNMM
    from functools import partial
    from hnpe.misc import make_label
    from .posterior import NPE_JRNMM_lampe_base

    PATH_EXPERIMENT = "saved_experiments/JR-NMM/fixed_gain_3d/"
    N_EXTRA = 0

    meta_parameters = {}
    # Data features
    meta_parameters["t_recording"] = 8
    meta_parameters["n_extra"] = N_EXTRA
    # Summary Features
    meta_parameters["summary"] = "Fourier"
    meta_parameters["n_sf"] = 33
    # Training Features
    meta_parameters["n_rd"] = 1  # amortized flow
    meta_parameters["n_sr"] = 50_000  # simulations per round

    # example cases we are considering here
    meta_parameters["case"] = (
        PATH_EXPERIMENT + "test_lampe/JRNMM_nextra_{:02}_"
        "naive_{}_"
        "single_rec_{}".format(N_EXTRA, True, False)
    )

    # label for saving directory of experiments
    meta_parameters["label"] = make_label(meta_parameters)
    folderpath = Path.cwd() / meta_parameters["label"]

    # Prior
    prior = prior_JRNMM(
        parameters=[("C", 10.0, 250.0), ("mu", 50.0, 500.0), ("sigma", 100.0, 5000.0)]
    )

    # Simulator
    simulator = partial(
        simulator_JRNMM,
        input_parameters=["C", "mu", "sigma"],
        t_recording=meta_parameters["t_recording"],
        n_extra=N_EXTRA,
        p_gain=prior,
    )

    summary_extractor = summary_JRNMM(
        n_extra=N_EXTRA,
        d_embedding=meta_parameters["n_sf"],
        n_time_samples=int(128 * meta_parameters["t_recording"]),
        type_embedding=meta_parameters["summary"],
    )

    summary_extractor.embedding.net.logscale = True  # log-PSD

    # train data
    dataset_train = torch.load(PATH_EXPERIMENT + "datasets_train.pkl")

    # ground truth for rounds > 1
    gt_theta = prior.sample((1,))
    gt_x = summary_extractor(simulator(gt_theta))[0]
    print(gt_x.shape)
    ground_truth = {"theta": gt_theta, "x": gt_x}

    _ = run_inference_lampe(
        simulator,
        prior,
        dataset_train=dataset_train,
        estimator=partial(NPE_JRNMM_lampe_base, randperm=False),
        meta_parameters=meta_parameters,
        ground_truth=ground_truth,
        summary_extractor=summary_extractor,
        save_rounds=True,
        training_batch_size=100,
        # optimizer=torch.optim.Adam,
        # lr=0.1,
    )
