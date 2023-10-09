import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from matplotlib.gridspec import GridSpec
from pathlib import Path
from valdiags.vanillaC2ST import sbibm_clf_kwargs
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
    calibration_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, brier_score_loss

from tqdm import tqdm
from tueplots import fonts, axes


# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Plotting settings
plt.rcParams.update(fonts.neurips2022())
plt.rcParams.update(axes.color(base="black"))
plt.rcParams["legend.fontsize"] = 20.0
plt.rcParams["xtick.labelsize"] = 23.0
plt.rcParams["ytick.labelsize"] = 23.0
plt.rcParams["axes.labelsize"] = 23.0
plt.rcParams["font.size"] = 23.0
plt.rcParams["axes.titlesize"] = 27.0


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="lc2st_nf")
parser.add_argument(
    "--task", type=str, default="jrnmm", choices=["jrnmm", "gaussian_mixture"]
)
parser.add_argument(
    "--experiment", "-e", type=str, required=True, choices=["clf_eval", "clf_choice"]
)
parser.add_argument("--param", "-p", type=str, default="l2_reg", choices=["l2_reg", "hf_nl"])
parser.add_argument("--cross_val", "-cv", action="store_true")
parser.add_argument("--calibration", "-cal", action="store_true")
parser.add_argument(
    "--scoring",
    "-s",
    type=str,
    default="accuracy",
    choices=[
        "accuracy",
        "neg_log_loss",
        "neg_brier_score",
    ],
)
args = parser.parse_args()

# Method name
experiment_name = r"$\ell$-C2ST"
if args.method == "lc2st_nf":
    experiment_name += "-NF"

# Specific to task
if args.task == "jrnmm":
    experiment_name += " on JRNMM"

    # Path to save results
    PATH_EXPERIMENT = Path("saved_experiments/lc2st_2023/exp_3")

    # Load posterior estimator
    npe = torch.load(PATH_EXPERIMENT / "posterior_estimator_jrnmm.pkl")

    # Load data
    # Calibration data (used to train / calibrate the classifier)
    joint_dataset_cal = torch.load(PATH_EXPERIMENT / "joint_data_jrnmm_cal.pkl")
    theta_cal, x_cal = joint_dataset_cal["theta"], joint_dataset_cal["x"]

    # Test data (used to evaluate the classifier: accuracy or calibration)
    joint_dataset_test = torch.load(PATH_EXPERIMENT / "joint_data_jrnmm_test.pkl")
    theta_test, x_test = joint_dataset_test["theta"], joint_dataset_test["x"]

    # Base dist samples
    base_dist_samples_cal = torch.load(
        PATH_EXPERIMENT / "base_dist_samples_n_cal_10000.pkl"
    )
    base_dist_samples_test = npe._flow._distribution.sample(x_test.shape[0]).detach()

    # Inverse transform on joint samples
    inv_transform_samples_cal = torch.load(
        PATH_EXPERIMENT / "inv_transform_samples_n_cal_10000.pkl"
    )
    inv_transform_samples_test = npe._transform(theta_test, context=x_test)[0].detach()

    # Joint Estimator samples
    for name, x_samples, base_dist_samples in zip(
        ["cal", "test"],
        [x_cal, x_test],
        [base_dist_samples_cal, base_dist_samples_test],
    ):
        if not os.path.exists(PATH_EXPERIMENT / f"npe_samples_{name}.pkl"):
            npe_samples = []
            for x, z in tqdm(
                zip(x_samples, base_dist_samples), desc=f"Sampling from reference ({name})"
            ):
                x = x[None, :]
                z = z[None, :]
                x_emb = npe._flow._embedding_net(x)
                z_transformed = npe._flow._transform.inverse(z, context=x_emb)[0].detach()
                npe_samples.append(z_transformed)
            npe_samples = torch.stack(npe_samples)[:, 0, :]
            torch.save(npe_samples, PATH_EXPERIMENT / f"npe_samples_{name}.pkl")
    npe_samples_cal = torch.load(PATH_EXPERIMENT / "npe_samples_cal.pkl")
    npe_samples_test = torch.load(PATH_EXPERIMENT / "npe_samples_test.pkl")

    x_cal = x_cal[:, :, 0]
    x_test = x_test[:, :, 0]

elif args.task == "gaussian_mixture":
    experiment_name += " on Gaussian Mixture"
    n_train = 1000

    # Path to save results
    PATH_EXPERIMENT = Path("saved_experiments/lc2st_2023/exp_2/gaussian_mixture")

    # Load posterior estimator
    npe = torch.load(PATH_EXPERIMENT / f"npe_{n_train}/posterior_estimator.pkl")

    # Load data
    # Calibration data (used to train / calibrate the classifier)
    joint_dataset_cal = torch.load(PATH_EXPERIMENT / "joint_samples_n_cal_10000.pkl")
    theta_cal, x_cal = joint_dataset_cal["theta"][:9000], joint_dataset_cal["x"][:9000]

    # Test data (used to evaluate the classifier: accuracy or calibration)
    theta_test, x_test = theta_cal[-1000:], x_cal[-1000:]

    # Base dist samples
    base_dist_samples_cal = torch.load(
        PATH_EXPERIMENT / "base_dist_samples_n_cal_10000.pkl"
    )[:9000]
    base_dist_samples_test = base_dist_samples_cal[-1000:]

    # Inverse transform on joint samples
    inv_transform_samples_cal = torch.load(
        PATH_EXPERIMENT / f"npe_{n_train}/inv_transform_samples_theta_cal_10000.pkl"
    )[:9000]
    inv_transform_samples_test = inv_transform_samples_cal[-1000:]

    # Joint Estimator samples
    npe_samples_cal = torch.load(
        PATH_EXPERIMENT / f"npe_{n_train}/npe_samples_x_cal_10000.pkl"
    )[:9000]
    npe_samples_test = npe_samples_cal[-1000:]

else:
    raise NotImplementedError(f"Unknown task {args.task}")

# compute classifier features
if args.method == "lc2st_nf":
    P = np.concatenate([base_dist_samples_cal, x_cal], axis=1)
    P_eval = np.concatenate([base_dist_samples_test, x_test], axis=1)
    Q = np.concatenate([inv_transform_samples_cal, x_cal], axis=1)
    Q_eval = np.concatenate([inv_transform_samples_test, x_test], axis=1)
else:
    P = np.concatenate([theta_cal, x_cal], axis=1)
    P_eval = np.concatenate([theta_test, x_test], axis=1)
    Q = np.concatenate([npe_samples_cal, x_cal], axis=1)
    Q_eval = np.concatenate([npe_samples_test, x_test], axis=1)

features_cal = np.concatenate([P, Q], axis=0)
features_test = np.concatenate([P_eval, Q_eval], axis=0)
labels_cal = np.concatenate([np.zeros(P.shape[0]), np.ones(Q.shape[0])], axis=0)
labels_test = np.concatenate(
    [np.zeros(P_eval.shape[0]), np.ones(Q_eval.shape[0])], axis=0
)

# Classifiers
# Evaluate if used classifier is calibrated
if args.experiment == "clf_eval":
    sbibm_kwargs = sbibm_clf_kwargs(ndim=theta_cal.shape[-1])
    sbibm_mlp = MLPClassifier(**sbibm_kwargs)
    sbibm_mlp_cal = CalibratedClassifierCV(sbibm_mlp, method="sigmoid")

    clf_list = [
        (sbibm_mlp, "sbibm_mlp"),
        (sbibm_mlp_cal, "sbibm_mlp + cal"),
    ]

    param_name = "eval"

    def colors(i):
        return ["blue", "orange"][i]


# Choose better classifier
elif args.experiment == "clf_choice":
    # see if more regularization help for lc2st
    if args.param == "l2_reg":
        param_label = "l2 regularization"
        # param_list = [0.0, 1.0, 10, 50, 100, 500, 1000]
        param_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]

        clf_list = [
            (MLPClassifier(alpha=l2_reg, max_iter=25000), f"MLP, l2_reg={l2_reg}")
            for l2_reg in param_list
        ]

        x_ticks = param_list

    # see if more hidden features help for lc2st_nf
    elif args.param == "hf_nl":
        param_label = "(hidden_features, n_layers)"
        hidden_features_list = [10, 50, 100, 500]
        n_layers_list = [1, 2, 3]
        param_list = [(hf, nl) for hf in hidden_features_list for nl in n_layers_list]

        clf_list = [
            (
                MLPClassifier(hidden_layer_sizes=tuple([hf] * nl), max_iter=25000),
                f"MLP, hf={hf}, nl={nl}",
            )
            for hf, nl in param_list
        ]
        param_list = [i for i in range(len(param_list))]
        x_ticks = [
            f"({hf}, {nl})" for hf in hidden_features_list for nl in n_layers_list
        ]

    colors = plt.get_cmap("Dark2")

    param_name = args.param


if args.task != "jrnmm":
    param_name += f"_ntrain_{n_train}"


# Cross validation
if args.cross_val:
    scores_cv = []
    for i, (clf, name) in tqdm(enumerate(clf_list), desc="CV", total=len(clf_list)):
        scores_cv.append(
            cross_val_score(
                clf, features_cal, labels_cal, scoring=args.scoring
            ).mean()
        )

    # plot cross validation scores
    fig = plt.figure(figsize=(12, 8))

    plt.plot(param_list, scores_cv, "-o")
    if "l2_reg" in param_name:
        plt.xscale("log")
    else:
        plt.xticks(param_list, x_ticks)
    plt.xlabel(param_label)
    plt.ylabel("CV " + args.scoring)
    if args.scoring == "accuracy":
        plt.ylim(0.45, 1)
    elif args.scoring == "neg_log_loss":
        plt.ylim(min(scores_cv)-0.1, 0)
    plt.title("MLP CV scores" + f"\n ({experiment_name})")
    plt.savefig(
        PATH_EXPERIMENT
        / "calibration"
        / f"cross_validation_{args.scoring}_{args.method}_{param_name}.pdf"
    )
    plt.show()

# Calibration plots
if args.calibration:
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in tqdm(
        enumerate(clf_list), desc="Calibration", total=len(clf_list)
    ):
        clf.fit(features_cal, labels_cal)
        if args.scoring == "accuracy":
            score_test = clf.score(features_test, labels_test)
            score_name = "accuracy"
        elif args.scoring == "neg_log_loss":
            score_test = log_loss(labels_test, clf.predict_proba(features_test))
            score_name = "log_loss"
        elif args.scoring == "neg_brier_score":
            score_test = brier_score_loss(
                labels_test, clf.predict_proba(features_test)[:, 1]
            )
            score_name = "brier_score_loss"
        else:
            raise NotImplementedError

        # expected calibration error
        prob_true, prob_pred = calibration_curve(
            labels_test, clf.predict_proba(features_test)[:, 1], n_bins=10
        )
        ece_score = np.mean(np.abs(prob_true - prob_pred))
        print("ECE", ece_score)

        display = CalibrationDisplay.from_estimator(
            clf,
            features_test,
            labels_test,
            n_bins=10,
            name=name + f" ({score_name}={score_test:.2f})",
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    # ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots" + f"\n ({experiment_name})")

    plt.tight_layout()
    plt.savefig(
        PATH_EXPERIMENT
        / "calibration"
        / f"calibration_plots_{args.scoring}_{args.method}_{param_name}.pdf"
    )
    plt.show()
