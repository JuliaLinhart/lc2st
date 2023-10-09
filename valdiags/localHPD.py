import numpy as np
import torch

from tqdm import tqdm

import sklearn
from sklearn.neural_network import MLPClassifier

from scipy.stats import uniform

# define default classifier
DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def hpd_values(
    Y,
    est_log_prob_fn,
    est_sample_fn,
    X=None,
    n_samples=1000,
    verbose=True,
):
    """Highest Predictive Density values for a (conditional) estimator q:

    We check if a true sample x_0 is in the highest predictive density region of the est-estimator q
    at level 1-alpha, which is equivalent to the proportion of samples x ~ q
    having a higher estimated density than x_0: E_x[I_{q(x)>q(x_0)}].

    By computing this for a large number of x_0, covering the space of the true distribution p(x),
    we get the expected coverage (or levels) over all possible covergage levels in [0,1].

    If q = p, these should be uniformly distributed over [0,1].

    Following the implementation from
    https://github.com/francois-rozet/lampe/blob/master/lampe/diagnostics.py
    adapted to non-lampe distributions.
    """
    values = []

    with torch.no_grad():
        if X is None:
            for y_0 in tqdm(Y, desc="Computing HPD values", disable=(not verbose)):
                samples = est_sample_fn(n_samples)
                mask = est_log_prob_fn(y_0[None, :]) < est_log_prob_fn(samples)
                rank = mask.sum() / mask.numel()
                values.append(rank)
        else:
            for y_0, x_0 in tqdm(
                zip(Y, X), desc="Computing joint HPD values", disable=(not verbose)
            ):
                y_0, x_0 = y_0[None, :], x_0[None, :]
                samples = est_sample_fn(n_samples, x_0)
                mask = est_log_prob_fn(y_0, x_0) < est_log_prob_fn(samples, x_0)
                rank = mask.sum() / mask.numel()
                values.append(rank)

    values = torch.stack(values).cpu()

    # values = torch.cat((values, torch.tensor([0.0, 1.0])))

    # ranks = torch.sort(ranks).values
    # alphas = torch.linspace(0.0, 1.0, len(ranks))
    return values


def train_lhpd(X, joint_hpd_values, n_alphas, clf, verbose=True):
    # define range of alpha levels such that the highest value will yield
    # data from both classes: hpd_values <= max(alpha) not always 1

    max_v = max(joint_hpd_values)
    alphas = np.linspace(0, max_v - 0.001, n_alphas)

    clfs = {}
    for alpha in tqdm(alphas, desc="Training L-HPD", disable=(not verbose)):
        # compute the binary regression targets
        W_a = (joint_hpd_values <= alpha) * 1
        # define classifier
        clf = sklearn.base.clone(clf)
        # train regression model
        clf.fit(X=X, y=W_a.ravel())
        clfs[alpha] = clf

    return clfs


def eval_lhpd(x_eval, clfs):
    alphas = np.array(list(clfs.keys()))
    r_alphas = {}
    for alpha in alphas:
        # evaluate in x_eval: P(HPD <= alpha | x_eval)
        r_alphas[alpha] = clfs[alpha].predict_proba(x_eval)[:, 1][0]
    return r_alphas


def lhpd_scores(
    Y,
    X,
    n_alphas,
    x_eval,
    est_log_prob_fn,
    est_sample_fn,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    n_ensemble=1,
    joint_hpd_values=None,
    trained_clfs=None,
    return_clfs=False,
    eval=True,
    verbose=True,
):
    """Estimate the 1D local HPD-distribution of a conditional estimator q(y|x) of p(y|x):

    Learn the point-wise c.d.f on the joint distribution of X and Y:
        r_{\alpha}(X) = P(HPD(Y,X) <= alpha | X) = E[1_{HPD(X,Y) <= alpha} | X]

    Args:
        Y (np.array): data drawn from p(y)
            of size (n_samples, dim).
        X (np.array): data drawn from p(x|y) such that [Y,X]~p(y,x)
            of size (n_samples, nb_features).
        alphas (np.array): alpha values to evaluate the c.d.f.
        x_eval (np.array): points at which to evaluate the c.d.f.
            of size (1, nb_features).
        est_log_prob_fn (function): log-probability function of the estimator.
        est_sample_fn (function): sample function of the imator.
        clf_class (sklearn.base.BaseEstimator): classifier class.
        clf_kwargs (dict): kwargs for the classifier.
        n_ensemble (int): number of ensemble models to train.
        trained_clfs (dict): pre-trained classifiers.
        return_r_alphas (bool): whether to return the r_alphas.

    Returns:
        (tuple): tuple containing:
            - scores (float): L2-distance between the estimated and the uniform c.d.f (alphas).
            - r_alphas (dict): estimated c.d.f. values.
    """
    # compute joint HPD values
    if joint_hpd_values is None and trained_clfs is None:
        joint_hpd_values = hpd_values(
            Y=Y,
            X=X,
            est_log_prob_fn=est_log_prob_fn,
            est_sample_fn=est_sample_fn,
            verbose=verbose,
        )

    # estimate r_alphas
    clfs_list = []
    for n in range(n_ensemble):
        if trained_clfs is not None:
            clfs_n = trained_clfs[n]
        else:
            # initialize classifier
            classifier = clf_class(random_state=n, **clf_kwargs)
            # train classifier
            clfs_n = train_lhpd(
                X, joint_hpd_values, n_alphas, clf=classifier, verbose=verbose
            )
        clfs_list.append(clfs_n)

    if not eval:
        return None, None, clfs_list

    alphas = np.array(list(clfs_list[0].keys()))
    ens_r_alphas = {alpha: [] for alpha in alphas}
    for clfs_n in clfs_list:
        # eval classifier
        r_alphas = eval_lhpd(x_eval, clfs_n)
        for alpha in alphas:
            ens_r_alphas[alpha].append(r_alphas[alpha])

    # compute proba of ensemble model
    r_alphas = {alpha: np.mean(ens_r_alphas[alpha]) for alpha in alphas}

    # compute L2-distance
    scores = ((np.array(list(r_alphas.values())) - alphas) ** 2).mean()

    if return_clfs:
        return scores, r_alphas, clfs_list
    else:
        return scores, r_alphas


def t_stats_lhpd(
    Y,
    X,
    n_alphas,
    x_eval,
    scores_fn=lhpd_scores,
    metrics=["mse"],  # only needed for eval_htest
    trained_clfs=None,
    null_hypothesis=False,
    n_trials_null=100,
    trained_clfs_null=None,
    return_r_alphas=False,
    return_clfs_null=False,
    verbose=True,
    **kwargs,  # kwargs for scores_fn
):
    if not null_hypothesis:
        t_stat_data, r_alphas_data = scores_fn(
            Y, X, n_alphas, x_eval, trained_clfs=trained_clfs, verbose=True, **kwargs
        )
        if return_r_alphas:
            return {"mse": t_stat_data}, r_alphas_data
        else:
            return {"mse": t_stat_data}

    else:
        r_alphas_null = {i: [] for i in range(n_alphas)}
        clfs_null = []
        t_stats_null = {"mse": []}

        if trained_clfs_null is None:
            trained_clfs_null = [None for _ in range(n_trials_null)]

        # loop over trials under the null hypothesis
        for t in tqdm(
            range(n_trials_null),
            desc="Training / Computing T under (H0)",
            disable=(not verbose),
        ):
            scores_t, r_alphas_t, clfs_t = scores_fn(
                Y,
                X,
                n_alphas,
                x_eval,
                joint_hpd_values=uniform().rvs((Y.shape[0]), random_state=t),
                trained_clfs=trained_clfs_null[t],
                return_clfs=True,
                verbose=False,
                **kwargs,
            )
            clfs_null.append(clfs_t)
            t_stats_null["mse"].append(scores_t)

            if r_alphas_t is not None:
                for i, alpha in enumerate(clfs_t[0].keys()):
                    r_alphas_null[i].append(r_alphas_t[alpha])

        if return_clfs_null:
            return t_stats_null, r_alphas_null, clfs_null
        elif return_r_alphas:
            return t_stats_null, r_alphas_null
        else:
            return t_stats_null
