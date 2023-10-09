# Implementation of the optimal Bayes classifier for
# - Gaussian L/QDA
# - Normal Gaussian vs. Student t distribution with varying df

import numpy as np

from scipy.stats import multivariate_normal as mvn, norm
from scipy.stats import t

from valdiags.vanillaC2ST import eval_c2st, compute_metric


class OptimalBayesClassifier:
    """Base classs for an optimal (binary) Bayes classifier to discriminate 
    between data from two distributions associated to the classes c0 and c1:
        
        - X|c0 ~ P 
        - X|c1 ~ Q
    
    The optimal Bayes classifier is given by:
                    
        f(x) = argmax_{p(c0|x), p(c1|x)} \in {0,1} 

    with p(c0|x) = p(x|c0) / (p(x|c0) + p(x|c1)) and p(c1|x) = 1 - p(c0|x).

    Methods:
        fit: fit the classifier to data from P and Q.
            This method is empty as the optimal Bayes classifier is deterministic
            and does not need to be trained.
        predict: predict the class of a given sample.
            returns a numpy array of size (n_samples,).
        predict_proba: predict the probability of the sample to belong to class 0/1.
            returns a numpy array of size (n_samples, 2) with the probabilities.
        score: compute the accuracy of the classifier on a given dataset.
            returns a float.

    """

    def __init__(self) -> None:
        self.dist_c0 = None
        self.dist_c1 = None

    def fit(self, P, Q):
        pass

    def predict(self, x):
        return np.argmax([self.dist_c0.pdf(x), self.dist_c1.pdf(x)], axis=0)

    def predict_proba(self, x):
        d = (self.dist_c0.pdf(x) / (self.dist_c0.pdf(x) + self.dist_c1.pdf(x))).reshape(
            -1, 1
        )
        return np.concatenate([d, 1 - d], axis=1,)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


class AnalyticGaussianLQDA(OptimalBayesClassifier):
    """`OptimalBayesClassifier` for the Gaussian Linear Quadratic Discriminant Analysis (LQDA).
    The two classes are multivariate Gaussians of size `dim`:

        - c0: N(0, I)
        - c1: N(mu, sigma^2*I) with mu and sigma^2 to be specified.
    
    """

    def __init__(self, dim, mu=0.0, sigma=1.0) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.dist_c0 = mvn(mean=np.zeros(dim))
        self.dist_c1 = mvn(mean=np.array([mu] * dim), cov=np.eye(dim) * sigma)

    def predict(self, x):
        if self.mu == 0.0 and self.sigma == 1.0:
            return np.random.binomial(size=x.shape[0], n=1, p=0.5)
        else:
            return super().predict(x)


class AnalyticStudentClassifier(OptimalBayesClassifier):
    """`OptimalBayesClassifier` between Normal and Student t distributions:
        - c0: norm(loc=0, scale=1)
        - c1: t(df=df, loc=mu, scale=sigma) with df, mu and sigma to be specified.
    """

    def __init__(self, mu=0, sigma=1, df=2) -> None:
        super().__init__()
        self.dist_c0 = norm(loc=0, scale=1)
        self.dist_c1 = t(df=df, loc=mu, scale=sigma)


def opt_bayes_scores(
    P,
    Q,
    clf,
    metrics=["accuracy", "mse", "div"],
    single_class_eval=True,
    P_eval=None,
    Q_eval=None,
):
    """Compute the scores of the optimal Bayes classifier on the data from P and Q.
    These scores can be used as test statistics for the C2ST test.

    Args:
        P (np.array): data drawn from P (c0)
            of size (n_samples, dim).
        Q (np.array): data drawn from Q (c1)
            of size (n_samples, dim).
        clf (OptimalBayesClassifier): the initialized classifier to use.
            needs to have a `score` and `predict_proba` method.
        metrics (list, optional): list of metric names (strings) to compute. 
            Defaults to ["accuracy", "div", "mse"].
        single_class_eval (bool, optional): if True, the classifier is evaluated on P only.
            Defaults to True.
        cross_val (bool, optional): never used. Defaults to False.

    Returns:
        dict: dictionary of scores for each metric.
    """
    # evaluate the classifier on the data
    accuracy, proba = eval_c2st(P=P, Q=Q, clf=clf, single_class_eval=single_class_eval)

    # compute the scores / metrics
    scores = dict(zip(metrics, [None] * len(metrics)))
    for m in metrics:
        if m == "accuracy":
            scores["accuracy"] = accuracy  # already computed
        else:
            scores[m] = compute_metric(
                proba, metrics=[m], single_class_eval=single_class_eval
            )[m]

    return scores


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sklearn.calibration import calibration_curve

    N_SAMPLES = 10_000
    DIM = 5

    # shifts = np.array([0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10])
    # shifts = np.sort(np.concatenate([-1 * shifts, shifts[1:]]))

    # uncomment this to do the scale-shift experiment
    # shifts = np.array([0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    shifts = np.array(
        [
            0.01,
            # 0.1,
            # 0.2,
            # 0.3,
            # 0.4,
            0.5,
            # 0.6,
            # 0.7,
            # 0.8,
            0.9,
            1,
            1.1,
            # 1.2,
            # 1.3,
            # 1.4,
            1.5,
            # 2,
            3,
        ]
    )

    test_stats_runs = {
        r"$\hat{t}_{Acc}$": [],
        r"$\hat{t}_{Acc0}$": [],
        r"$\hat{t}_{Reg}$": [],
        r"$\hat{t}_{Reg0}$": [],
        r"$\hat{t}_{Max}$": [],
        r"$\hat{t}_{Max0}$": [],
    }

    for r in range(1):
        # ref norm samples
        ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)

        # shifted_samples = [
        #     mvn(mean=np.array([s] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
        #     for s in shifts
        # ]

        # uncomment this to do the scale-shift experiment
        shifted_samples = [
            mvn(mean=np.zeros(DIM), cov=np.eye(DIM) * s).rvs(N_SAMPLES) for s in shifts
        ]

        # # uncomment this to do the student df-shift experiment
        # # ref student samples
        # ref_samples = norm(loc=0, scale=1).rvs(N_SAMPLES)
        # shifts = np.arange(0.01, 21)
        # shifted_samples = [t(df=s, loc=0, scale=1).rvs(N_SAMPLES) for s in shifts]

        test_stats = {
            r"$\hat{t}_{Acc}$": [],
            r"$\hat{t}_{Acc0}$": [],
            r"$\hat{t}_{Reg}$": [],
            r"$\hat{t}_{Reg0}$": [],
            r"$\hat{t}_{Max}$": [],
            r"$\hat{t}_{Max0}$": [],
        }

        cal_curves = {"oracle": [], "single_class": []}

        for s, s_samples in zip(shifts, shifted_samples):
            # uncomment this to do the mean-shift experiment
            # clf = AnalyticGaussianLQDA(dim=DIM, mu=s)
            # uncomment this to do the scale-shift experiment
            clf = AnalyticGaussianLQDA(dim=DIM, sigma=s)

            # # uncomment this to do the student mean-shift experiment
            # clf = AnalyticStudentClassifier(df=s)

            for b in [True, False]:

                # scores
                scores = opt_bayes_scores(
                    P=ref_samples, Q=s_samples, clf=clf, single_class_eval=b
                )
                if b:
                    test_stats[r"$\hat{t}_{Acc0}$"].append(scores["accuracy"])
                    test_stats[r"$\hat{t}_{Reg0}$"].append(scores["mse"] + 0.5)
                    # test_stats[r"$\hat{t}_{Max0}$"].append(scores["div"])
                else:
                    test_stats[r"$\hat{t}_{Acc}$"].append(scores["accuracy"])
                    test_stats[r"$\hat{t}_{Reg}$"].append(scores["mse"] + 0.5)
                    # test_stats[r"$\hat{t}_{Max}$"].append(scores["div"])

                # calibration
                if b:
                    features = ref_samples
                    y_true = np.zeros(N_SAMPLES)
                    label = "single_class"
                else:
                    features = np.concatenate([ref_samples, s_samples])
                    y_true = np.concatenate([np.zeros(N_SAMPLES), np.ones(N_SAMPLES)])
                    label = "oracle"
                y_pred = clf.predict_proba(features)[:, 1]
                prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
                cal_curves[label].append((prob_true, prob_pred))
                print(prob_true, prob_pred)

        for k in test_stats.keys():
            test_stats_runs[k].append(test_stats[k])

    test_stats_mean = {k: np.mean(v, axis=0) for k, v in test_stats_runs.items()}
    test_stats_std = {k: np.std(v, axis=0) for k, v in test_stats_runs.items()}
    colors = ["orange", "orange", "blue", "blue"]  # , "red", "red"]

    # # Mean-shift experiment plot
    # for name, color in zip(test_stats.keys(), colors):
    #     linestyle = "-"
    #     if "0" not in name:
    #         linestyle = "--"
    #     # plt.errorbar(shifts, test_stats_mean, color=color, label=name, linestyle=linestyle)
    #     plt.plot(
    #         shifts, test_stats_mean[name], color=color, label=name, linestyle=linestyle,
    #     )
    #     plt.fill_between(
    #         x=shifts,
    #         y1=test_stats_mean[name] - test_stats_std[name],
    #         y2=test_stats_mean[name] + test_stats_std[name],
    #         alpha=0.2,
    #         color=color,
    #     )
    # plt.plot(shifts, [0.5] * len(shifts), color="grey", linestyle="--", label="H_0")
    # plt.plot(
    #     [0] * len(np.arange(0.4, 1.1, 0.1)),
    #     np.arange(0.4, 1.1, 0.1),
    #     color="grey",
    #     linestyle="--",
    # )
    # plt.xlabel("m (mean shift)")
    # plt.ylabel(r"$\hat{t}$ (test statistic)")
    # plt.legend(loc="upper right")
    # plt.title(f"Optimal Bayes Classifier for H_0: N(0, I) = N(m, I), dim={DIM}")
    # plt.savefig(f"lqda_mean_shift_dim_{DIM}_n_{N_SAMPLES}.pdf")
    # plt.show()

    # Scale-shift experiment plot
    for name, color in zip(test_stats.keys(), colors):
        linestyle = "-"
        if "0" not in name:
            linestyle = "--"
        # plt.errorbar(shifts, test_stats_mean, color=color, label=name, linestyle=linestyle)
        plt.plot(
            shifts, test_stats_mean[name], color=color, label=name, linestyle=linestyle,
        )
        plt.fill_between(
            x=shifts,
            y1=test_stats_mean[name] - test_stats_std[name],
            y2=test_stats_mean[name] + test_stats_std[name],
            alpha=0.2,
            color=color,
        )
    plt.plot(shifts, [0.5] * len(shifts), color="grey", linestyle="--", label="H_0")
    plt.plot(
        [1] * len(np.arange(0.4, 1.1, 0.1)),
        np.arange(0.4, 1.1, 0.1),
        color="grey",
        linestyle="--",
    )
    plt.xlabel("s (scale shift)")
    plt.ylabel(r"$\hat{t}$ (test statistic)")
    plt.legend(loc="upper right")
    plt.title(f"Optimal Bayes Classifier for H_0: N(0, I) = N(0, s), dim={DIM}")
    plt.savefig(f"lqda_scale_shift_dim_{DIM}_n_{N_SAMPLES}.pdf")
    plt.show()

    # calibration
    for label in cal_curves.keys():
        for i, s in enumerate(shifts):
            plt.plot(
                cal_curves[label][i][0],
                cal_curves[label][i][1],
                label=f"scale shift = {s}",
                linestyle="-",
            )
        plt.plot(
            np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black"
        )
        plt.legend()
        plt.xlabel("True probability")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title("Calibration curves, " + label)
        plt.show()

    # # DF-shift experiment plot
    # for name, color in zip(test_stats.keys(), colors):
    #     linestyle = "-"
    #     if "0" not in name:
    #         linestyle = "--"
    #     # plt.errorbar(shifts, test_stats_mean, color=color, label=name, linestyle=linestyle)
    #     plt.plot(
    #         shifts, test_stats_mean[name], color=color, label=name, linestyle=linestyle,
    #     )
    #     plt.fill_between(
    #         x=shifts,
    #         y1=test_stats_mean[name] - test_stats_std[name],
    #         y2=test_stats_mean[name] + test_stats_std[name],
    #         alpha=0.2,
    #         color=color,
    #     )
    # plt.plot(shifts, [0.5] * len(shifts), color="grey", linestyle="--", label="H_0")
    # plt.xlabel("df (degrees of freedom)")
    # plt.ylabel(r"$\hat{t}$ (test statistic)")
    # plt.legend(loc="upper right")
    # plt.title(f"Optimal Bayes Classifier for H_0: N(0, I) = Student-t(df)")
    # plt.savefig(f"student_df_shift_n_{N_SAMPLES}.pdf")
    # plt.show()
