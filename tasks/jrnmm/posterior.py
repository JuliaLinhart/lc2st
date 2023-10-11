# Code adapted from https://github.com/plcrodrigues/HNPE

import torch
import torch.nn as nn

from lampe.inference import NPE
from pathlib import Path
from pyknos.nflows.distributions import base
from sbi.utils.get_nn_models import build_maf
from zuko.flows import MAF


class AggregateInstances(torch.nn.Module):
    def __init__(self, mean=True):

        super().__init__()
        self.mean = mean

    def forward(self, x):
        if x.shape[-1] == 1:
            if x.ndim == 3:
                return x.view(len(x), -1)
            else:
                return x[:, 0]
        else:
            if self.mean:
                if x.ndim == 3:
                    xobs = x[:, :, 0]  # n_batch, n_embed
                    xagg = x[:, :, 1:].mean(dim=2)  # n_batch, n_embed
                    x = torch.cat([xobs, xagg], dim=1)  # n_batch, 2*n_embed
                else:  # no batch, single observation (used in sample..)
                    xobs = x[:, 0]  # n_embed
                    xagg = x[:, 1:].mean(dim=1)  # n_embed
                    x = torch.cat([xobs, xagg])  # 2*n_embed
                return x

            else:
                return x.view(len(x), -1)


class IdentityJRNMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_extra=0):
        return x


class StackContext(torch.nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, y):
        """
        Parameters
        ----------
        y : torch.Tensor, shape (n_batch, n_times + 1)
            Input of the StackContext layer.

        Returns
        --------
        context : torch.Tensor, shape (n_batch, n_embed + 1)
            Context where the input y has been encoded, except the last entry
            which is pass thru.
        """
        # The embedding net expect an extra dimension to handle n_extra. Add it
        # in x and remove it in x_embeded
        x = y[:, :-1, None]
        x_embed = self.embedding_net(x, n_extra=0)[:, :, 0]
        theta = y[:, -1:]
        return torch.cat([x_embed, theta], dim=1)


class JRNMMFlow_nflows_base(base.Distribution):
    def __init__(
        self,
        batch_theta,
        batch_x,
        embedding_net,
        n_layers=10,
        z_score_theta=True,
        z_score_x=True,
        aggregate=True,
    ):

        super().__init__()

        embedding_net = torch.nn.Sequential(
            embedding_net, AggregateInstances(mean=aggregate)
        )
        self._embedding_net = embedding_net

        # instantiate the flow
        flow = build_maf(
            batch_x=batch_theta,
            batch_y=batch_x,
            z_score_x=z_score_theta,
            z_score_y=z_score_x,
            embedding_net=embedding_net,
            num_transforms=n_layers,
        )

        self._flow = flow

    def _log_prob(self, inputs, context):
        logp = self._flow.log_prob(inputs, context)
        return logp

    def _sample(self, num_samples, context):
        samples = self._flow.sample(num_samples, context)[0]
        return samples

    def _transform(self, input, context):
        context = self._flow._embedding_net(context)
        transform = (self._flow._transform(input, context=context)[0], None)
        return transform

    def save_state(self, filename):
        state_dict = {}
        state_dict["flow"] = self._flow.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location="cpu")
        self._flow.load_state_dict(state_dict["flow"])


class JRNMMFlow_nflows_factorized(base.Distribution):
    def __init__(
        self,
        batch_theta,
        batch_x,
        embedding_net,
        n_layers_factor=5,
        z_score_theta=True,
        z_score_x=True,
    ):

        super().__init__()

        # flow_1 estimates p(gain | x, x1, ..., xn)
        # create a new net that embeds all n+1 observations and then aggregates
        # n of them via a sum operation
        embedding_net_1 = torch.nn.Sequential(
            embedding_net, AggregateInstances(mean=True)
        )
        self._embedding_net_1 = embedding_net_1

        # choose whether the embedding of the context should be done inside
        # the flow object or not; this can have an impact over the z-scoring
        batch_theta_1 = batch_theta[:, -1:]
        batch_context_1 = batch_x
        flow_1 = build_maf(
            batch_x=batch_theta_1,
            batch_y=batch_context_1,
            z_score_x=z_score_theta,
            z_score_y=z_score_x,
            embedding_net=embedding_net_1,
            num_transforms=n_layers_factor,
        )

        self._flow_1 = flow_1

        # flow_2 estimates p(C, mu, sigma, ... | x, gain)
        # create a new embedding next that handles the fact of having
        # a context that is a stacking of the embedded observation x
        # and the gain parameter
        embedding_net_2 = StackContext(embedding_net)
        self._embedding_net_2 = embedding_net_2

        batch_theta_2 = batch_theta[:, :-1]
        batch_context_2 = torch.cat(
            [batch_x[:, :, 0], batch_theta[:, -1:]], dim=1
        )  # shape (n_batch, n_times+1)
        flow_2 = build_maf(
            batch_x=batch_theta_2,
            batch_y=batch_context_2,
            z_score_x=z_score_theta,
            z_score_y=z_score_x,
            embedding_net=embedding_net_2,
            num_transforms=n_layers_factor,
        )

        self._flow_2 = flow_2

    def _log_prob(self, inputs, context):

        # logprob of the flow that models p(gain | x, x1, ..., xn)
        context_1 = context
        theta_1 = inputs[:, -1:]  # gain is the last parameter
        logp_1 = self._flow_1.log_prob(theta_1, context_1)

        # logprob of the flow that models p(C, mu, sigma | x, gain)
        gain = inputs[:, -1:]
        context_2 = torch.cat([context[:, :, 0], gain], dim=1)
        theta_2 = inputs[:, :-1]
        logp_2 = self._flow_2.log_prob(theta_2, context_2)

        return logp_1 + logp_2

    def _sample(self, num_samples, context):
        """Draw sample from the posterior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the posterior.
        context : torch.Tensor, shape (n_ctx, n_times, 1 + n_extra)
            Conditionning for the draw.

        Returns
        -------
        samples : shape (n_ctx, num_samples, n_params)
            Sample drawn from the distribution.
        """

        context_1 = context
        # shape (n_samples, 1)
        samples_flow_1 = self._flow_1.sample(num_samples, context_1)[0]
        context_2 = torch.cat(
            [context[:, :, 0].repeat(num_samples, 1), samples_flow_1], dim=1
        )
        context_2 = self._flow_2._embedding_net(context_2)
        noise = self._flow_2._distribution.sample(num_samples)
        samples_flow_2, _ = self._flow_2._transform.inverse(noise, context=context_2)

        samples = torch.cat([samples_flow_2, samples_flow_1], dim=1)
        return samples

    def _transform(self, input, context):
        # of the flow that models p(gain | x, x1, ..., xn)
        context_1 = self._flow_1._embedding_net(context)
        theta_1 = input[:, -1:]  # gain is the last parameter
        transform_1 = self._flow_1._transform(theta_1, context=context_1)[0]
        # of the flow that models p(C, mu, sigma | x, gain)
        context_2 = torch.cat([context[:, :, 0], theta_1], dim=1)
        context_2 = self._flow_2._embedding_net(context_2)
        theta_2 = input[:, :-1]
        transform_2 = self._flow_2._transform(theta_2, context=context_2)[0]
        transform = (torch.cat([transform_1, transform_2], dim=1), None)
        return transform

    def save_state(self, filename):
        state_dict = {}
        state_dict["flow_1"] = self._flow_1.state_dict()
        state_dict["flow_2"] = self._flow_2.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location="cpu")
        self._flow_1.load_state_dict(state_dict["flow_1"])
        self._flow_2.load_state_dict(state_dict["flow_2"])


def build_flow(
    batch_theta, batch_x, embedding_net, naive=False, aggregate=True, **kwargs
):

    if naive:
        flow = JRNMMFlow_nflows_base(
            batch_theta, batch_x, embedding_net, aggregate=aggregate, **kwargs
        )
    else:
        flow = JRNMMFlow_nflows_factorized(
            batch_theta, batch_x, embedding_net, **kwargs
        )

    return flow


def get_posterior(
    simulator,
    prior,
    summary_extractor,
    build_nn_posterior,
    meta_parameters,
    round_=0,
    batch_theta=None,
    batch_x=None,
    path=None,
):
    if path is not None:
        folderpath = path
    else:
        folderpath = Path.cwd() / meta_parameters["label"]

    if batch_theta is None:
        batch_theta = prior.sample((2,))
    if batch_x is None:
        batch_x = simulator(batch_theta)
        if summary_extractor is not None:
            batch_x = summary_extractor(batch_x)

    nn_posterior = build_nn_posterior(batch_theta=batch_theta, batch_x=batch_x)
    # nn_posterior.eval()
    # posterior = DirectPosterior(
    #     posterior_estimator=nn_posterior, prior=prior,
    #     x_shape=batch_x[0][None, :].shape
    # )

    state_dict_path = folderpath + f"nn_posterior_round_{round_:02}.pkl"
    nn_posterior.load_state(state_dict_path)
    # posterior = posterior.set_default_x(ground_truth["observation"])

    return nn_posterior


class NPE_JRNMM_lampe_base(nn.Module):
    def __init__(
        self,
        batch_theta,
        batch_x,
        flow=MAF,
        embedding_net=IdentityJRNMM(),
        aggregate=True,
        n_layers=10,
        hidden_features=[50] * 3,
        randperm=False,
        **kwargs,
    ) -> None:
        super().__init__()

        x_dim = AggregateInstances(mean=aggregate)(batch_x).shape[-1]

        self.npe = NPE(
            theta_dim=batch_theta.shape[-1],
            x_dim=x_dim,
            build=flow,
            transforms=n_layers,
            hidden_features=hidden_features,
            randperm=randperm,
            **kwargs,
        )

        embedding_net = torch.nn.Sequential(
            embedding_net, AggregateInstances(mean=aggregate)
        )
        self.embedding_net = embedding_net

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.npe(theta, self.embedding_net(x))

    def flow(self, x: torch.Tensor):  # -> Distribution
        return self.npe.flow(self.embedding_net(x))

