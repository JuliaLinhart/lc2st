import nflows.distributions as distributions
import nflows.transforms as transforms
import torch
import torch.distributions as D
import torch.nn as nn

from nflows.flows import Flow
from nflows.nn import nets
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.sbiutils import standardizing_transform
from torch.nn import functional as F


def construct_maf(
    features,
    hidden_features,
    context_features=None,
    num_layers=5,
    random_permutation=False,
    standardize_transform=True,
):
    """ MAF as in nflows but with standardizing_transform from sbi

    Returns:
        Neural network (nflows.Flow).
    """

    d = features[0].size(0)

    # Base dist
    base_dist = distributions.StandardNormal(shape=torch.Size([d]))

    # Transformations: MAF/MADE layers
    trans_components = []
    num_layers = num_layers
    for _ in range(num_layers):
        if random_permutation:
            trans_components.append(transforms.RandomPermutation(features=d))
        else:
            trans_components.append(transforms.ReversePermutation(features=d))
        trans_components.append(
            transforms.MaskedAffineAutoregressiveTransform(
                features=d,
                hidden_features=hidden_features,
                context_features=context_features,
            )
        )
    transform = transforms.CompositeTransform(trans_components)

    if standardize_transform:
        transform = transforms.CompositeTransform(
            [standardizing_transform(features), transform]
        )

    # Construct Flow
    flow = Flow(transform, base_dist)

    return flow


def construct_nsf(
    batch_x: torch.Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    context_features: int = None,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds NSF as in sbi.flow (copy paste but without y_batch and 1D case)

    Returns:
        Neural network (nflows.Flow).
    """
    x_numel = batch_x[0].numel()
    mask_in_layer = lambda i: create_alternating_binary_mask(
        features=x_numel, even=(i % 2 == 0)
    )
    conditioner = lambda in_features, out_features: nets.ResidualNet(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        context_features=context_features,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    )

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.PiecewiseRationalQuadraticCouplingTransform(
                        mask=mask_in_layer(i),
                        transform_net_create_fn=conditioner,
                        num_bins=num_bins,
                        tails="linear",
                        tail_bound=3.0,
                        apply_unconditional_transform=False,
                    ),
                    transforms.LULinear(x_numel, identity_init=True),
                ]
            )
            for i in range(num_transforms)
        ]
    )

    if z_score_x:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    distribution = distributions.StandardNormal((x_numel,))
    neural_net = Flow(transform, distribution, embedding_net)

    return neural_net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# CDF function of a (conditional) flow evaluated in x: F_{Q|context}(x)
def cdf_flow(x, context, flow, base_dist=D.Normal(0, 1)):
    return base_dist.cdf(flow._transform(x, context=context)[0])
