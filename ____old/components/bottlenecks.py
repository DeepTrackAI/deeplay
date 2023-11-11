from ..core.templates import Layer
from ..core.core import DeeplayModule
from ..core.config import Config, Ref

import torch
import torch.nn as nn

__all__ = ["Bottleneck", "VariationalBottleneck"]


class Bottleneck(DeeplayModule):
    defaults = (
        Config()
        .hidden_dim(2)
        .flatten(nn.Flatten)
        .layer(nn.LazyLinear, out_features=Ref("hidden_dim"))
        .activation(nn.Identity)
    )

    def __init__(self, out_channels=1, activation=nn.ReLU):
        super().__init__(out_channels=out_channels, activation=activation)

        self.hidden_dim = self.attr("hidden_dim")
        self.flatten = self.new("flatten")
        self.layer = self.new("layer")
        self.activation = self.new("activation")

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer(x)
        x = self.activation(x)

        return x


class AbstractSampler(DeeplayModule):
    n_inputs: int
    n_outputs: int


class NormalDistribtionSampler(DeeplayModule):
    defaults = (
        Config()
        .prior(torch.distributions.Normal, loc=0, scale=1)
        .loc(nn.Identity)
        .scale(lambda: torch.exp)
    )
    n_inputs = 2
    n_outputs = 1

    def __init__(self, prior=None, loc=None, scale=None):
        super().__init__(prior=prior, loc=loc, scale=scale)
        self.prior = self.new("prior")
        self.loc = self.new("loc")
        self.scale = self.new("scale")

    def forward(self, x):
        loc = self.loc(x[:, 0])
        scale = self.scale(x[:, 1])
        return torch.distributions.Normal(loc=loc, scale=scale)


class VariationalBottleneck(DeeplayModule):
    defaults = (
        Config()
        .hidden_dim(4)
        .flatten(nn.Flatten)
        .layer(nn.LazyLinear)
        .samplers(NormalDistribtionSampler)
        .activation(nn.Identity)
    )

    def __init__(self):
        super().__init__()

        self.hidden_dim: int = self.attr("hidden_dim")

        # Create samplers until we have enough outputs
        samplers = []
        __n_sampler_inputs = 0
        __n_sampler_outputs = 0

        while __n_sampler_outputs < self.hidden_dim:
            sampler: AbstractSampler = self.new("samplers", i=len(samplers))
            __n_sampler_inputs += sampler.n_inputs
            __n_sampler_outputs += sampler.n_outputs
            samplers.append(sampler)

        self.flatten = self.new("flatten")
        self.layer = self.new(
            "layer", extra_kwargs=dict(out_features=__n_sampler_inputs)
        )
        self.samplers = nn.ModuleList(samplers)

        self.activation = self.new("activation")

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer(x)
        splits = torch.split(x, [s.n_inputs for s in self.samplers], dim=1)
        posteriors = [s(_x) for s, _x in zip(self.samplers, splits)]
        latent = torch.stack([p.rsample() for p in posteriors], dim=1)
        x = self.activation(latent)

        return x, posteriors
        # for i in range(hidden_dim):

        # return x

    @property
    def priors(self):
        return [s.prior for s in self.samplers]
