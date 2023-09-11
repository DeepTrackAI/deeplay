from ..templates import Layer
from ..core import DeepTorchModule
from ..config import Config, Ref

import torch
import torch.nn as nn

__all__ = ["Bottleneck"]


class Bottleneck(DeepTorchModule):
    defaults = (
        Config()
        .hidden_dim(2)
        .flatten(nn.Flatten)
        .layer(nn.LazyLinear, out_features=Ref("hidden_dim"))
        .activation(nn.Tanh)
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


class NormalDistribtionSampler(DeepTorchModule):
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
        loc = self.loc(x)
        scale = self.scale(x)
        return torch.distributions.Normal(loc=loc, scale=scale).sample(x.shape)


class VariationalBottleneck(DeepTorchModule):
    defaults = (
        Config()
        .hidden_dim(4)
        .flatten(nn.Flatten)
        .layer(nn.LazyLinear, out_features=Ref("hidden_dim"))
        .samplers(NormalDistribtionSampler)
        .activation(nn.Tanh)
    )

    def __init__(self, out_channels=1, activation=nn.ReLU):
        super().__init__(out_channels=out_channels, activation=activation)

        self.hidden_dim = self.attr("hidden_dim")

        # Create samplers until we have enough outputs
        self.samplers = []
        n_sampler_outputs = 0
        while n_sampler_outputs < self.hidden_dim:
            sampler = self.new("samplers", i=len(self.samplers))
            n_sampler_outputs += sampler.n_outputs
            self.samplers.append(sampler)

        self.flatten = self.new("flatten")
        self.layer = self.new("layer")
        self.activation = self.new("activation")

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer(x)

        # for i in range(hidden_dim):

        # return x
