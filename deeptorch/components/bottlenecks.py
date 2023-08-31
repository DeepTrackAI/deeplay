from ..templates import Layer
from ..core import DeepTorchModule
from ..config import Config, Ref

import torch
import torch.nn as nn


class Bottleneck:
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
