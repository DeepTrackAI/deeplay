from typing import Tuple, Type

from deeplay.initializers.initializer import Initializer
import torch.nn as nn


_kaiming_default_targets = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


class Kaiming(Initializer):
    def __init__(
        self,
        targets: Tuple[Type[nn.Module], ...] = _kaiming_default_targets,
        mode: str = "fan_out",
        nonlinearity: str = "relu",
        fill_bias: bool = True,
        bias: float = 0.0,
    ):
        super().__init__(targets)
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.fill_bias = fill_bias
        self.bias = bias

    def initialize_tensor(self, tensor, name):

        if name == "bias" and self.fill_bias:
            tensor.data.fill_(self.bias)
        else:
            nn.init.kaiming_normal_(
                tensor, mode=self.mode, nonlinearity=self.nonlinearity
            )
