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
    ):
        super().__init__(targets)
        self.mode = mode
        self.nonlinearity = nonlinearity

    def initialize_weight(self, tensor):
        nn.init.kaiming_normal_(tensor, mode=self.mode, nonlinearity=self.nonlinearity)

    def initialize_bias(self, tensor):
        tensor.data.fill_(0.0)
