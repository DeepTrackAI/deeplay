from typing import Tuple, Type

from deeplay.initializers.initializer import Initializer
import torch.nn as nn

_normal_default_targets = (
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


class Normal(Initializer):
    def __init__(
        self,
        targets: Tuple[Type[nn.Module], ...] = _normal_default_targets,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__(targets)
        self.mean = mean
        self.std = std

    def initialize_tensor(self, tensor, name):
        tensor.data.normal_(mean=self.mean, std=self.std)
