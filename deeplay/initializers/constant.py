from typing import Tuple, Type

from deeplay.initializers.initializer import Initializer
import torch.nn as nn

_constant_default_targets = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


class InitializerConstant(Initializer):
    def __init__(
        self,
        targets: Tuple[Type[nn.Module], ...] = _constant_default_targets,
        weight: float = 1.0,
        bias: float = 0.0,
    ):
        super().__init__(targets)
        self.weight = weight
        self.bias = bias

    def initialize_weight(self, tensor):
        tensor.data.fill_(self.weight)

    def initialize_bias(self, tensor):
        tensor.data.fill_(self.bias)
