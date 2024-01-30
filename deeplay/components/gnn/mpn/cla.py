from typing import Optional, Literal, Any, overload, Union, Type, List

from deeplay.blocks.sequential import SequentialBlock
from deeplay import DeeplayModule

import torch.nn as nn


class CombineLayerActivation(DeeplayModule):
    combine: nn.Module
    layer: nn.Module
    activation: nn.Module

    def __init__(
        self,
        combine: nn.Module,
        layer: nn.Module,
        activation: nn.Module,
    ):
        super().__init__()
        self.combine = combine
        self.layer = layer
        self.activation = activation

    def forward(self, *x):
        x = self.get_forward_args(x)
        x = self.combine(*x)
        x = self.layer(x)
        x = self.activation(x)
        return x

    def get_forward_args(self, x):
        return x

    @overload
    def configure(self, name: Literal["combine"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["activation"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
