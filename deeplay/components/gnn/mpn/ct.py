from typing import Optional, Literal, Any, overload, Union, Type

from deeplay import DeeplayModule, Layer
from deeplay.blocks import LayerActivation
from deeplay.components.cnn import concat

import torch.nn as nn


class CombineTransform(DeeplayModule):
    in_features: Optional[int]
    out_features: int
    activation: DeeplayModule

    def __init__(
        self,
        in_features: Optional[int],
        out_features: int,
        activation: DeeplayModule = Layer(nn.ReLU),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if out_features <= 0:
            raise ValueError(
                f"Number of output features must be positive, got {out_features}"
            )

        if in_features is not None and in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        self.combine = concat()
        self.mlp = LayerActivation(
            layer=(
                Layer(nn.Linear, in_features, out_features)
                if in_features
                else Layer(nn.LazyLinear, out_features)
            ),
            activation=activation.new(),
        )

    def forward(self, *x):
        return self.mlp(
            self.combine(*self.get_forward_args(x)),
        )

    def get_forward_args(self, x):
        return x

    @overload
    def configure(self, name: Literal["combine"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["mlp"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
