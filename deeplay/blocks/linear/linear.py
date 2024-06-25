from __future__ import annotations
from typing import Optional, Type, Union
from typing_extensions import Self
import warnings
import torch.nn as nn

from deeplay.blocks.base import BaseBlock
from deeplay.external import Layer
from deeplay.module import DeeplayModule
from deeplay.ops.merge import Add, MergeOp
import torch
import torch.nn as nn


class LinearBlock(BaseBlock):
    """Convolutional block with optional normalization and activation."""

    def __init__(
        self,
        in_features: Optional[int],
        out_features: int,
        bias: bool = True,
        **kwargs,
    ):

        self.in_features = in_features
        self.out_features = out_features

        if in_features is None:
            layer = Layer(
                nn.LazyLinear,
                out_features,
                bias=bias,
            )
        else:
            layer = Layer(
                nn.Linear,
                in_features,
                out_features,
                bias=bias,
            )

        super().__init__(layer=layer, **kwargs)

    def get_default_normalization(self) -> DeeplayModule:
        return Layer(nn.BatchNorm1d, self.out_features)

    def get_default_activation(self) -> DeeplayModule:
        return Layer(nn.ReLU)

    def get_default_shortcut(self) -> DeeplayModule:
        if self.in_features == self.out_features:
            return Layer(nn.Identity)
        else:
            return Layer(nn.Linear, self.in_features, self.out_features)

    def get_default_merge(self) -> MergeOp:
        return Add()

    def call_with_dummy_data(self):
        x = torch.randn(2, self.in_features)
        return self(x)
