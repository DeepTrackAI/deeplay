from typing import (
    List,
    overload,
    Optional,
    Literal,
    Union,
    Any,
    Type,
    Callable,
    Sequence,
)


import torch
import torch.nn as nn

from .block import Block
from deeplay.external import Layer
from deeplay import DeeplayModule


class BaseResidual(Block):

    in_layer: Layer
    in_normalization: Layer
    in_activation: Layer
    out_layer: Layer
    out_normalization: Layer
    out_activation: Layer
    shortcut: Layer
    order: List[str]

    def __init__(
        self,
        in_layer: Union[DeeplayModule, Type[nn.Module]],
        in_normalization: Union[DeeplayModule, Type[nn.Module]],
        in_activation: Union[DeeplayModule, Type[nn.Module]],
        out_layer: Union[DeeplayModule, Type[nn.Module]],
        out_normalization: Union[DeeplayModule, Type[nn.Module]],
        out_activation: Union[DeeplayModule, Type[nn.Module]],
        shortcut: Union[DeeplayModule, Type[nn.Module]] = nn.Identity,
        order: List[str] = [
            "in_layer",
            "in_normalization",
            "in_activation",
            "out_layer",
            "out_normalization",
            "merge",
            "out_activation",
        ],
        **kwargs,
    ):
        super().__init__(
            in_layer=in_layer,
            in_normalization=in_normalization,
            in_activation=in_activation,
            out_layer=out_layer,
            out_normalization=out_normalization,
            out_activation=out_activation,
            shortcut=shortcut,
            **kwargs,
        )

        # self.layer = self["in_layer|out_layer"]
        # self.normalization = self["in_normalization|out_normalization"]
        # self.activation = self["in_activation|out_activation"]
        self.order = order

    def forward(self, x):
        shortcut = self.shortcut(x)
        for name in self.order:
            if name == "merge":
                x = x + shortcut
            else:
                x = getattr(self, name)(x)
        return x


class Conv2dResidual(BaseResidual):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        in_normalization: Union[DeeplayModule, Type[nn.Module]] = nn.LazyBatchNorm2d,
        in_activation: Union[DeeplayModule, Type[nn.Module]] = nn.ReLU,
        out_normalization: Union[DeeplayModule, Type[nn.Module]] = nn.LazyBatchNorm2d,
        out_activation: Union[DeeplayModule, Type[nn.Module]] = nn.ReLU,
        shortcut: Union[DeeplayModule, Type[nn.Module]] = nn.Identity,
        order: List[str] = [
            "in_layer",
            "in_normalization",
            "in_activation",
            "out_layer",
            "out_normalization",
            "merge",
            "out_activation",
        ],
        **kwargs,
    ):
        if in_channels != out_channels and shortcut == nn.Identity:
            raise ValueError(
                "Shortcut must be set to a non-identity layer if in_channels != out_channels"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        super().__init__(
            in_layer=Layer(
                nn.Conv2d,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
            in_normalization=in_normalization,
            in_activation=in_activation,
            out_layer=Layer(
                nn.Conv2d,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
            out_normalization=out_normalization,
            out_activation=out_activation,
            shortcut=shortcut,
            order=order,
            **kwargs,
        )

    def validate_after_build(self):
        in_channels = self.in_channels
        x = torch.rand(1, in_channels, 32, 32)
        y = self(x)
