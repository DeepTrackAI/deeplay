from __future__ import annotations
from typing import Optional, Type, Union
from typing_extensions import Self
import warnings
import torch.nn as nn

from deeplay.blocks.base import BaseBlock
from deeplay.external import Layer
from deeplay.module import DeeplayModule


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

    def normalized(
        self,
        normalization: Union[Type[nn.Module], DeeplayModule] = nn.BatchNorm1d,
        mode="append",
        after=None,
    ) -> Self:
        did_replace = mode == "replace" and "normalization" in self.order

        super().normalized(normalization, mode=mode, after=after)

        if did_replace:
            # Assume num_features is already correct
            return self

        idx = self.order.index("normalization")
        # if layer or blocks before normalization
        if any(name in self.order[:idx] for name in ["layer", "blocks"]):
            channels = self.out_features
        else:
            channels = self.in_features

        self._configure_normalization(channels)

        return self

    def _configure_normalization(self, channels):
        ntype: Type[nn.Module] = self.normalization.classtype

        if ntype == nn.BatchNorm1d:
            self.normalization.configure(num_features=channels)
        elif ntype == nn.GroupNorm:
            num_groups = self.normalization.kwargs.get("num_groups", 1)
            self.normalization.configure(num_groups=num_groups, num_channels=channels)
        elif ntype == nn.InstanceNorm1d:
            self.normalization.configure(num_features=channels)
        elif ntype == nn.LayerNorm:
            self.normalization.configure(normalized_shape=channels)
