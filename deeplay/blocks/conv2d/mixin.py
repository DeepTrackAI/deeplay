from __future__ import annotations

import warnings
from typing import List

import torch.nn as nn
from typing_extensions import Self

from deeplay.external import Layer
from deeplay.list import LayerList


class BaseConvBlockMixin(nn.Module):

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def pooled(self, pool: Layer) -> Self:
        raise NotImplementedError

    def normalized(self, normalization: Layer) -> Self:
        raise NotImplementedError

    def strided(self, stride: int | tuple[int], remove_pool: bool = True) -> Self:
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def base(self) -> Self:
        self.configure(mode="base")
        return self

    def multi(self, hidden_channels: List[int]) -> Self:
        self.configure(mode="multi", hidden_channels=hidden_channels)
        return self

    def residual(
        self,
        hidden_channels: Optional[List[int]] = None,
        merge_after: str = "activation",
        merge_block: int = -1,
        shortcut: Layer = Layer(nn.Identity),
    ) -> Self:
        if hidden_channels is None:
            hidden_channels = [self.out_channels]
        self.configure(
            mode="residual",
            hidden_channels=hidden_channels,
            merge_after=merge_after,
            merge_block=merge_block,
            shortcut=shortcut,
        )
        return self
