from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union, Callable

from deeplay.module import DeeplayModule


class ShapeOp(DeeplayModule): ...


__all__ = ["Flatten", "View", "Reshape", "Squeeze", "Unsqueeze", "Permute"]


class Flatten(ShapeOp):

    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


class View(ShapeOp):
    def __init__(
        self,
        *shape: int | Callable[[Tuple[int, ...]], Tuple[int, ...]],
    ):
        if len(shape) == 1:
            self.shape = shape[0]
        else:
            self.shape = shape

    def forward(self, x):
        if callable(self.shape):
            shape = self.shape(x.shape)
        else:
            shape = self.shape
        return x.view(*shape)


class Reshape(View):
    def __init__(self, *shape: int | Callable[[Tuple[int, ...]], Tuple[int, ...]]):
        super().__init__(*shape)


class Squeeze(ShapeOp):
    def __init__(self, dim: Optional[int] = None):
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)


class Unsqueeze(ShapeOp):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Permute(ShapeOp):
    def __init__(self, *dims: int):
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
