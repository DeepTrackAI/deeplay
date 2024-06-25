from typing import Union
import torch.nn as nn
import torch
from deeplay.external.layer import Layer
from deeplay.module import DeeplayModule

__all__ = ["MergeOp", "Add", "Cat", "Lambda"]


class MergeOp(DeeplayModule):
    def __init__(self): ...

    def forward(self, *x):
        raise NotImplementedError


class Add(MergeOp):
    def forward(self, *x):
        return torch.stack(x).sum(dim=0)

    def _compute_output_shape(self, *x):
        return x[0]


class Cat(MergeOp):
    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)

    def _compute_output_shape(self, *x):
        x0 = list(x[0])
        x0[self.dim] = sum([v[self.dim] for v in x])
        return tuple(x0)


class Lambda(MergeOp):
    def __init__(self, fn):
        self.fn = fn

    def forward(self, *x):
        return self.fn(x)
