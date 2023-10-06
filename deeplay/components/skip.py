from ..core.templates import Layer
from ..core.core import DeeplayModule
from ..core.config import Config
from ..core.utils import center_pad_to_largest, center_crop_to_smallest

import torch
import torch.nn as nn

__all__ = ["Skip", "Concatenate", "Add"]


class Skip(DeeplayModule):
    defaults = Config().inputs[0](nn.Identity)

    def __init__(self, inputs, func):
        """Skip module.

        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        func : callable
            Function to apply to the inputs.
        """

        super().__init__(inputs=inputs, func=func)

        self.func = self.attr("func")
        self.inputs = self.new("inputs")

    def forward(self, *x):
        inputs = [inp(*x) for inp in self.inputs]
        return self.func(*inputs)


class Concatenate(DeeplayModule):
    defaults = Config().merge(None, Skip.defaults).dim(1).mismatch_strategy(None)

    def __init__(self, inputs, dim=1, mismatch_strategy=None):
        """Concatenate module.

        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        dim : int
            Dimension to concatenate on.
            Default is 1.
        mismatch_strategy : str or none
            Strategy to use when the inputs have different shapes.
            Allowed values are "pad" and "crop" or None.
        """

        super().__init__(inputs=inputs, dim=dim, mismatch_strategy=mismatch_strategy)

        self.dim = self.attr("dim")
        self.mismatch_strategy = self.attr("mismatch_strategy")

    def forward(self, *x):
        if self.mismatch_strategy == "pad":
            x = center_pad_to_largest(x)
        elif self.mismatch_strategy == "crop":
            x = center_crop_to_smallest(x)
        print([i.shape for i in x])
        return torch.cat(x, dim=self.dim)


class Add(DeeplayModule):
    defaults = Config().merge(None, Skip.defaults)

    def __init__(self):
        """Add module."""

        super().__init__()

    def forward(self, *x):
        return sum(x)
