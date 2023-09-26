from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref

import torch
import torch.nn as nn

__all__ = ["Skip", "Concatenate", "Add"]


def center_pad_to_largest(inputs):
    """Pad the inputs to the largest input.
    Two first dimensions are assumed to be batch and channel and are not padded.

    Parameters
    ----------
    inputs : list of torch.Tensor
        List of inputs.

    Returns
    -------
    list of torch.Tensor
        List of padded inputs.
    """

    max_shape = max(inp.shape[2:] for inp in inputs)
    return [center_pad(inp, max_shape) for inp in inputs]


def center_pad(inp, shape):
    """Pad the input to the given shape.
    Two first dimensions are assumed to be batch and channel and are not padded.

    Parameters
    ----------
    inp : torch.Tensor
        Input.
    shape : tuple of int
        Shape to pad to.

    Returns
    -------
    torch.Tensor
        Padded input.
    """
    pads = []
    for i, s in zip(inp.shape[2:], shape):
        before = (s - i) // 2
        after = s - i - before
        pads.extend([before, after])
    
    # reversed since pad expects padding for the last dimension first
    pads = pads[::-1]

    return nn.functional.pad(inp, pads)


def center_crop_to_smallest(inputs):
    """Crop the inputs to the smallest input.
    Two first dimensions are assumed to be batch and channel and are not cropped.

    Parameters
    ----------
    inputs : list of torch.Tensor
        List of inputs.

    Returns
    -------
    list of torch.Tensor
        List of cropped inputs.
    """

    min_shape = min(inp.shape[2:] for inp in inputs)
    return [center_crop(inp, min_shape) for inp in inputs]


def center_crop(inp, shape):
    """Crop the input to the given shape.
    Two first dimensions are assumed to be batch and channel and are not cropped.

    Parameters
    ----------
    inp : torch.Tensor
        Input.
    shape : tuple of int
        Shape to crop to.

    Returns
    -------
    torch.Tensor
        Cropped input.
    """

    slices = []
    for i, s in zip(inp.shape[2:], shape):
        before = (i - s) // 2
        after = i - s - before
        slices.append(slice(before, i - after))
    
    return inp[..., slices]

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

        super().__init__(inputs=inputs, dim=dim)

        self.dim = self.attr("dim")
        self.mismatch_strategy = self.attr("mismatch_strategy")
        self.inputs = self.new("inputs")


    def forward(self, *x):
        inputs = [inp(*x) for inp in self.inputs]

        if self.mismatch_strategy == "pad":
            inputs = center_pad_to_largest(inputs)
        elif self.mismatch_strategy == "crop":
            inputs = center_crop_to_smallest(inputs)

        return torch.cat(inputs, dim=self.dim)


class Add(DeeplayModule):
    defaults = Config().merge(None, Skip.defaults)

    def __init__(self, inputs):
        """Add module.

        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        """

        super().__init__(inputs=inputs)

        self.inputs = self.new("inputs")

    def forward(self, *x):
        inputs = [inp(*x) for inp in self.inputs]
        return sum(inputs)
