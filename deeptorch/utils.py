import torch
import torch.nn as nn
import torch.nn.functional as F


class NamedSequential(nn.Module):
    """A sequential module with named layers.
    Used for pretty printing.

    Is evaluted in the order of insertion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._modules = nn.ModuleDict(*args, **kwargs)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
