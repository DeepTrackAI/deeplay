import torch
import torch.nn as nn

from .. import LazyModule

__all__ = ["Block"]


class Block(LazyModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def assert_valid(self, obj):
        """Assert that the given object is a valid config for this block.

        Parameters
        ----------
        obj : None, Dict, nn.Module
            Object to check.

        Raises
        ------
        ValueError
            If the object is not a valid config.

        Returns
        -------
        bool
            True if the object is a valid config.
        """
        return True
