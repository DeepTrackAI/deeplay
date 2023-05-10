import torch
import torch.nn as nn


class Block:
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
        if obj is None:
            return True
        if isinstance(obj, dict):

            if "in_channels" in obj or "out_channels" in obj:
                raise ValueError(
                    f"in_channels and out_channels should not be specified in the config dict. Got {obj}."
                )

            return True

        if isinstance(obj, nn.Module):
            return True

        raise ValueError(f"Invalid config: {obj}")


__all__ = [Block]
