import torch.nn as nn


class Default(nn.Module):
    """Layer node with default value.

    For example, if you want to use a default nn.Conv2d layer, you can use
    `Default(config, nn.Conv2d)`. If `config` is None, a default nn.Conv2d
    layer is used. If `config` is a dict, it is used as kwargs for nn.Conv2d.
    If `config` is a nn.Module, it is used directly.

    The order of precedence for kwargs is: `__init__` < `config` < `build`.
    Meaning, if you specify a value in `__init__`, it will be overwritten by
    `config` and `build`. If you specify a value in `config`, it will be
    overwritten by `build`. If you specify a value in `build`, it will be used.

    Parameters
    ----------
    value : None, Dict, nn.Module
        Config for the layer. If None, the default value is used.
    default : nn.Module class
        Default value to use if `value` is None.
    **kwargs
        Additional kwargs to use for the default value.


    """

    def __init__(self, value, default, **kwargs):
        self.value = value
        self.default = default
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        """Builds the layer.

        Main purpose of this method is to allow the user to specify the
        `in_channels` and `out_channels` of the layer.

        Parameters
        ----------
        *args
            Positional arguments to pass to the layer. Typically, this is
            `in_channels` and `out_channels`.
        **kwargs
            Keyword arguments to pass to the layer. These will overwrite the
            values in `__init__` and `value`.

        """
        if self.value is None:
            kwargs = {**self.kwargs, **kwargs}
            return self._try_build(self.default, *args, **kwargs)
        elif isinstance(self.value, dict):
            kwargs = {**self.kwargs, **self.value, **kwargs}
            return self._try_build(self.default, *args, **kwargs)
        elif isinstance(self.value, nn.Module):
            return self.value
        else:
            raise ValueError(f"Invalid value: {self.value}")

    def _try_build(self, default, *args, **kwargs):
        # Try to build the layer. If it fails, try to build it without the
        # positional arguments. If that fails, raise the original error.
        try:
            return default(*args, **kwargs)
        except TypeError as e:
            try:
                # Try to build without the positional arguments. This is
                # useful for layers that don't have `in_channels` and
                # `out_channels` as the first two arguments.
                return default(**kwargs)
            except TypeError:
                raise e


__all__ = [Default]
