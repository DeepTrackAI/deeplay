import torch.nn as nn

default = object()


class LazyModule:
    def __init__(*args, **kwargs):
        ...

    def build(self, *args, **kwargs):
        raise NotImplementedError


class Default(LazyModule):
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

    def __init__(self, value, default, *args, **kwargs):
        self.value = value
        self.default = default
        self.args = args
        self.kwargs = kwargs

    def build(self):
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
        if self.value is None or self.value is default:
            return self._try_build(self.default, *self.args, **self.kwargs)
        elif isinstance(self.value, dict):
            kwargs = {
                **self.kwargs,
                **self.value,
            }
            return self._try_build(self.default, *self.args, **kwargs)
        elif isinstance(self.value, nn.Module):
            return self.value
        elif isinstance(self.value, LazyModule):
            return self.value.build()
        else:
            print(self.value, self.default, self.args, self.kwargs)
            raise ValueError(f"Invalid value: {self.value}")

    def _try_build(self, default, *args, **kwargs):
        # Try to build the layer. If it fails, try to build it without the

        if isinstance(default, LazyModule):
            return default.build()

        res = default(*args, **kwargs)
        if isinstance(res, LazyModule):
            return res.build()
        else:
            return res
