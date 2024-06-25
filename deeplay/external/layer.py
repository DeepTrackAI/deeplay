from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, overload
from warnings import warn

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from .external import External

__all__ = ["Layer"]


class Layer(External):

    _classwise_computed_values: Dict[Type[nn.Module], Dict[str, Any]] = {}

    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(classtype, *args, **kwargs)

    @overload
    def configure(self, **kwargs: Any) -> None: ...

    @overload
    def configure(self, classtype: Type[nn.Module], *args, **kwargs) -> None: ...

    @overload
    def configure(
        self,
        classtype: Type[nn.Conv2d],
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None: ...

    configure = External.configure

    def forward(self, *x, **kwargs):
        if torch.is_grad_enabled():
            warn(
                "The forward path of a Layer was called with gradients enabled. "
                "This likely means you forgot to build the model `model.build()`. "
                "If this was intentional, you can disable this warning by calling"
                "the module without gradients."
            )

        if self.classtype in self._classwise_computed_values:
            for name, func in self._classwise_computed_values[self.classtype].items():
                self.computed(name, func)
        new_values = self._update_computed_values(*x, **kwargs)
        layer = self.build()
        try:
            return layer(*x, **kwargs)
        except Exception as e:
            # We undo the computation of the computed values.
            my_config = self._user_config.take(self.tags, keep_list=True)
            for key, value in new_values.items():
                for keytuple, vlist in my_config.items():
                    if keytuple[-1] == key and vlist and vlist[-1].value == value:
                        vlist.pop()
                        break
            raise e

    @classmethod
    def register_computed(cls, ocls: Type[nn.Module], signal="forward"):
        """Register a function to compute a value from the inputs to the forward pass.

        This will register the function for a classtype. This allows for lazy computation
        of attrivutes that are not known at initialization time. For examples, the number
        of input channels to a convolutional layer can be computed from the input tensor.

        Parameters
        ----------
        ocls : Type[nn.Module]
            The class type to register the computed value for.
        signal : str, optional
            The signal to register the computed value for, by default "forward".

        Examples
        --------
        >>> @Layer.register_computed(nn.Conv2d)
        ... def in_channels(x):
        ...     return x.shape[-3]

        """

        def decorator(func: Callable):
            if ocls not in cls._classwise_computed_values:
                cls._classwise_computed_values[ocls] = {}
            cls._classwise_computed_values[ocls][func.__name__] = func
            return func

        return decorator


# 1D modules
@Layer.register_computed(nn.Linear)
def in_features(x):
    return x.shape[-1]


@Layer.register_computed(nn.Conv1d)
def in_channels(x):
    return x.shape[-2]


@Layer.register_computed(nn.ConvTranspose1d)
def in_channels(x):
    return x.shape[-2]


# 2D modules
@Layer.register_computed(nn.Conv2d)
def in_channels(x):
    return x.shape[-3]


@Layer.register_computed(nn.ConvTranspose2d)
def in_channels(x):
    return x.shape[-3]


# 3D modules
@Layer.register_computed(nn.Conv3d)
def in_channels(x):
    return x.shape[-4]


@Layer.register_computed(nn.ConvTranspose3d)
def in_channels(x):
    return x.shape[-4]


# Normalization modules
@Layer.register_computed(nn.BatchNorm1d)
def num_features(x):
    return x.shape[1] if x.dim() == 2 else x.shape[-2]


@Layer.register_computed(nn.GroupNorm)
def num_channels(x):
    return x.shape[1]


@Layer.register_computed(nn.InstanceNorm1d)
def num_features(x):
    return x.shape[-2]


@Layer.register_computed(nn.LayerNorm)
def normalized_shape(x):
    return x.shape[-1]


@Layer.register_computed(nn.BatchNorm2d)
def num_features(x):
    return x.shape[-3]


@Layer.register_computed(nn.InstanceNorm2d)
def num_features(x):
    return x.shape[-3]


@Layer.register_computed(nn.BatchNorm3d)
def num_features(x):
    return x.shape[-4]


@Layer.register_computed(nn.InstanceNorm3d)
def num_features(x):
    return x.shape[-4]


# Misc modules
@Layer.register_computed(nn.Bilinear)
def in1_features(x0, x1):
    return x0.shape[-1]


@Layer.register_computed(nn.Bilinear)
def in2_features(x0, x1):
    return x1.shape[-1]
