from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, overload

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from .external import External


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
        if self.classtype in self._classwise_computed_values:
            for name, func in self._classwise_computed_values[self.classtype].items():
                self.computed(name, func)
        self._update_computed_values(*x, **kwargs)
        layer = self.build()
        return layer(*x, **kwargs)

    @classmethod
    def register_computed(cls, ocls: Type[nn.Module], signal="forward"):
        def decorator(func: Callable):
            if ocls not in cls._classwise_computed_values:
                cls._classwise_computed_values[cls] = {}
            cls._classwise_computed_values[cls][func.__name__] = func
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

@Layer.register_computed(nn.LayerNorm):
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

