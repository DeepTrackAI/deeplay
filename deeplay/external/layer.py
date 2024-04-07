from typing import Any, Type, overload, Dict, Union, List, Optional
from .external import External
from functools import partial

import torch.nn as nn
from torch.nn.modules.conv import _size_2_t


class Layer(External):
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
        raise RuntimeError(
            "Unexpected call to forward. Did you forget to call `.build()` or `.create()` on the model?"
        )
