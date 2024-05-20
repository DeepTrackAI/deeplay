from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, overload

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from .external import External


class Layer(External):
    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        self._computed: Dict[str, Callable[..., Any]] = {}
        self._input_shape: Optional[Tuple[Tuple[int, ...], ...]] = None
        super().__pre_init__(classtype, *args, **kwargs)

    def computed(self, name: str, func: Callable[..., Any]):
        return self._add_computed(name, func)

    def _add_computed(self, name: str, func: Callable[..., Any]):
        self._computed[name] = func
        return self

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if self._input_shape is None:
            raise RuntimeError(
                "Input shape is not set. Please set the input shape before building the model."
            )
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Tuple[Tuple[int, ...], ...]):
        self._input_shape = value
        self._update_computed_values()

    def _update_computed_values(self):
        for name, func in self._computed.items():
            self.configure(**{name: func(self.input_shape)})

    def output_shape(
        self, *args: Union[torch.Tensor, Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        if args:
            self.input_shape = tuple(
                [
                    tuple(arg.shape) if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ]
            )

        x = self.input_shape
        layer = self.build()
        x = tuple([torch.randn(_x) for _x in x])
        x = layer(*x)
        if isinstance(x, torch.Tensor):
            x = tuple(x.shape)
        elif isinstance(x, (list, tuple)):
            x = tuple([_x.shape for _x in x])

        return x

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
