from typing import Any, Callable, Type, overload, ParamSpec
from .external import External

import torch.nn as nn

P = ParamSpec("P")


class Layer(External):
    def __pre_init__(self, cls: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(cls, *args, **kwargs)

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, classtype: Callable[P, nn.Module], **kwargs: P.kwargs) -> None:
        ...

    configure = External.configure
