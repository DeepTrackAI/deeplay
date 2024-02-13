from typing import Any, Type, overload, Dict, Union, List, Optional
from .external import External
from functools import partial

import torch.nn as nn


class Layer(External):
    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(classtype, *args, **kwargs)

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, classtype, **kwargs) -> None:
        ...

    configure = External.configure

    def forward(self, x):
        raise RuntimeError(
            "Unexpected call to forward. Did you forget to call `.build()` or `.create()` on the model?"
        )
