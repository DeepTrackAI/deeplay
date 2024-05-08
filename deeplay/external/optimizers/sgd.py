from typing import Callable, Dict, List, Tuple, Union, Iterable

from deeplay.external import External
from .optimizer import Optimizer

import torch


class SGD(Optimizer):
    def __pre_init__(self, **optimzer_kwargs):
        optimzer_kwargs.pop("classtype", None)
        super().__pre_init__(torch.optim.SGD, **optimzer_kwargs)

    def __init__(
        self,
        params=None,
        lr=0.1,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach=None,
        differentiable=False
    ): ...
