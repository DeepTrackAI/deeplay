from typing import Callable, Dict, List, Tuple, Union, Iterable

from deeplay.external import External
from .optimizer import Optimizer

import torch

torch.optim.Adam


class RMSprop(Optimizer):
    def __pre_init__(self, classtype=None, **optimzer_kwargs):
        optimzer_kwargs.pop("classtype", None)
        super().__pre_init__(torch.optim.RMSprop, **optimzer_kwargs)

    def __init__(
        self,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        **kwargs
    ): ...
