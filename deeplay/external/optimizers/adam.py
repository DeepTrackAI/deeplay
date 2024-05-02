from typing import Callable, Dict, List, Tuple, Union, Iterable

from deeplay.external import External
from .optimizer import Optimizer

import torch


class Adam(Optimizer):
    def __pre_init__(self, **optimzer_kwargs):
        optimzer_kwargs.pop("classtype", None)
        super().__pre_init__(torch.optim.Adam, **optimzer_kwargs)

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        **kwargs
    ): ...


class AdamW(Optimizer):
    def __pre_init__(self, **optimzer_kwargs):
        optimzer_kwargs.pop("classtype", None)
        super().__pre_init__(torch.optim.AdamW, **optimzer_kwargs)

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        **kwargs
    ): ...
