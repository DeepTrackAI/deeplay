from typing import Callable, Dict, List, Tuple, Union, Iterable

from ..external import External
from .optimizer import Optimizer

import torch


class Adam(Optimizer):
    def __pre_init__(self, **optimzer_kwargs):
        optimzer_kwargs.pop("classtype", None)
        super().__pre_init__(torch.optim.Adam, **optimzer_kwargs)
