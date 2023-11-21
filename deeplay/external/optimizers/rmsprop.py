from typing import Callable, Dict, List, Tuple, Union, Iterable

from ..external import External
from .optimizer import Optimizer

import torch

torch.optim.Adam


class RMSprop(Optimizer):
    def __pre_init__(self, classtype=None, **optimzer_kwargs):
        super().__pre_init__(torch.optim.RMSprop, **optimzer_kwargs)

    def __init__(self, **optimzer_kwargs):
        super().__init__(**optimzer_kwargs)

    # def __init__(self, **optimzer_kwargs):
    #     super().__init__(**optimzer_kwargs)

    def params(
        self,
        func: Callable[
            [],
            Iterable[torch.nn.Parameter]
            | Dict[str, Iterable[torch.nn.Parameter]]
            | List[Dict[str, Iterable[torch.nn.Parameter]]],
        ],
    ):
        self.configure(params=func)
        return self
