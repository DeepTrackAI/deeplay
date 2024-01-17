from typing import Callable, Dict, List, Tuple, Union, Iterable, Type

from ..external import External

import torch


class Optimizer(External):
    @property
    def kwargs(self):
        kwargs = super().kwargs
        params = kwargs.get("params", None)
        if callable(params):
            kwargs["params"] = params()

        return kwargs

    def __init__(self, classtype: Type[torch.optim.Optimizer], **optimzer_kwargs):
        super().__init__(classtype=classtype, **optimzer_kwargs)

    def params(
        self,
        func: Callable[
            [],
            Union[
                Iterable[torch.nn.Parameter],
                Dict[str, Iterable[torch.nn.Parameter]],
                List[Dict[str, Iterable[torch.nn.Parameter]]],
            ],
        ],
    ):
        self.configure(params=func)
        return self
