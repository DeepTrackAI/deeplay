from typing import Callable, Dict, List, Tuple, Union, Iterable, Type

from ..external import External
from ...decorators import before_build

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

    @before_build
    def params(
        self,
        func: Callable[
            [torch.nn.Module],
            Union[
                Iterable[torch.nn.Parameter],
                Dict[str, Iterable[torch.nn.Parameter]],
                List[Dict[str, Iterable[torch.nn.Parameter]]],
            ],
        ],
    ):
        self.configure(params=func(self.root_module))
        return self
