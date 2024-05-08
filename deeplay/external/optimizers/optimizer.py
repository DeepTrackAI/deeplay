from typing import Callable, Dict, List, Tuple, Union, Iterable, Type

from deeplay.external import External
from deeplay.decorators import before_build

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
        try:
            self.configure(params=func(self.root_module))
        except TypeError:
            import warnings

            # deprecation warning
            warnings.warn(
                "Providing a parameter function to the optimizer with no arguments is deprecated. Please use a function with one argument (the root model).",
                DeprecationWarning,
            )
            self.configure(params=func())
        return self
