import warnings

from .block import Block

from typing import List, Optional, overload, Any
from ..module import DeeplayModule


class SequentialBlock(Block):
    def __init__(self, order: Optional[List[str]] = None, **kwargs: DeeplayModule):
        super().__init__()

        if order is None:
            order = list(kwargs.keys())

        self.order = []

        for name in order:
            if not name in kwargs:
                warnings.warn(
                    f"Block {self.__class__.__name__} does not have a module called `{name}`. "
                    "You can provide it using `configure({name}=module)` or "
                    "by passing it as a positional argument to the constructor."
                )
            else:
                setattr(self, name, kwargs[name])
                self.order.append(name)

    def configure(self, *args, **kwargs):
        super().configure(*args, **kwargs)

    def forward(self, x):
        for name in self.order:
            x = getattr(self, name)(x)
        return x
