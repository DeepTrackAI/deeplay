from typing import (
    List,
    TypeVar,
    overload,
    Optional,
    Any,
)

import torch.nn as nn

import warnings

from ..module import DeeplayModule


T = TypeVar("T")


class Block(DeeplayModule):
    @property
    def configurables(self):
        return super().configurables.union(self.kwargs.keys()).union(self.kwargs.get("order", []))

    def __init__(self, **kwargs: DeeplayModule):
        super().__init__()

        for name, val in kwargs.items():
            setattr(self, name, val)

    def set_input_map(self, *args: str, **kwargs: str):
        for name in self.order:
            getattr(self, name).set_input_map(*args, **kwargs)

    def set_output_map(self, *args: str, **kwargs: int):
        for name in self.order:
            getattr(self, name).set_output_map(*args, **kwargs)

    @overload
    def configure(self, **kwargs: DeeplayModule) -> None:
        ...

    @overload
    def configure(self, order: List[str], **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, order=None, **kwargs):
        # We do this to make sure that the order is set before the
        # rest of the kwargs are set. This is important because
        # the order is used to determine allowed kwargs.
        if order is not None:
            super().configure(*args, order=order, **kwargs)
        else:
            super().configure(*args, **kwargs)
