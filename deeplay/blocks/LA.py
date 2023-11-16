from typing import (
    List,
    TypeVar,
    overload,
    Literal,
    Optional,
    Any,
)

import torch.nn as nn


from ..module import DeeplayModule
from .sequential import SequentialBlock


class LayerAct(SequentialBlock):
    layer: DeeplayModule
    act: DeeplayModule
    order: List[str]

    def __init__(
        self,
        layer: DeeplayModule,
        act: DeeplayModule,
        order=["layer", "act"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(layer=layer, act=act, order=order, **kwargs)

    @overload
    def configure(
        self,
        order: Optional[List[str]] = None,
        layer: Optional[DeeplayModule] = None,
        act: Optional[DeeplayModule] = None,
        **kwargs: DeeplayModule,
    ) -> None:
        ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["act"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
