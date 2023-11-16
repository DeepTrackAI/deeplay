from typing import (
    List,
    overload,
    Optional,
    Literal,
    Any,
)

import torch.nn as nn


from ..module import DeeplayModule
from .sequential import SequentialBlock


class LayerActNorm(SequentialBlock):
    layer: nn.Module
    act: nn.Module
    norm: nn.Module
    order: List[str]

    def __init__(
        self,
        layer: nn.Module,
        act: nn.Module,
        norm: nn.Module,
        order: List[str] = ["layer", "act", "norm"],
        **kwargs: nn.Module,
    ):
        super().__init__(
            layer=layer,
            act=act,
            norm=norm,
            order=order,
            **kwargs,
        )

    @overload
    def configure(self, **kwargs: nn.Module) -> None:
        ...

    @overload
    def configure(
        self,
        order: Optional[List[str]],
        layer: Optional[nn.Module],
        act: Optional[nn.Module],
        **kwargs: nn.Module,
    ) -> None:
        ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["act"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["norm"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
