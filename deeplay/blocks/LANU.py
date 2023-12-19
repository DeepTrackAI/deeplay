from typing import (
    List,
    overload,
    Optional,
    Literal,
    Any,
)

import torch.nn as nn

from .sequential import SequentialBlock


class LayerActivationNormalizationUnpool(SequentialBlock):
    layer: nn.Module
    activation: nn.Module
    normalization: nn.Module
    unpool: nn.Module
    order: List[str]

    def __init__(
        self,
        layer: nn.Module,
        activation: nn.Module,
        normalization: nn.Module,
        unpool: nn.Module,
        order: List[str] = ["layer", "activation", "normalization", "unpool"],
        **kwargs: nn.Module,
    ):
        super().__init__(
            layer=layer,
            activation=activation,
            normalization=normalization,
            unpool=unpool,
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
        activation: Optional[nn.Module],
        **kwargs: nn.Module,
    ) -> None:
        ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["activation"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["normalization"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["unpool"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
