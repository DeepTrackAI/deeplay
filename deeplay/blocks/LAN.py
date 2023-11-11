from typing import (
    List,
    overload,
    Optional,
    Literal,
    Any,
)

import torch.nn as nn


from deeplay import DeeplayModule
from .block import Block


class LayerActivationNormalizationBlock(Block):
    layer: DeeplayModule
    activation: DeeplayModule
    normalization: DeeplayModule
    order: List[str]

    def __init__(
        self,
        layer: DeeplayModule,
        activation: DeeplayModule,
        normalization: DeeplayModule,
        order: List[str] = ["layer", "activation", "normalization"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(
            layer=layer,
            activation=activation,
            normalization=normalization,
            order=order,
            **kwargs,
        )

    @overload
    def configure(self, **kwargs: DeeplayModule) -> None:
        ...

    @overload
    def configure(
        self,
        order: Optional[List[str]],
        layer: Optional[DeeplayModule],
        activation: Optional[DeeplayModule],
        **kwargs: DeeplayModule,
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
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
