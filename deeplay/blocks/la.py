from typing import (
    List,
    TypeVar,
    overload,
    Literal,
    Optional,
    Any,
)

import torch.nn as nn

from deeplay import DeeplayModule
from .sequential import SequentialBlock


class LayerActivation(SequentialBlock):
    layer: DeeplayModule
    activation: DeeplayModule
    order: List[str]

    def __init__(
        self,
        layer: DeeplayModule,
        activation: DeeplayModule,
        order=["layer", "activation"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(layer=layer, activation=activation, order=order, **kwargs)

    @overload
    def configure(
        self,
        order: Optional[List[str]] = None,
        layer: Optional[DeeplayModule] = None,
        activation: Optional[DeeplayModule] = None,
        **kwargs: DeeplayModule,
    ) -> None: ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["activation"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None: ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
