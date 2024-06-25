from typing import (
    overload,
    Literal,
    Optional,
    Any,
)

from deeplay.module import DeeplayModule
from .sequential import SequentialBlock


class LayerSkip(SequentialBlock):
    layer: DeeplayModule
    skip: DeeplayModule

    def __init__(
        self,
        layer: DeeplayModule,
        skip: DeeplayModule,
    ):
        super().__init__(layer=layer, skip=skip)

    def forward(self, x):
        y = self.layer(x)
        y = self.skip(y, x)
        return y

    @overload
    def configure(
        self,
        layer: Optional[DeeplayModule] = None,
        skip: Optional[DeeplayModule] = None,
    ) -> None: ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["skip"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None: ...

    def configure(self, *args, **kwargs):
        super().configure(*args, **kwargs)
