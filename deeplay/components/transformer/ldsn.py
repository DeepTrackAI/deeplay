from typing import List, overload, Optional, Literal, Any, Union, Type, Sequence

from torch import Tensor
import torch.nn as nn

from deeplay import DeeplayModule
from deeplay.blocks.sequential import SequentialBlock


class LayerDropoutSkipNormalization(SequentialBlock):
    layer: DeeplayModule
    dropout: DeeplayModule
    skip: DeeplayModule
    normalization: DeeplayModule

    def __init__(
        self,
        layer: DeeplayModule,
        dropout: DeeplayModule,
        skip: DeeplayModule,
        normalization: DeeplayModule,
        order: List[str] = ["layer", "dropout", "skip", "normalization"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(
            layer=layer,
            dropout=dropout,
            skip=skip,
            normalization=normalization,
            order=order,
            **kwargs,
        )

    def forward(self, x):
        y = x
        for name in self.order:
            if name == "skip":
                y = self.skip(y, x)
            y = getattr(self, name)(y)
        return y

    @overload
    def configure(self, **kwargs: nn.Module) -> None: ...

    @overload
    def configure(
        self,
        order: Optional[List[str]],
        layer: Optional[nn.Module],
        dropout: Optional[nn.Module],
        skip: Optional[nn.Module],
        normalization: Optional[nn.Module],
        **kwargs: nn.Module,
    ) -> None: ...

    @overload
    def configure(self, name: Literal["layer"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["dropout"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["skip"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["normalization"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None: ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
