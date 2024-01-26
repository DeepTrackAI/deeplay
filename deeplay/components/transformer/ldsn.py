from typing import List, overload, Optional, Literal, Any, Union, Type, Sequence

import torch.nn as nn

from deeplay import DeeplayModule


class LayerSkipNormalization(DeeplayModule):
    layer: nn.Module
    skip: nn.Module
    normalization: nn.Module

    def __init__(
        self,
        layer: nn.Module,
        skip: nn.Module,
        normalization: nn.Module,
    ):
        super().__init__()

        self.layer = layer
        self.skip = skip
        self.normalization = normalization

    def forward(self, x):
        x = self.layer(x)
        x = self.skip(x)
        x = self.normalization(x)
        return x

    @overload
    def configure(
        self,
        /,
        layer: Union[Type[nn.Module], nn.Module],
        skip: Union[Type[nn.Module], nn.Module],
        normalization: Union[Type[nn.Module], nn.Module],
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]], None] = None,
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        skip: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
