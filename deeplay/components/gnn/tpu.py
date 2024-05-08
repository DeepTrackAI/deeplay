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
from deeplay.blocks.sequential import SequentialBlock


class TransformPropagateUpdate(SequentialBlock):
    transform: DeeplayModule
    propagate: DeeplayModule
    update: DeeplayModule
    order: List[str]

    def __init__(
        self,
        transform: DeeplayModule,
        propagate: DeeplayModule,
        update: DeeplayModule,
        order=["transform", "propagate", "update"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(
            transform=transform,
            propagate=propagate,
            update=update,
            order=order,
            **kwargs,
        )

    @overload
    def configure(
        self,
        order: Optional[List[str]] = None,
        transform: Optional[DeeplayModule] = None,
        propagate: Optional[DeeplayModule] = None,
        update: Optional[DeeplayModule] = None,
        **kwargs: DeeplayModule,
    ) -> None: ...

    @overload
    def configure(self, name: Literal["transform"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["propagate"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: Literal["update"], *args, **kwargs) -> None: ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None: ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
