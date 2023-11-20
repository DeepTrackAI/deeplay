from typing import (
    Any,
    overload,
    Iterator,
    List,
    Generic,
    TypeVar,
)

from torch import nn
from .module import DeeplayModule


T = TypeVar("T", bound=nn.Module)


class LayerList(DeeplayModule, nn.ModuleList, Generic[T]):
    def __pre_init__(self, *layers: T | List[T], _args: tuple[T, ...] = ()):
        if len(layers) == 1 and isinstance(layers[0], list):
            input_layers: tuple[T] = layers[0]
        else:
            input_layers: tuple[T] = layers
        layers = input_layers + _args
        super().__pre_init__(_args=layers)

    def __init__(self, *layers: T):
        super().__init__()

        while len(self):
            super().pop(0)

        for idx, layer in enumerate(layers):
            super().append(layer)
            if isinstance(layer, DeeplayModule):
                self._give_user_configuration(layer, self._get_abs_string_index(idx))
                layer.__construct__()

    def append(self, module: DeeplayModule) -> "LayerList[T]":
        if not self._has_built:
            self._args = (*self._args, module)
            self.__construct__()

        else:
            super().append(module)

        return self

    def pop(self, index: int = -1) -> T:
        args = list(self._args)
        args.pop(index)
        self._args = tuple(args)
        self.__construct__()

        return self[index]

    @overload
    def configure(self, *args: int | slice | List[int | slice], **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, name: str, *args: Any, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], int):
                self[args[0]].configure(*args[1:], **kwargs)
            elif isinstance(args[0], slice):
                for layer in self[args[0]]:
                    layer.configure(*args[1:], **kwargs)
            elif isinstance(args[0], list):
                for arg in args[0]:
                    self.configure(arg, *args[1:], **kwargs)
            else:
                for layer in self:
                    layer.configure(*args, **kwargs)

        else:
            for layer in self:
                layer.configure(*args, **kwargs)

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore

    def __getattr__(self, name: str) -> "LayerList[T]":
        try:
            return super().__getattr__(name)
        except AttributeError:
            # check if name is integer string
            if name[0] in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
                # is an invalid attribute name so must be an index
                raise

            submodules = [
                getattr(layer, name)
                for layer in self
                if hasattr(layer, name)
                and isinstance(getattr(layer, name), DeeplayModule)
            ]
            if len(submodules) > 0:
                return LayerList(*submodules)
            else:
                raise

    @overload
    def __getitem__(self, index: int) -> "T":
        ...

    @overload
    def __getitem__(self, index: slice) -> "LayerList[T]":
        ...

    def __getitem__(self, index: int | slice) -> "T | LayerList[T]":
        return super().__getitem__(index)  # type: ignore


class Sequential(LayerList, Generic[T]):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x
