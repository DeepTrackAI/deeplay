from typing import Any, overload, Iterator, List, Generic, TypeVar, Union, Tuple, Dict

import torch
from torch import nn

from .module import DeeplayModule
from .decorators import after_init

T = TypeVar("T", bound=nn.Module)


class LayerList(DeeplayModule, nn.ModuleList, Generic[T]):
    def __pre_init__(self, *layers: Union[T, List[T]], _args: Tuple[T, ...] = ()):
        if len(layers) == 1 and isinstance(layers[0], list):
            input_layers: Tuple[T] = layers[0]
        else:
            input_layers: tuple[T] = layers
        layers = tuple(input_layers) + _args
        super().__pre_init__(_args=layers)

    def __init__(self, *layers: T):
        super().__init__()

        while len(self):
            super().pop(0)

        for idx, layer in enumerate(layers):
            super().append(layer)
            if isinstance(layer, DeeplayModule) and not layer._has_built:
                self._give_user_configuration(layer, self._get_abs_string_index(idx))
                layer.__construct__()

    @after_init
    def append(self, module: DeeplayModule) -> "LayerList[T]":
        super(LayerList, self).append(module)
        if isinstance(module, DeeplayModule) and not module._has_built:
            self._give_user_configuration(module, self._get_abs_string_index(-1))
            module.__construct__()
        return self

    @after_init
    def pop(self, key: int = -1) -> T:
        return super().pop(key)

    @after_init
    def insert(self, index: int, module: DeeplayModule) -> "LayerList[T]":
        super().insert(index, module)
        if isinstance(module, DeeplayModule) and not module._has_built:
            self._give_user_configuration(module, self._get_abs_string_index(index))
            module.__construct__()
        return self

    @after_init
    def extend(self, modules: List[DeeplayModule]) -> "LayerList[T]":
        super().extend(modules)
        for idx, module in enumerate(modules):
            if isinstance(module, DeeplayModule) and not module._has_built:
                self._give_user_configuration(
                    module, self._get_abs_string_index(idx + len(self) - len(modules))
                )
                module.__construct__()
        return self

    @after_init
    def remove(self, module: DeeplayModule) -> "LayerList[T]":
        super().remove(module)
        return self

    @overload
    def configure(
        self, *args: Union[int, slice, List[int], slice], **kwargs: Any
    ) -> None:
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

    def set_input_map(self, *args: str, **kwargs: str):
        for layer in self:
            layer.set_input_map(*args, **kwargs)

    def set_output_map(self, *args: str, **kwargs: int):
        for layer in self:
            layer.set_output_map(*args, **kwargs)

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
                if hasattr(layer, name) and isinstance(getattr(layer, name), nn.Module)
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

    def __getitem__(self, index: Union[int, slice]) -> "Union[T, LayerList[T]]":
        return nn.ModuleList.__getitem__(self, index)  # type: ignore


class Sequential(LayerList, Generic[T]):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Parallel(LayerList, Generic[T]):
    _keys: List[Tuple[int, str]]

    def __pre_init__(
        self,
        *layers: Union[T, List[T]],
        _args: Tuple[T, ...] = (),
        **kwargs: Dict[str, T],
    ):
        super().__pre_init__(
            *(layers + tuple(kwargs.values())),
            _args=_args,
        )
        self._keys = [(idx + len(layers), key) for idx, key in enumerate(kwargs)]

    def __init__(self, *layers: T, **kwargs):
        for idx, key in self._keys:
            if isinstance(layers[idx], DeeplayModule):
                layers[idx].set_output_map(key)
            else:
                raise TypeError(
                    f"Keyword argument '{key}' must correspond to a DeeplayModule instance. Received {type(layers[idx].__class__)} instead."
                )
        super().__init__(*layers)

    def forward(self, x):
        if (
            isinstance(x, torch.Tensor)
            or (isinstance(x, tuple) and all(isinstance(_x, torch.Tensor) for _x in x))
        ) and self._keys:
            raise ValueError(
                f"Key arguments {[key for _, key in self._keys]} were provided but input was not a dictionary. Got {type(x)} instead."
            )

        if isinstance(x, dict):
            updates = [layer(x, overwrite_output=False) for layer in self]
            return {**x, **{k: v for update in updates for k, v in update.items()}}

        return [layer(x) for layer in self]
