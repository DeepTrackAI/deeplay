from typing import Any, Callable, Type, overload
from .external import External
from functools import partial

import torch.nn as nn

from ..decorators import after_build


def _create_forward_with_extras(old_forward, extra_arguments: dict[str, str]):
    def forward_with_extras(self, *args, extras={}, **kwargs):
        assert all(
            key not in kwargs for key in extra_arguments.keys()
        ), f"Cannot pass {extra_arguments.keys()} as keyword arguments. Use extras instead."

        missing_extras = set(extra_arguments.values()) - set(extras.keys())
        assert (
            not missing_extras
        ), f"module {self} did not receive the following extras: {missing_extras}"

        my_extras = {key: extras[value] for key, value in extra_arguments.items()}
        return old_forward(self, *args, **my_extras)

    return forward_with_extras


class Layer(External):
    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(classtype, *args, **kwargs)

    @after_build
    def set_extras(layer: nn.Module, *args: str, **kwargs: str):
        extras = {}

        for arg in args:
            extras[arg] = arg

        extras.update(kwargs)

        # monkey patch the forward method to include the extras
        # using type(layer) to get the base implementation of forward.
        # This is necessary so that multiple calls to set_extras don't
        # chain the monkey patching.
        # We use partial to bind the instance to make it a method.
        layer.forward = partial(
            _create_forward_with_extras(type(layer).forward, extras),
            layer,
        )

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, classtype, **kwargs) -> None:
        ...

    configure = External.configure

    def forward(self, x):
        raise RuntimeError(
            "Unexpected call to forward. Did you forget to `create` or `build`?"
        )
