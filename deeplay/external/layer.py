from typing import Any, Type, overload, Dict, Union, List, Optional
from .external import External
from functools import partial

import torch.nn as nn

from ..decorators import after_build


def _create_forward_with_input_dict(
    old_forward,
    input_args: List[str],
    input_kwargs: Dict[str, str],
    output_args: Optional[Dict[str, int]],
):
    def forward_with_input_dict(self, x):
        x = x.copy()

        outputs = old_forward(
            self,
            *map(x.get, input_args),
            **{key: x.get(value) for key, value in input_kwargs.items()},
        )

        if not output_args:
            return outputs

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        expected_outputs = len(set(output_args.values()))
        assert len(outputs) == expected_outputs, (
            f"layer {self} returned {len(outputs)} outputs, "
            f"but it should return {expected_outputs}"
        )

        x.update(
            map(
                lambda key, value: (key, outputs[value]),
                *zip(*output_args.items()),
            )
        )
        return x

    return forward_with_input_dict


class Layer(External):
    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(classtype, *args, **kwargs)

    def __init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__init__(classtype, *args, **kwargs)

        # By default, we assume that the layer takes a single input and returns a single output
        self.set_input_map("x")
        self.set_output_map("x")

    def set_input_map(self, *args: str, **kwargs: str):
        self.__dict__.update(
            {"input_args": args, "input_kwargs": kwargs, "_input_map_called": True}
        )
        self._check_and_execute_mapping()

    def set_output_map(self, *args: str, **kwargs: int):
        output_args = {arg: i for i, arg in enumerate(args)}
        output_args.update(kwargs)

        self.__dict__.update({"output_args": output_args, "_output_map_called": True})
        self._check_and_execute_mapping()

    def _check_and_execute_mapping(self):
        if getattr(self, "_input_map_called", False) and getattr(
            self, "_output_map_called", False
        ):
            self._set_mapping(self.input_args, self.input_kwargs, self.output_args)

    @after_build
    def _set_mapping(
        layer: nn.Module,
        input_args: List[str],
        input_kwargs: Dict[str, str],
        output_args: Dict[str, int],
    ):
        # monkey patch the forward method to include dict
        # using type(layer) to get the base implementation of forward.
        # This is necessary so that multiple calls to set_input_dict don't
        # chain the monkey patching.
        # We use partial to bind the instance to make it a method.
        layer.forward = partial(
            _create_forward_with_input_dict(
                type(layer).forward, input_args, input_kwargs, output_args
            ),
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
