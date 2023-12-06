from typing import Any, Type, overload, Dict, Union, List
from .external import External
from functools import partial

import torch.nn as nn

from ..decorators import after_build


def _create_forward_with_input_dict(
    old_forward,
    input_arguments: List[str],
    input_kwargs: Dict[str, str],
    output_arguments: Dict[str, int],
):
    def forward_with_input_dict(self, x):
        x = x.copy()

        outputs = old_forward(
            self,
            *[x.get(arg) for arg in input_arguments],
            **{key: x.get(value) for key, value in input_kwargs.items()},
        )

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        expected_outputs = len(set(output_arguments.values()))
        assert len(outputs) == expected_outputs, (
            f"layer {self} returned {len(outputs)} outputs, "
            f"but it should return {expected_outputs}"
        )

        x.update({key: outputs[value] for key, value in output_arguments.items()})
        return x

    return forward_with_input_dict


class Layer(External):
    def __pre_init__(self, classtype: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(classtype, *args, **kwargs)

    def set_dict_input_mapping(self, *args: str, **kwargs: str):
        self.__dict__.update(
            {
                "input_arguments": args,
                "input_kwargs": kwargs,
                "_dict_input_mapping_called": True,
            }
        )
        self._check_and_execute_dict_mapping()

    def set_dict_output_mapping(self, *args: str, **kwargs: int):
        output_arguments = {arg: i for i, arg in enumerate(args)}
        output_arguments.update(kwargs)

        self.__dict__.update(
            {
                "output_arguments": output_arguments,
                "_dict_output_mapping_called": True,
            }
        )
        self._check_and_execute_dict_mapping()

    def _check_and_execute_dict_mapping(self):
        if getattr(self, "_dict_input_mapping_called", False) and getattr(
            self, "_dict_output_mapping_called", False
        ):
            self.set_dict_mapping(
                self.input_arguments, self.input_kwargs, self.output_arguments
            )

    @after_build
    def set_dict_mapping(
        layer: nn.Module,
        input_arguments: List[str],
        input_kwargs: Dict[str, str],
        output_arguments: Dict[str, int],
    ):
        # monkey patch the forward method to include dict
        # using type(layer) to get the base implementation of forward.
        # This is necessary so that multiple calls to set_input_dict don't
        # chain the monkey patching.
        # We use partial to bind the instance to make it a method.
        layer.forward = partial(
            _create_forward_with_input_dict(
                type(layer).forward, input_arguments, input_kwargs, output_arguments
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
