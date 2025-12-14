from ast import Attribute
from re import A, T
from typing import Any, List, Optional, Type, Union, Tuple
from abc import ABC, abstractmethod
from warnings import warn

import torch
import torch.nn as nn

from deeplay.blocks.sequential import SequentialBlock
from deeplay.external.layer import Layer
from deeplay.module import DeeplayModule
from deeplay.ops.merge import Add
from deeplay.list import Sequential
from typing_extensions import Self

from deeplay.ops.merge import MergeOp


class DeferredConfigurableLayer:

    def __init__(self, parent: "BaseBlock", name: str, mode="append"):
        self.parent = parent
        self.name = name
        self.mode = mode

    def configure(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], type):
            args = Layer(*args, **kwargs)
            self.parent.set(self.name, args, mode=self.mode)
        else:
            self.parent.configure(self.name, *args, **kwargs)

        if self.name == "normalization" and hasattr(
            self.parent, "_configure_normalization"
        ):
            self.parent._configure_normalization()


class BaseBlock(SequentialBlock):

    normalization: Union[DeferredConfigurableLayer, nn.Module]

    def __init__(self, order: Optional[List[str]] = None, **kwargs: DeeplayModule):
        # self.activation = DeferredConfigurableLayer(self, "activation", after="layer")
        self._input_shape = None
        self.normalization = DeferredConfigurableLayer(self, "normalization")
        self._forward_has_been_called_once = False
        self._error_on_failed_forward = False
        super(BaseBlock, self).__init__(order=order, **kwargs)

    def multi(self, n=1) -> Self:

        # Remove configurations before making new blocks
        tags = self.tags
        for key, vlist in self._user_config.items():
            if key[:-1] in tags and vlist:

                if (
                    any(isinstance(v.value, DeeplayModule) for v in vlist)
                    or key[-1] == "order"
                ):
                    vlist.clear()

        def make_new_self():
            args, kwargs = self.get_init_args()
            args = list(args) + list(self._args)
            args = [args.new() if isinstance(args, Layer) else args for args in args]
            for key, value in kwargs.items():
                if isinstance(value, Layer):
                    kwargs[key] = value.new(detach=True)
            return type(self)(*args, **kwargs)

        blocks = Sequential([make_new_self() for _ in range(n)])
        self.configure(order=["blocks"], blocks=blocks)
        if hasattr(self, "in_features") and hasattr(self, "out_features"):
            self["blocks", 1:].configure(in_features=self.out_features)
        elif hasattr(self, "in_channels") and hasattr(self, "out_channels"):
            self["blocks", 1:].configure(in_channels=self.out_channels)
        return self

    def get_default_activation(self) -> DeeplayModule:
        """Returns the default activation function for the block."""
        return Layer(nn.ReLU)

    @abstractmethod
    def get_default_normalization(self) -> DeeplayModule:
        """Returns the default normalization function for the block."""

    def get_default_merge(self) -> MergeOp:
        """Returns the default merge operation for the block."""
        return Add()

    def get_default_shortcut(self) -> DeeplayModule:
        """Returns the default shortcut function for the block."""
        return Layer(nn.Identity)

    @abstractmethod
    def call_with_dummy_data(self):
        """Calls the forward method with dummy data to build the block."""

    def shortcut(
        self,
        merge: Optional[MergeOp] = None,
        shortcut: Union[Type[nn.Module], DeeplayModule, None] = None,
    ) -> Self:
        merge = merge or self.get_default_merge()
        shortcut = shortcut or self.get_default_shortcut()

        shortcut = Layer(shortcut) if isinstance(shortcut, type) else shortcut
        self.prepend(shortcut, name="shortcut_start")
        self.append(merge.new(), name="shortcut_end")
        return self

    def activated(
        self,
        activation: Union[Type[nn.Module], DeeplayModule, None] = None,
        mode="append",
        after=None,
    ) -> Self:
        activation = activation or self.get_default_activation()
        self.set("activation", activation, mode=mode, after=after)
        return self

    def normalized(
        self,
        normalization: Optional[Union[Type[nn.Module], DeeplayModule]] = None,
        mode="append",
        after=None,
    ) -> Self:
        normalization = normalization or self.get_default_normalization()
        self.set("normalization", normalization, mode=mode, after=after)
        return self

    def set(
        self,
        name,
        module: Union[Type[nn.Module], DeeplayModule],
        mode="append",
        after=None,
    ) -> Self:
        if isinstance(module, type):
            module = Layer(module)
        if name in self.order:
            self.configure(**{name: module.new()})
        elif mode == "append":
            self.append(module.new(), name=name)
        elif mode == "prepend":
            self.prepend(module.new(), name=name)
        elif mode == "insert":
            assert (
                after is not None
            ), "Set mode 'insert' requires the 'after' parameter to be set."
            self.insert(module.new(), after=after, name=name)
        elif mode == "replace":
            ...
        return self

    def forward(self, x):
        self._forward_has_been_called_once = True
        for name in self.order:
            block = getattr(self, name)

            if name == "shortcut_start":
                shortcut = block(x)
            elif name == "shortcut_end":
                x = block(x, shortcut)
            else:
                x = block(x)
        return x

    def build(self, *args, **kwargs):
        if args or kwargs:
            return super().build(*args, **kwargs)
        if self._forward_has_been_called_once:
            return super().build()

        try:
            with torch.no_grad():
                self.call_with_dummy_data()
        except RuntimeError as e:
            if self._error_on_failed_forward:
                raise e
            warn(
                f"{self.tags[0]} could not be built with default input. This likely means the block is not configured correctly, "
                "or that it uses lazy initialization. "
                "To suppress this warning, call `model.build(example_input)` with a valid input. "
                "To raise an error instead, call `block.error_on_failed_forward()`. ",
            )
        except TypeError as e:
            if self._error_on_failed_forward:
                raise e
            ...

        return super().build()

    def error_on_failed_forward(self):
        self._error_on_failed_forward = True
        return self

    def _assert_valid_configurable(self, *args):
        return True
