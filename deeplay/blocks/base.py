from typing import Type, Union
from numpy import short
import torch
import torch.nn as nn

from deeplay.blocks.sequential import SequentialBlock
from deeplay.external.layer import Layer
from deeplay.module import DeeplayModule
from deeplay.ops.merge import Add
from deeplay.list import Sequential
from typing_extensions import Self

from deeplay.ops.merge import MergeOp


class BaseBlock(SequentialBlock):
    def __init__(self, *args, **kwargs):
        super(BaseBlock, self).__init__(*args, **kwargs)

    def multi(self, n=1) -> Self:
        blocks = Sequential(*[self.new(True) for _ in range(n)])

        self.configure(order=["blocks"], blocks=blocks)

        return self

    def shortcut(
        self,
        merge: MergeOp = Add(),
        shortcut: Union[Type[nn.Module], DeeplayModule] = nn.Identity,
    ) -> Self:

        shortcut = Layer(shortcut) if isinstance(shortcut, type) else shortcut
        # print(shortcut.new())
        self.prepend(shortcut, name="shortcut_start")
        self.append(merge.new(), name="shortcut_end")
        return self

    def activated(
        self,
        activation: Union[Type[nn.Module], DeeplayModule] = nn.ReLU,
        mode="append",
        after=None,
    ) -> Self:
        self.set("activation", activation, mode=mode, after=after)
        return self

    def normalized(
        self,
        normalization: Union[Type[nn.Module], DeeplayModule],
        mode="append",
        after=None,
    ) -> Self:
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
        for name in self.order:
            block = getattr(self, name)

            if name == "shortcut_start":
                shortcut = block(x)
            elif name == "shortcut_end":
                x = block(x, shortcut)
            else:
                x = block(x)

        return x

    def _assert_valid_configurable(self, *args):
        return True
