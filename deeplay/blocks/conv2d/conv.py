import warnings
from typing import List, Type, Union

import torch.nn as nn

from deeplay.blocks.conv2d.mixin import BaseConvBlockMixin
from deeplay.blocks.sequential import SequentialBlock
from deeplay.external import Layer
from deeplay.list import LayerList
from deeplay.module import DeeplayModule


class Conv2dBlock(SequentialBlock, BaseConvBlockMixin):
    """Convolutional block with optional normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation: Layer = Layer(nn.ReLU),
        mode: str = "base",
        **kwargs,
    ):
        self.mode = mode
        if mode == "base":
            self.class_impl = ConvBlockMixin
        elif mode == "multi":
            self.class_impl = MultiConvBlockMixin
        elif mode == "residual":
            self.class_impl = ResidualConvBlockMixin
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.class_impl.__init__(self, activation=activation, **kwargs)

    def pooled(
        self,
        pool: Union[DeeplayModule, Type[nn.Module]] = Layer(nn.MaxPool2d, 2, 2),
        **kwargs,
    ):
        self.class_impl.pooled(self, pool=pool, **kwargs)

    def normalized(self, normalization: Layer = Layer(nn.BatchNorm2d), **kwargs):
        self.class_impl.normalized(self, normalization=normalization, **kwargs)

    def strided(self, stride: int | tuple[int, ...], remove_pool=True, **kwargs):
        self.class_impl.strided(self, stride=stride, remove_pool=remove_pool, **kwargs)

    def forward(self, x):
        self.class_impl.forward(self, x)


class ConvBlockMixin(BaseConvBlockMixin):

    order: List[str]

    def __init__(self, **kwargs):
        if self.in_channels is not None and self.in_channels > 0:
            layer = Layer(
                nn.Conv2d,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
            )
        else:
            layer = Layer(
                nn.LazyConv2d,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
            )
        kwargs.setdefault("layer", layer)

        self.order = kwargs.pop("order", ["layer", "activation"])

        for name in self.order:
            if not name in kwargs:
                warnings.warn(
                    f"Block {self.__class__.__name__} does not have a module called `{name}`. "
                    "You can provide it using `configure({name}=module)` or "
                    "by passing it as a positional argument to the constructor."
                )
            setattr(self, name, kwargs[name])

    def forward(self, x):
        for name in self.order:
            x = getattr(self, name)(x)
        return x

    def pooled(self, pool: Layer):
        if "pool" in self.order:
            self.configure(pool=pool)
        else:
            self.append(pool.new(), name="pool")
        return self

    def normalized(self, normalization: Layer):
        if "normalization" in self.order:
            self.configure(normalization=normalization)
        else:
            self.append(normalization.new(), name="normalization")
            self.normalization.configure(num_features=self.out_channels)
        return self

    def strided(self, stride: int | tuple[int], remove_pool=True):
        if "strided" in self.order:
            self.configure(stride=stride)
        else:
            self.append(stride, name="stride")

        if remove_pool:
            self.remove("pool", allow_missing=True)

        return self


class MultiConvBlockMixin(BaseConvBlockMixin):

    blocks: LayerList[Conv2dBlock]
    hidden_channels: List[int]

    def __init__(self, hidden_channels: List[int], **kwargs):
        self.hidden_channels = hidden_channels
        self.blocks = LayerList()
        for in_c, out_c in zip(
            [self.in_channels, *hidden_channels], [*hidden_channels, self.out_channels]
        ):
            self.blocks.append(
                Conv2dBlock(
                    in_c,
                    out_c,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    activation=self.activation,
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def pooled(self, pool: Layer, block: int = 0):
        self.blocks[block].pooled(pool)
        return self

    def normalized(self, normalization: Layer):
        for block in self.blocks:
            block.normalized(normalization)
        return self

    def strided(self, stride: int | tuple[int, ...], block: int = 0, remove_pool=True):
        self.blocks[block].strided(stride, remove_pool=remove_pool)
        return self


class ResidualConvBlockMixin(MultiConvBlockMixin):

    merge_after: str
    merge_block: int

    def __init__(
        self,
        hidden_channels: List[int],
        merge_after: str = "activation",
        merge_block: int = -1,
        shortcut: Layer = Layer(nn.Identity),
        **kwargs,
    ):
        super().__init__(hidden_channels, **kwargs)
        self.merge_after = merge_after
        self.merge_block = merge_block
        self.shortcut = shortcut.new()

    def forward(self, x):
        shortcut = x

        merge_block = (
            self.merge_block
            if self.merge_block >= 0
            else len(self.blocks) + self.merge_block
        )

        for block in self.blocks[:merge_block]:
            x = block(x)

        for name in self.blocks[-1].order:
            x = getattr(self.blocks[-1], name)(x)
            if name == self.merge_after:
                x = x + shortcut

        for block in self.blocks[merge_block:]:
            x = block(x)

        return x

    def strided(self, stride: int, block: int = 0, remove_pool=True):
        super().strided(stride, block, remove_pool)
        self.shortcut.configure(
            nn.Conv2d,
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )
        return self

    def pooled(self, pool: Layer, block: int = 0):
        super().pooled(pool, block)
        self.shortcut.configure(
            nn.Conv2d,
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            padding=0,
            stride=pool.stride,
        )
        return self
