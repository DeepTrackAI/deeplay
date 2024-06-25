from __future__ import annotations
import warnings
from typing import List, Optional, Type, Union, Literal
from typing_extensions import Self
import torch.nn as nn

from deeplay.blocks.base import BaseBlock
from deeplay.external import Layer
from deeplay.module import DeeplayModule
from deeplay.ops.logs import FromLogs
from deeplay.ops.merge import Add, MergeOp
from deeplay.ops.shape import Permute
from deeplay.blocks.base import DeferredConfigurableLayer
from deeplay.shapes import Variable


class Conv2dBlock(BaseBlock):
    """Convolutional block with optional normalization and activation."""

    pool: Union[DeferredConfigurableLayer, nn.Module]

    @property
    def expected_input_shape(self):
        return (self.in_channels, Variable(), Variable())

    def __init__(
        self,
        in_channels: Optional[int],
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=0,
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = DeferredConfigurableLayer(self, "pool", mode="prepend")

        if in_channels is None:
            layer = Layer(
                nn.LazyConv2d,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            layer = Layer(
                nn.Conv2d,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        super().__init__(layer=layer, **kwargs)

    def normalized(
        self,
        normalization: Union[Type[nn.Module], DeeplayModule] = nn.BatchNorm2d,
        mode="append",
        after=None,
    ) -> Self:
        did_replace = mode == "replace" and "normalization" in self.order

        super().normalized(normalization, mode=mode, after=after)

        if did_replace:
            # Assume num_features is already correct
            return self

        self._configure_normalization()

        return self

    def _configure_normalization(self):

        idx = self.order.index("normalization")
        # if layer or blocks before normalization
        if any(name in self.order[:idx] for name in ["layer", "blocks"]):
            channels = self.out_channels
        else:
            channels = self.in_channels

        type: Type[nn.Module] = self.normalization.classtype

        if type == nn.BatchNorm2d:
            self.normalization.configure(num_features=channels)
        elif type == nn.GroupNorm:
            num_groups = self.normalization.kwargs.get("num_groups", 1)
            self.normalization.configure(num_groups=num_groups, num_channels=channels)
        elif type == nn.InstanceNorm2d:
            self.normalization.configure(num_features=channels)
        elif type == nn.LayerNorm:
            self.normalization.configure(normalized_shape=channels)

    def pooled(
        self, pool: Layer = Layer(nn.MaxPool2d, 2, 2), mode="prepend", after=None
    ) -> Self:
        self.set("pool", pool, mode=mode, after=after)
        return self

    def upsampled(
        self,
        upsample: Layer = Layer(nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0),
        mode="append",
        after=None,
    ) -> Self:
        upsample = upsample.new()
        if "in_channels" in upsample.configurables:
            upsample.configure(in_channels=self.out_channels)
        if "out_channels" in upsample.configurables:
            upsample.configure(out_channels=self.out_channels)
        self.set("upsample", upsample, mode=mode, after=after)
        return self

    def transposed(
        self,
        transpose: Layer = Layer(
            nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0
        ),
        mode="prepend",
        after=None,
        remove_upsample=True,
        remove_layer=True,
    ) -> Self:
        self.set("transpose", transpose, mode=mode, after=after)
        if remove_upsample:
            self.remove("upsample", allow_missing=True)
        if remove_layer:
            self.remove("layer", allow_missing=True)
        return self

    def strided(self, stride: int | tuple[int, ...], remove_pool=True) -> Self:
        self.configure(stride=stride)
        self["layer"].configure(nn.Conv2d)  # Might be Identity
        if hasattr(self, "blocks"):
            self.blocks[0].strided(stride, remove_pool=remove_pool)
        elif hasattr(self, "layer"):
            self.layer.configure(stride=stride)
            if remove_pool:
                self.remove("pool", allow_missing=True)

        if hasattr(self, "shortcut_start"):
            if isinstance(self.shortcut_start, Conv2dBlock):
                self.shortcut_start.strided(stride, remove_pool=remove_pool)
            elif isinstance(self.shortcut_start, Layer):
                self.shortcut_start.configure(
                    nn.Conv2d,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )
        return self

    def get_default_normalization(self) -> DeeplayModule:
        return Layer(nn.BatchNorm2d, self.out_channels)

    def get_default_activation(self) -> DeeplayModule:
        return Layer(nn.ReLU)

    def get_default_shortcut(self) -> DeeplayModule:
        if self.in_channels == self.out_channels and (
            self.stride == 1 or self.in_channels is None
        ):
            block = Conv2dBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
            )
            block.configure("layer", nn.Identity)
            return block
        else:
            return Conv2dBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
            )

    def get_default_merge(self) -> MergeOp:
        return Add()

    def call_with_dummy_data(self):
        import torch

        x = torch.randn(2, self.in_channels, 16, 16)
        self(x)

    def _assert_valid_configurable(self, *args):
        return True


@Conv2dBlock.register_style
def residual(
    block: Conv2dBlock,
    order: str = "lanlan|",
    activation: Union[Type[nn.Module], Layer] = nn.ReLU,
    normalization: Union[Type[nn.Module], Layer] = nn.BatchNorm2d,
    dropout: float = 0.1,
):
    """Make a residual block with the given order of layers.

    Parameters
    ----------
    order : str
        The order of layers in the residual block. The shorthand is a string of 'l', 'a', 'n', 'd' and '|'.
        'l' stands for layer, 'a' stands for activation, 'n' stands for normalization, 'd' stands for dropout,
        and '|' stands for the skip connection. The order of the characters in the string determines the order
        of the layers in the residual block. The characters after the '|' determine the order of the layers after
        the skip connection.
    activation : Union[Type[nn.Module], Layer]
        The activation function to use in the residual block.
    normalization : Union[Type[nn.Module], Layer]
        The normalization layer to use in the residual block.
    dropout : float
        The dropout rate to use in the residual block.
    """
    order = order.lower()
    if "|" not in order:
        order += "|"
    # only accept the characters 'l', 'a', 'n', 'd' and '|'
    assert all(
        c in "land|" for c in order
    ), f"The residual order shorthand must only contain the characters 'l', 'a', 'n', 'd' and '|'. Received: {order}"

    after_skip_order = order[order.index("|") + 1 :]
    assert all(
        c in "and" for c in after_skip_order
    ), f"The residual order shorthand must only contain the characters 'a', 'n', 'd' after the skip connection. Received: {order}"

    letter_count_map = {c: after_skip_order.count(c) for c in "lan"}
    assert all(
        count <= 1 for count in letter_count_map.values()
    ), f"The residual order shorthand must contain at most one of each of the characters 'l', 'a', 'n' after the skip connection. Received: {order}"

    block_orders = []
    _order = []
    for c in order[: order.index("|")]:
        if c == "l":
            _name = "layer"
        elif c == "a":
            _name = "activation"
        elif c == "n":
            _name = "normalization"
        elif c == "d":
            _name = "dropout"

        if _name in _order:
            block_orders.append(_order)
            _order = []

        _order.append(_name)

    if _order:
        block_orders.append(_order)
    ksize = block.kernel_size
    if isinstance(ksize, tuple):
        padding = tuple(k // 2 for k in ksize)
    else:
        padding = ksize // 2
    block.configure(padding=padding)
    block.multi(n=len(block_orders))
    block.shortcut()

    for i, block_order in enumerate(block_orders):
        if "activation" in block_order:
            block.blocks[i].activated(activation)
        if "normalization" in block_order:
            block.blocks[i].normalized(normalization)
        if "dropout" in block_order:
            block.blocks[i].set_dropout(dropout)
        block.blocks[i].configure(order=block_order)

    for i, letter in enumerate(after_skip_order):
        if letter == "a":
            block.activated(activation)
        elif letter == "n":
            block.normalized(normalization)
        elif letter == "d":
            block.set_dropout(dropout)

    return block


@Conv2dBlock.register_style
def spatial_self_attention(
    block: Conv2dBlock,
    to_channel_last: bool = False,
    normalization: Union[Layer, Type[nn.Module]] = nn.LayerNorm,
):
    if block.out_channels != block.in_channels:
        warnings.warn(
            "Spatial self-attention should be used with the same number of input and output channels. "
            "Setting the output channels to the input channels."
        )
    block.out_channels = block.in_channels

    block.shortcut()

    from deeplay.ops.attention.self import MultiheadSelfAttention

    block.layer.configure(
        MultiheadSelfAttention,
        features=block.in_channels,
        num_heads=1,
        batch_first=True,
    )
    block.normalized(normalization, mode="insert", after="shortcut_start")

    if to_channel_last:
        block.prepend(Permute(0, 2, 3, 1), name="channel_last")
        block.append(Permute(0, 3, 1, 2), name="channel_first")


@Conv2dBlock.register_style
def spatial_cross_attention(
    block: Conv2dBlock,
    to_channel_last: bool = False,
    normalization: Union[Layer, Type[nn.Module]] = nn.LayerNorm,
    condition_name: str = "condition",
):
    block.out_channels = block.in_channels

    block.residual(hidden_channels=[], merge_after="layer", merge_block=-1)

    from deeplay.ops.attention.cross import MultiheadCrossAttention

    block[..., "layer"].configure(
        MultiheadCrossAttention,
        features=block.in_channels,
        num_heads=1,
        batch_first=True,
        values=0,
        keys=FromLogs(condition_name),
        queries=FromLogs(condition_name),
    )
    normalization = (
        Layer(normalization) if not isinstance(normalization, Layer) else normalization
    )
    block.normalized(normalization)
    block.blocks.configure(order=["normalization", "layer"])

    if to_channel_last:
        block.blocks[0].prepend(Permute(0, 2, 3, 1), name="channel_last")
        block.blocks[-1].append(Permute(0, 3, 1, 2), name="channel_first")


@Conv2dBlock.register_style
def spatial_transformer(
    block: Conv2dBlock,
    to_channel_last: bool = False,
    normalization: Union[Layer, Type[nn.Module]] = nn.LayerNorm,
    condition_name: Optional[str] = "condition",
):
    block.residual(
        hidden_channels=[block.in_channel, block.in_channel],
        merge_after="layer",
        merge_block=-1,
    )

    normalization = (
        Layer(normalization) if not isinstance(normalization, Layer) else normalization
    )

    block.blocks[0].style(
        "spatial_self_attention", normalization=normalization, to_channel_last=False
    )
    if condition_name is not None:
        block.blocks[1].style(
            "spatial_cross_attention",
            normalization=normalization,
            to_channel_last=False,
            condition_name=condition_name,
        )
    else:
        block.blocks[1].style(
            "spatial_self_attention",
            normalization=normalization,
            to_channel_last=False,
        )

    block.blocks[2].residual(
        hidden_channels=[block.out_channels], merge_after="activation", merge_block=-1
    )
    block.blocks[2].blocks[0].layer.configure(
        nn.Linear, block.in_channel, block.out_channel
    )
    block.blocks[2].blocks[0].activation.configure(nn.GELU)
    block.blocks[2].blocks[0].normalized(normalization)
    block.blocks[2].blocks[0].configure(order=["normalization", "layer", "activation"])
    block.blocks[2].blocks[1].layer.configure(
        nn.Linear, block.out_channel, block.out_channel
    )
    block.blocks[2].blocks[1].activation.configure(nn.Identity)

    block.normalized(normalization)
    block.blocks.configure(order=["normalization", "layer"])

    if to_channel_last:
        block.blocks[0].prepend(Permute(0, 2, 3, 1), name="channel_last")
        block.blocks[-1].append(Permute(0, 3, 1, 2), name="channel_first")
