from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref

import torch.nn as nn

__all__ = ["Encoder", "ConvolutionalEncoder", "DenseEncoder"]


class Encoder(DeeplayModule):
    defaults = (
        Config().depth(4).blocks(Layer("layer") >> Layer("activation") >> Layer("pool"))
    )

    def __init__(self, depth=4, blocks=None):
        """Encoder module.
        blocks:
            Default: Layer("layer") >> Layer("activation") >> Layer("pool")
            Specification for the blocks of the encoder.
        depth:
            Default: 4
        """

        super().__init__(depth=depth, blocks=blocks)

        self.depth = self.attr("depth")
        self.blocks = nn.ModuleList(self.new("blocks", i) for i in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvolutionalEncoder(Encoder):
    defaults = (
        Config()
        .merge(None, Encoder.defaults)
        .blocks.populate("layer.out_channels", lambda i: 8 * 2**i, length=8)
        .blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
        .blocks.activation(nn.ReLU)
        .blocks.pool(nn.MaxPool2d, kernel_size=2)
    )


class DenseEncoder(Encoder):
    defaults = (
        Config()
        .merge(None, Encoder.defaults)
        .blocks.layer(nn.LazyLinear, out_features=128)
        .blocks.activation(nn.ReLU)
        .blocks.pool(nn.Identity)
    )
