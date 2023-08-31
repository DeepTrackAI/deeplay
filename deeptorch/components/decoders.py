from ..templates import Layer
from ..core import DeepTorchModule
from ..config import Config, Ref

import torch.nn as nn


class Decoder(DeepTorchModule):
    defaults = (
        Config()
        .depth(4)
        .blocks(Layer("layer") >> Layer("activation"))
        .blocks.activation(nn.ReLU)
    )

    def __init__(self, depth=4, blocks=None):
        super().__init__(depth=depth, blocks=blocks)

        self.depth = self.attr("depth")
        self.blocks = nn.ModuleList(self.new("blocks", i) for i in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvolutionalDecoder(Decoder):
    defaults = (
        Config()
        .merge(None, Decoder.defaults)
        .blocks.populate("layer.out_channels", lambda i: 8 * 2 ** (3 - i), length=8)
        .blocks.layer(nn.LazyConvTranspose2d, kernel_size=2, stride=2)
    )
