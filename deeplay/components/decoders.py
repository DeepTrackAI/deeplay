from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref

import torch.nn as nn
import torch


__all__ = ["Decoder", "ImageToImageDecoder", "VectorToImageDecoder"]


def _prod(x):
    p = x[0]
    for i in x[1:]:
        p *= i
    return p


class Decoder(DeeplayModule):
    defaults = Config().depth(4).input(nn.Identity).blocks(nn.Identity)

    def __init__(self, depth=4, blocks=None):
        super().__init__(depth=depth, blocks=blocks)

        self.depth = self.attr("depth")
        self.input = self.new("input")
        self.blocks = nn.ModuleList(self.new("blocks", i) for i in range(self.depth))

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return x


class Base2dConvolutionalDecoder(Decoder):
    defaults = (
        Config()
        .merge(None, Decoder.defaults)
        .blocks(Layer("layer") >> Layer("activation") >> Layer("upsample"))
        .blocks.populate("layer.out_channels", lambda i: 8 * 2 ** (3 - i), length=8)
        .blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
        .blocks.activation(nn.ReLU)
        .blocks.upsample(nn.Upsample, scale_factor=2)
    )


class ImageToImageDecoder(Decoder):
    defaults = Config().merge(None, Base2dConvolutionalDecoder.defaults)


class VectorToImageDecoder(ImageToImageDecoder):
    defaults = (
        Config()
        .merge(None, ImageToImageDecoder.defaults)
        .output_size(None)
        .input(Layer("layer") >> Layer("connector"))
        .input.layer(nn.LazyLinear, out_features=Ref("base_size", lambda x: _prod(x)))
        .input.connector(nn.Unflatten, dim=1, unflattened_size=Ref("base_size"))
    )

    def __init__(self, base_size, **kwargs):
        super().__init__(base_size, **kwargs)

        self.base_size = self.attr("base_size")
        self.output_size = self.attr("output_size")

    def forward(self, x):
        x = super().forward(x)
        if self.output_size is not None:
            x = nn.functional.interpolate(x, size=self.output_size)
        return x
