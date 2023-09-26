from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref

import torch.nn as nn

__all__ = [
    "ImageToVectorEncoder",
    "ImageToImageEncoder",
    "VolumeToVectorEncoder",
    "VolumeToVolumeEncoder",
]

class BaseEncoder(DeeplayModule):
    defaults = Config().depth(4).blocks(nn.Identity).output(nn.Identity)

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
        self.output = self.new("output")

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x


class Base2dConvolutionalEncoder(BaseEncoder):
    defaults = (
        Config()
        .merge(None, BaseEncoder.defaults)
        .blocks(Layer("layer") >> Layer("activation") >> Layer("pool"))
        .blocks.populate("layer.out_channels", lambda i: 8 * 2**i, length=8)
        .blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
        .blocks.activation(nn.ReLU)
        .blocks.pool(nn.MaxPool2d, kernel_size=2)
    )


class Base3dConvolutionalEncoder(BaseEncoder):
    defaults = (
        Config()
        .merge(None, BaseEncoder.defaults)
        .blocks(Layer("layer") >> Layer("activation") >> Layer("pool"))
        .blocks.populate("layer.out_channels", lambda i: 8 * 2**i, length=8)
        .blocks.layer(nn.LazyConv3d, kernel_size=3, padding=1)
        .blocks.activation(nn.ReLU)
        .blocks.pool(nn.MaxPool3d, kernel_size=2)
    )


class ImageToVectorEncoder(Base2dConvolutionalEncoder):
    defaults = (
        Config().merge(None, Base2dConvolutionalEncoder.defaults).output(nn.Flatten)
    )


class ImageToImageEncoder(Base2dConvolutionalEncoder):
    defaults = (
        Config().merge(None, Base2dConvolutionalEncoder.defaults).output(nn.Identity)
    )


class VolumeToVectorEncoder(Base3dConvolutionalEncoder):
    defaults = Config().merge(None, Base3dConvolutionalEncoder.defaults)


class VolumeToVolumeEncoder(Base3dConvolutionalEncoder):
    defaults = (
        Config().merge(None, Base3dConvolutionalEncoder.defaults).output(nn.Identity)
    )
