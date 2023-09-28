from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref
from .encodings import (
    PositionalEncodingLinear1d,
    PositionalEncodingLinear2d,
    PositionalEncodingLinear3d,
)
import torch.nn as nn
import torch


__all__ = [
    "Decoder",
    "ImageToImageDecoder",
    "VectorToImageDecoder",
    "SpatialBroadcastDecoder1d",
    "SpatialBroadcastDecoder2d",
    "SpatialBroadcastDecoder3d",
]


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


class _BaseSpatialBroadcastDecoder(Decoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .depth(4)
            .output_size(None)
            .input(nn.Identity)
            .blocks(Layer("layer") >> Layer("activation"))
            .blocks.activation(nn.ReLU)
        )

    def __init__(
        self, depth=4, output_size=None, input=None, encoding=None, blocks=None
    ):
        DeeplayModule.__init__(
            self,
            depth=depth,
            output_size=output_size,
            input=input,
            encoding=encoding,
            blocks=blocks,
        )

        self.depth = self.attr("depth")
        self.output_size = self.attr("output_size")
        self.input = self.new("input")
        self.encoding = self.new("encoding")
        self.blocks = nn.ModuleList(self.new("blocks", i) for i in range(self.depth))

    def forward(self, x, positions=None):
        """Forward pass.
        Can optionally pass in a grid of coordinates representing the spatial location of each pixel.
        Should be of shape (batch_size, x)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        positions : torch.Tensor
            Grid of xy coordinates to broadcast to each spatial location.
            (batch_size, 2, height, width)
            Default is None.
        """
        x = self.input(x)

        if positions is None:
            output_size = self.output_size
        else:
            output_size = positions.shape[2:]

        x = self.broadcast(x, output_size)

        x = self.encoding(x, positions=positions)
        for block in self.blocks:
            x = block(x)

        return x

    def broadcast(self, x, size):
        """Broadcast a tensor to a given size.
        Expects the tensor to be of shape (batch_size, channels)
        Returns a tensor of shape (batch_size, channels, *size)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        size : tuple
            Size to broadcast to.

        Returns
        -------
        torch.Tensor
            Broadcasted tensor.
        """
        if len(x.shape) != 2:
            raise RuntimeError("The input tensor has to be 2d!")

        batch_size, channels = x.shape
        for _ in size:
            x = x.unsqueeze(-1)
        x = x.expand(batch_size, channels, *size)

        return x


class SpatialBroadcastDecoder1d(_BaseSpatialBroadcastDecoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, _BaseSpatialBroadcastDecoder.defaults())
            .encoding(PositionalEncodingLinear1d)
            .blocks.layer(nn.LazyConv1d, kernel_size=1, padding=0, out_channels=128)
        )


class SpatialBroadcastDecoder2d(_BaseSpatialBroadcastDecoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, _BaseSpatialBroadcastDecoder.defaults())
            .encoding(PositionalEncodingLinear2d)
            .blocks.layer(nn.LazyConv2d, kernel_size=1, padding=0, out_channels=128)
        )


class SpatialBroadcastDecoder3d(_BaseSpatialBroadcastDecoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, _BaseSpatialBroadcastDecoder.defaults())
            .encoding(PositionalEncodingLinear3d)
            .blocks.layer(nn.LazyConv3d, kernel_size=1, padding=0, out_channels=128)
        )
