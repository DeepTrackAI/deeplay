import torch.nn as nn
import uuid

from ...config import Config, Ref
from ...core import DeeplayModule
from ...templates import Layer, OutputOf, MultiInputLayer
from ..decoders import ImageToImageDecoder
from ..encoders import ImageToImageEncoder
from ..skip import Concatenate
from .fullyconvolutional import ConvolutionalEncoderDecoder


def doubleConvTemplate():
    return (
        Layer("layer") >> Layer("activation") >> Layer("layer") >> Layer("activation")
    )


def originalUnetEncoderBlockConfig():
    return (
        Config()
        .layer(nn.LazyConv2d, kernel_size=3, padding=0)
        .activation(nn.ReLU)
        .pool(nn.MaxPool2d, kernel_size=2)
    )


def originalUnetDecoderBlockConfig():
    return (
        Config()
        .upsample[0](nn.Upsample, scale_factor=2, mode="bilinear", align_corners=True)
        .upsample[1](nn.Identity)
        .combine(Concatenate, mismatch_strategy="crop")
        .layer(nn.LazyConv2d, kernel_size=3, padding=0)
        .activation(nn.ReLU)
    )


def originalUnetConfig():
    return (
        Config()
        .depth(4)
        .input(doubleConvTemplate())
        .input.layer(nn.LazyConv2d, kernel_size=3, padding=0, out_channels=64)
        .input.activation(nn.ReLU)
        .merge("encoder_blocks", originalUnetEncoderBlockConfig())
        .encoder_blocks(Layer("pool") >> doubleConvTemplate())
        .encoder_blocks.populate("layer.out_channels", lambda i: 128 * 2**i, length=8)
        .merge("decoder_blocks", originalUnetDecoderBlockConfig())
        .decoder_blocks(
            (MultiInputLayer("upsample") >> Layer("combine") >> doubleConvTemplate())
        )
        .decoder_blocks.populate(
            "layer.out_channels", lambda i: int(128 * 2 ** (2 - i)), length=8
        )
    )


class UNet(ConvolutionalEncoderDecoder):
    @staticmethod
    def defaults():
        return originalUnetConfig()

    def __init__(self, depth=4, input=None, encoder_blocks=None, decoder_blocks=None):
        DeeplayModule.__init__(self)

        self.depth = self.attr("depth")
        self.input = self.new("input")
        self.encoder_blocks = nn.ModuleList(
            self.new("encoder_blocks", i) for i in range(self.depth)
        )
        self.decoder_blocks = nn.ModuleList(
            self.new("decoder_blocks", i) for i in range(self.depth)
        )

    def forward(self, x):
        x = self.input(x)
        encoder_pyramid = []

        for block in self.encoder_blocks:
            encoder_pyramid.append(x)
            x = block(x)

        reversed_pyramid = reversed(encoder_pyramid)

        for block, skip in zip(self.decoder_blocks, reversed_pyramid):
            x = block(x, skip)

        return x


class UNetTiny(UNet):
    @staticmethod
    def defaults():
        return (
            originalUnetConfig()
            .depth(2)
            .encoder_blocks[0](doubleConvTemplate())
            .decoder_blocks[0](doubleConvTemplate())
        )
