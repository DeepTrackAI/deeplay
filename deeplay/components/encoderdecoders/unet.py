
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
    return Layer("layer") >> Layer("activation") >> Layer("layer") >> Layer("activation")

def originalUnetEncoderBlockConfig():
    return (
        Config()
        (Layer("pool") >> doubleConvTemplate())
        .layer(nn.Conv2d, kernel_size=3, padding=0)
        .populate("layer.out_channels", lambda i: 64 * 2**i, length=8)
        .activation(nn.ReLU)
        .pool(nn.MaxPool2d, kernel_size=2)
    )


def originalUnetDecoderBlockConfig():
    return (
        Config()
        (MultiInputLayer("upsample") >> Layer("combine") >> doubleConvTemplate())
        .upsample[0](nn.Upsample, scale_factor=2, mode="bilinear", align_corners=True)
        .upsample[1](nn.Identity)
        .combine(Concatenate, mismatch_strategy="crop")
        .layer(nn.Conv2d, kernel_size=3, padding=0)
        .populate("layer.out_channels", lambda i: 64 * 2**(4 - i), length=8)
        .activation(nn.ReLU)
    )

def originalUnetConfig():
    return (
        Config()
        .depth(4)
        .merge("encoder_blocks", originalUnetEncoderBlockConfig())
        .encoder_blocks[0](doubleConvTemplate())
        .merge("decoder_blocks", originalUnetDecoderBlockConfig())
    )



class UNet(ConvolutionalEncoderDecoder):
    
    @staticmethod
    def defaults():
        return originalUnetConfig()
    
    def __init__(self, encoder_blocks=None, decoder_blocks=None, skips=None):
        super().__init__(encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks, skips=skips)
        self.skips = self.new("skips")

    def forward(self, x):
        encoder_pyramid = []

        for block in self.encoder_blocks:
            x = block(x)
            encoder_pyramid.append(x)

        reversed_pyramid = reversed(encoder_pyramid[:-1])

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