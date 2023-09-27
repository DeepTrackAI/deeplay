from . import EncoderDecoder
from ..encoders import ImageToImageEncoder
from ..decoders import ImageToImageDecoder
from ...templates import Layer
from ...core import DeeplayModule
from ...config import Config, Ref

import torch.nn as nn


def convolutionalEncoderDecoderConfig():
    return (
        Config()
        .merge(None, EncoderDecoder.defaults())
        .depth(4)
        .encoder_blocks(Layer("layer") >> Layer("activation") >> Layer("pool"))
        .decoder_blocks(Layer("layer") >> Layer("activation") >> Layer("upsample"))
        .encoder_blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
        .encoder_blocks.populate("layer.out_channels", lambda i: 8 * 2**i, length=8)
        .encoder_blocks.activation(nn.ReLU)
        .encoder_blocks.pool(nn.MaxPool2d, kernel_size=2)
        .decoder_blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
        .decoder_blocks.populate(
            "layer.out_channels", lambda i: 8 * 2 ** (4 - i), length=8
        )
        .decoder_blocks.activation(nn.ReLU)
        .decoder_blocks.upsample(
            nn.Upsample, scale_factor=2, mode="bilinear", align_corners=True
        )
    )


class ConvolutionalEncoderDecoder(EncoderDecoder):
    @staticmethod
    def defaults():
        return convolutionalEncoderDecoderConfig()
