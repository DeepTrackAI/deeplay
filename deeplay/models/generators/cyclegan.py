from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from deeplay.blocks import Conv2dBlock
from deeplay.components import (
    ConvolutionalEncoderDecoder2d,
    ConvolutionalEncoder2d,
    ConvolutionalDecoder2d,
    ConvolutionalNeuralNetwork,
)
from deeplay.external import Layer
from deeplay.list import LayerList
from deeplay.module import DeeplayModule

import torch.nn as nn

__all__ = ["CycleGANResnetGenerator"]


@ConvolutionalEncoder2d.register_style
def cyclegan_resnet_encoder(encoder: ConvolutionalEncoder2d):
    encoder.strided(2)
    encoder.normalized(Layer(nn.InstanceNorm2d))
    encoder.blocks.configure(order=["layer", "normalization", "activation"])
    encoder.blocks[0].configure(
        "layer", kernel_size=7, stride=1, padding=3, padding_mode="reflect"
    )


@ConvolutionalDecoder2d.register_style
def cyclegan_resnet_decoder(decoder: ConvolutionalDecoder2d):
    decoder["blocks", :-1].all.normalized(
        nn.InstanceNorm2d, mode="insert", after="layer"
    )
    decoder.blocks.configure(order=["layer", "normalization", "activation"])
    decoder.blocks[:-1].configure(
        "layer", nn.ConvTranspose2d, stride=2, output_padding=1
    )
    decoder.blocks[-1].configure(
        "layer", kernel_size=7, stride=1, padding=3, padding_mode="reflect"
    )


@ConvolutionalNeuralNetwork.register_style
def cyclegan_resnet_bottleneck(cnn: ConvolutionalNeuralNetwork, n_blocks=7):
    cnn.configure(hidden_channels=[256] * (n_blocks - 1))
    cnn["blocks", :].all.style(
        "residual", order="lnalna|", normalization=nn.InstanceNorm2d
    )


class CycleGANResnetGenerator(ConvolutionalEncoderDecoder2d):
    """
    CycleGAN generator.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    n_residual_blocks : int
        Number of residual blocks in the generator.

    Shorthands
    ----------
    - input: `.blocks[0]`
    - hidden: `.blocks[:-1]`
    - output: `.blocks[-1]`
    - layer: `.blocks.layer`
    - activation: `.blocks.activation`

    Examples
    --------
    >>> generator = CycleGANResnetGenerator(in_channels=1, out_channels=3)
    >>> generator.build()
    >>> x = torch.randn(1, 1, 256, 256)
    >>> y = generator(x)
    >>> y.shape

    Return values
    -------------
    The forward method returns the processed tensor.

    """

    in_channels: int
    out_channels: int
    n_residual_blocks: int
    blocks: LayerList[Layer]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 9,
    ):
        super().__init__(
            in_channels=in_channels,
            encoder_channels=[64, 128, 256],
            bottleneck_channels=[256] * n_residual_blocks,
            decoder_channels=[128, 64],
            out_channels=out_channels,
            out_activation=Layer(nn.Tanh),
        )

        # Encoder style
        self.encoder.style("cyclegan_resnet_encoder")
        self.bottleneck.style("cyclegan_resnet_bottleneck", n_residual_blocks)
        self.decoder.style("cyclegan_resnet_decoder")

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 9,
    ) -> None: ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None: ...

    configure = DeeplayModule.configure
