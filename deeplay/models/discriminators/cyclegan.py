from typing import List, Any

from deeplay.components import ConvolutionalEncoder2d
from deeplay.external import Layer

import torch.nn as nn

__all__ = ["CycleGANDiscriminator"]


@ConvolutionalEncoder2d.register_style
def cyclegan_discriminator(encoder: ConvolutionalEncoder2d):
    encoder[..., "layer"].configure(kernel_size=4, padding=1)
    encoder["blocks", 1:-1].all.normalized(
        nn.InstanceNorm2d, mode="insert", after="layer"
    )
    encoder["blocks", :].all.remove("pool", allow_missing=True)
    encoder["blocks", :-1].configure("activation", nn.LeakyReLU, negative_slope=0.2)
    encoder["blocks", :-2].configure(stride=2)


class CycleGANDiscriminator(ConvolutionalEncoder2d):
    """
    CycleGAN discriminator.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.

    Examples
    --------
    >>> discriminator = CycleGANDiscriminator(in_channels=3)
    >>> discriminator.build()
    >>> x = torch.randn(1, 3, 256, 256)
    >>> y = discriminator(x)
    >>> y.shape

    Return values
    -------------
    The forward method returns the processed tensor.

    """

    def __init__(self, in_channels: int = 1):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=[64, 128, 256, 512],
            out_channels=1,
            out_activation=Layer(nn.Sigmoid),
        )
        self.style("cyclegan_discriminator")
