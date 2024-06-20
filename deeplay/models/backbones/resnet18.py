import deeplay as dl
import torch.nn as nn

from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.components.cnn.encdec import ConvolutionalEncoder2d

import torchvision.models as models

from deeplay.external.layer import Layer


@Conv2dBlock.register_style
def resnet(block: Conv2dBlock, stride: int = 1):
    """ResNet style block composed of two residual blocks.

    Parameters
    ----------
    stride : int
        Stride of the first block, by default 1
    """
    # 1. create two blocks
    block.multi(2)

    # 2. make the two blocks
    block.blocks[0].style("residual", order="lnaln|a")
    block.blocks[1].style("residual", order="lnaln|a")

    # 3. if stride > 1, stride the first block and add normalization to the shortcut
    if stride > 1:
        block.blocks[0].strided(stride)
        block.blocks[0].shortcut_start.normalized()

    # 4. remove the pooling layer if it exists.
    block[...].isinstance(Conv2dBlock).all.remove("pool", allow_missing=True)


@Conv2dBlock.register_style
def resnet18_input(block: Conv2dBlock):
    block.layer.configure(kernel_size=7, stride=2, padding=3, bias=False)
    block.normalized(mode="insert", after="layer")
    block.activated(Layer(nn.ReLU, inplace=True), mode="insert", after="normalization")
    block.pooled(
        Layer(
            nn.MaxPool2d,
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=False,
            dilation=1,
        ),
        mode="append",
    )


@ConvolutionalEncoder2d.register_style
def resnet18(encoder: ConvolutionalEncoder2d):
    encoder.blocks[0].style("resnet18_input")
    encoder.blocks[1].style("resnet", stride=1)
    encoder["blocks", 2:].hasattr("style").all.style("resnet", stride=2)
    encoder.initialize(dl.initializers.Kaiming(targets=(nn.Conv2d,)))
    encoder.initialize(dl.initializers.Constant(targets=(nn.BatchNorm2d,)))
    encoder.pool = Layer(nn.AdaptiveAvgPool2d, (1, 1))


class BackboneResnet18(ConvolutionalEncoder2d):

    pool: Layer

    def __init__(self, in_channels: int = 3, pool_output: bool = False):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=[64, 64, 128, 256],
            out_channels=512,
        )
        self.pool_output = pool_output
        self.style("resnet18")

    def forward(self, x):
        x = super().forward(x)
        if self.pool_output:
            x = self.pool(x).squeeze()
        return x
