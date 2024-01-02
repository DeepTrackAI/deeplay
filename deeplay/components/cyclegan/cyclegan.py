from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import (
    DeeplayModule,
    Layer,
    LayerList,
    Sequential,
    LayerActivation,
    LayerActivationNormalization,
)

import torch
import torch.nn as nn


class CycleGANBlock(LayerActivationNormalization):
    """
    CycleGANBlock is the basic block used in the CycleGAN generator. It consists of either a convolution layer or a transposed convolution layer, an instance normalization layer, and a ReLU activation layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        transposed_conv=False,
        activation=None,
        normalization=None,
        **kwargs,
    ):
        super().__init__(
            layer=Layer(nn.ConvTranspose2d, in_channels, out_channels, **kwargs)
            if transposed_conv
            else Layer(
                nn.Conv2d, in_channels, out_channels, padding_mode="reflect", **kwargs
            ),
            activation=activation or Layer(nn.ReLU),
            normalization=normalization or Layer(nn.InstanceNorm2d, out_channels),
            order=["layer", "normalization", "activation"],
        )


class ResidualBlock(DeeplayModule):
    """
    ResidualBlock for CycleGAN generator. It consists of two Blocks defined above with convolution layers having the same number of input and output channels. The output of the blocks is added to the input of the blocks to form the residual connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.blocks = LayerList()
        self.blocks.append(CycleGANBlock(channels, channels, kernel_size=3, padding=1))
        self.blocks.append(
            CycleGANBlock(
                channels, channels, kernel_size=3, padding=1, activation=nn.Identity()
            )
        )

    def forward(self, x):
        x_input = x
        for block in self.blocks:
            x = block(x)
        return x_input + x


class CycleGANGenerator(DeeplayModule):
    """
    CycleGAN generator.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 9,
    ):
        super().__init__()

        self.blocks = LayerList()

        # Initial convolution block
        self.blocks.append(
            CycleGANBlock(in_channels, 64, kernel_size=7, stride=1, padding=3)
        )

        # Downsampling convolutions
        self.blocks.append(CycleGANBlock(64, 128, kernel_size=3, stride=2, padding=1))
        self.blocks.append(CycleGANBlock(128, 256, kernel_size=3, stride=2, padding=1))

        # Residual blocks
        for _ in range(n_residual_blocks):
            self.blocks.append(ResidualBlock(channels=256))

        # Upsampling convolutions
        self.blocks.append(
            CycleGANBlock(
                256,
                128,
                transposed_conv=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        self.blocks.append(
            CycleGANBlock(
                128,
                64,
                transposed_conv=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )

        # Output layer
        self.blocks.append(
            CycleGANBlock(
                64,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                activation=nn.Tanh(),
                normalization=nn.Identity(),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PatchGANBlock(LayerActivationNormalization):
    """
    PatchGANBlock is the basic block used in the PatchGAN discriminator. It consists of a convolution layer, an instance normalization layer, and a LeakyReLU activation layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=None,
        normalization=None,
        **kwargs,
    ):
        super().__init__(
            layer=Layer(
                nn.Conv2d,
                in_channels,
                out_channels,
                kernel_size=4,
                padding=1,
                bias=True,
                padding_mode="reflect",
                **kwargs,
            ),
            activation=activation or Layer(nn.LeakyReLU, 0.2),
            normalization=normalization or Layer(nn.InstanceNorm2d, out_channels),
            order=["layer", "normalization", "activation"],
        )


class CycleGANDiscriminator(DeeplayModule):
    """
    PatchGAN discriminator. This is inspired from the 70 x 70 PatchGAN discriminator used in the original CycleGAN paper.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.blocks = LayerList()

        conv_layer_dims = [64, 128, 256, 512]

        # Initial convolution block
        self.blocks.append(
            PatchGANBlock(
                in_channels, conv_layer_dims[0], stride=2, normalization=nn.Identity()
            )
        )

        # Convolution blocks
        for i in range(1, len(conv_layer_dims)):
            self.blocks.append(
                PatchGANBlock(
                    conv_layer_dims[i - 1],
                    conv_layer_dims[i],
                    stride=1 if i == len(conv_layer_dims) - 1 else 2,
                )
            )

        # Output layer
        self.blocks.append(
            PatchGANBlock(
                conv_layer_dims[-1],
                1,
                stride=1,
                activation=nn.Sigmoid(),
                normalization=nn.Identity(),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
