# from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

# from ... import (
#     DeeplayModule,
#     Layer,
#     LayerList,
#     LayerActivationNormalization,
#     ConvolutionalEncoderDecoder2d,
# )

# import torch.nn as nn


# class CycleGANBlock(LayerActivationNormalization):
#     """
#     Basic block used in the CycleGAN generator. It consists of a convolution layer (or a transposed convolution layer), an instance normalization layer, and a ReLU activation layer.
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         transposed_conv=False,
#         activation=None,
#         normalization=None,
#         **kwargs,
#     ):
#         super().__init__(
#             layer=(
#                 Layer(nn.ConvTranspose2d, in_channels, out_channels, **kwargs)
#                 if transposed_conv
#                 else Layer(
#                     nn.Conv2d,
#                     in_channels,
#                     out_channels,
#                     padding_mode="reflect",
#                     **kwargs,
#                 )
#             ),
#             activation=activation or Layer(nn.ReLU),
#             normalization=normalization or Layer(nn.InstanceNorm2d, out_channels),
#             order=["layer", "normalization", "activation"],
#         )


# class ResidualBlock(DeeplayModule):
#     """
#     ResidualBlock used in the CycleGAN generator. It combines two CycleGAN blocks defined above through a skip connection.
#     """

#     def __init__(self, channels):
#         super().__init__()
#         self.blocks = LayerList()
#         self.blocks.append(CycleGANBlock(channels, channels, kernel_size=3, padding=1))
#         self.blocks.append(
#             CycleGANBlock(
#                 channels, channels, kernel_size=3, padding=1, activation=nn.Identity()
#             )
#         )

#     def forward(self, x):
#         x_input = x
#         for block in self.blocks:
#             x = block(x)
#         return x_input + x


# class CycleGANResnetGenerator(ConvolutionalEncoderDecoder2d):
#     """
#     CycleGAN generator.

#     Parameters
#     ----------
#     in_channels : int
#         Number of channels in the input image.
#     out_channels : int
#         Number of channels in the output image.
#     n_residual_blocks : int
#         Number of residual blocks in the generator.

#     Shorthands
#     ----------
#     - input: `.blocks[0]`
#     - hidden: `.blocks[:-1]`
#     - output: `.blocks[-1]`
#     - layer: `.blocks.layer`
#     - activation: `.blocks.activation`

#     Examples
#     --------
#     >>> generator = CycleGANGenerator(in_channels=1, out_channels=3)
#     >>> generator.build()
#     >>> x = torch.randn(1, 1, 256, 256)
#     >>> y = generator(x)
#     >>> y.shape

#     Return values
#     -------------
#     The forward method returns the processed tensor.

#     """

#     in_channels: int
#     out_channels: int
#     n_residual_blocks: int
#     blocks: LayerList[Layer]

#     @property
#     def input(self):
#         """Return the input layer of the network. Equivalent to `self.blocks[0]`."""
#         return self.blocks[0]

#     @property
#     def hidden(self):
#         """Return the hidden layers of the network. Equivalent to `self.blocks[:-1]`."""
#         return self.blocks[:-1]

#     @property
#     def output(self):
#         """Return the last layer of the network. Equivalent to `self.blocks[-1]`."""
#         return self.blocks[-1]

#     @property
#     def activation(self) -> LayerList[Layer]:
#         """Return the activations of the network. Equivalent to `.blocks.activation`."""
#         return self.blocks.activation

#     def __init__(
#         self,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         n_residual_blocks: int = 9,
#     ):
#         super().__init__(
#             in_channels=in_channels,
#             encoder_channels=[64, 128, 256],
#             bottleneck_channels=[256] * n_residual_blocks,
#             decoder_channels=[128, 64],
#             out_channels=out_channels,
#         )

#         self.encoder.strided(2)
#         self.encoder.normalized(Layer(nn.InstanceNorm2d))
#         self.encoder.blocks[0].prepend(nn.ReflectionPad2d(3))
#         self.encoder.blocks[0].configure("layer", kernel_size=7, stride=1, padding=0)

#         # Downsampling convolutions
#         self.bottleneck.residual(skip_after="")

#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             self.blocks.append(ResidualBlock(channels=256))

#         # Upsampling convolutions
#         self.blocks.append(
#             CycleGANBlock(
#                 256,
#                 128,
#                 transposed_conv=True,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             )
#         )
#         self.blocks.append(
#             CycleGANBlock(
#                 128,
#                 64,
#                 transposed_conv=True,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             )
#         )

#         # Output layer
#         self.blocks.append(
#             CycleGANBlock(
#                 64,
#                 out_channels,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3,
#                 activation=nn.Tanh(),
#                 normalization=nn.Identity(),
#             )
#         )

#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x

#     @overload
#     def configure(
#         self,
#         /,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         n_residual_blocks: int = 9,
#     ) -> None: ...

#     @overload
#     def configure(
#         self,
#         name: Literal["blocks"],
#         order: Optional[Sequence[str]] = None,
#         layer: Optional[Type[nn.Module]] = None,
#         activation: Optional[Type[nn.Module]] = None,
#         normalization: Optional[Type[nn.Module]] = None,
#         **kwargs: Any,
#     ) -> None: ...

#     configure = DeeplayModule.configure


# class PatchGANBlock(LayerActivationNormalization):
#     """
#     PatchGANBlock is the basic block used in the CycleGAN discriminator. It consists of a convolution layer, an instance normalization layer, and a LeakyReLU activation layer.
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         activation=None,
#         normalization=None,
#         **kwargs,
#     ):
#         super().__init__(
#             layer=Layer(
#                 nn.Conv2d,
#                 in_channels,
#                 out_channels,
#                 kernel_size=4,
#                 padding=1,
#                 bias=True,
#                 padding_mode="reflect",
#                 **kwargs,
#             ),
#             activation=activation or Layer(nn.LeakyReLU, 0.2),
#             normalization=normalization or Layer(nn.InstanceNorm2d, out_channels),
#             order=["layer", "normalization", "activation"],
#         )


# class CycleGANDiscriminator(DeeplayModule):
#     """
#     CycleGAN discriminator.

#     Parameters
#     ----------
#     in_channels : int
#         Number of channels in the input image.

#     Shorthands
#     ----------
#     - input: `.blocks[0]`
#     - hidden: `.blocks[:-1]`
#     - output: `.blocks[-1]`
#     - layer: `.blocks.layer`
#     - activation: `.blocks.activation`

#     Examples
#     --------
#     >>> discriminator = CycleGANDiscriminator(in_channels=3)
#     >>> discriminator.build()
#     >>> x = torch.randn(1, 3, 256, 256)
#     >>> y = discriminator(x)
#     >>> y.shape

#     Return values
#     -------------
#     The forward method returns the processed tensor.

#     """

#     in_channels: int
#     blocks: LayerList[Layer]

#     @property
#     def input(self):
#         """Return the input layer of the network. Equivalent to `self.blocks[0]`."""
#         return self.blocks[0]

#     @property
#     def hidden(self):
#         """Return the hidden layers of the network. Equivalent to `self.blocks[:-1]`."""
#         return self.blocks[:-1]

#     @property
#     def output(self):
#         """Return the last layer of the network. Equivalent to `self.blocks[-1]`."""
#         return self.blocks[-1]

#     @property
#     def activation(self) -> LayerList[Layer]:
#         """Return the activations of the network. Equivalent to `.blocks.activation`."""
#         return self.blocks.activation

#     def __init__(self, in_channels: int = 1):
#         super().__init__()

#         self.blocks = LayerList()

#         conv_layer_dims = [64, 128, 256, 512]

#         # Initial convolution block
#         self.blocks.append(
#             PatchGANBlock(
#                 in_channels, conv_layer_dims[0], stride=2, normalization=nn.Identity()
#             )
#         )

#         # Convolution blocks
#         for i in range(1, len(conv_layer_dims)):
#             self.blocks.append(
#                 PatchGANBlock(
#                     conv_layer_dims[i - 1],
#                     conv_layer_dims[i],
#                     stride=1 if i == len(conv_layer_dims) - 1 else 2,
#                 )
#             )

#         # Output layer
#         self.blocks.append(
#             PatchGANBlock(
#                 conv_layer_dims[-1],
#                 1,
#                 stride=1,
#                 activation=nn.Sigmoid(),
#                 normalization=nn.Identity(),
#             )
#         )

#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x

#     @overload
#     def configure(
#         self,
#         /,
#         in_channels: int = 1,
#     ) -> None: ...

#     @overload
#     def configure(
#         self,
#         name: Literal["blocks"],
#         order: Optional[Sequence[str]] = None,
#         layer: Optional[Type[nn.Module]] = None,
#         activation: Optional[Type[nn.Module]] = None,
#         normalization: Optional[Type[nn.Module]] = None,
#         **kwargs: Any,
#     ) -> None: ...

#     configure = DeeplayModule.configure


# # @encoder.register_style
# # def strided(encoder, stride=2):
# #     encoder[..., "pool"].configure("pool", nn.Identity)
# #     encoder[..., "layer"].configure("stride", stride)
