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


class BaseDecoder(DeeplayModule):
    """BaseDecoder module.

    This module serves as a foundational base for building decoder structures, providing a flexible and extensible framework for constructing various types of decoders.

    Configurables
    -------------
    - depth (int): The number of decoder blocks in the decoder. (Default: 4)
    - input_block (template-like): Serves as the initial transformation block before the main decoding process. (Default: nn.Identity)
    - decoder_blocks (template-like): Constitute the main computational blocks of the decoder. (Default: nn.Identity)
    - output_block (template-like): Transforms the decoded representation into the required shape and dimensionality. (Default: nn.Identity)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, ...)
    - output shape: (batch_size, ch_out, ...)
    - depth >= 1

    Evaluation
    ----------
    This section represents the internal processing of the input through the blocks of the BaseDecoder.
    >>> x = input_block(x)
    >>> for block in decoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> decoder = BaseDecoder()  # Uses Config for default configuration
    >>> # Customizing depth and input block
    >>> decoder = BaseDecoder(depth=6, input_block=Config().input_block(nn.Linear))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> decoder = BaseDecoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Linear)
    >>> )

    Return Values
    -------------
    The processed tensor is returned after the forward method processes the input tensor through the input, decoder, and output blocks of the BaseDecoder.

    Additional Notes
    ----------------
    The `Config` class is used for configuring the blocks of the BaseDecoder. For more details refer to [Config Documentation](#). For a deeper understanding of decoders, refer to [External Reference](#).

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .depth(4)
            .input_block(nn.Identity)
            .decoder_blocks(nn.Identity)
            .output_block(nn.Identity)
        )

    def __init__(
        self, depth=4, input_block=None, decoder_blocks=None, output_block=None
    ):
        super().__init__(
            depth=depth,
            input_block=input_block,
            decoder_blocks=decoder_blocks,
            output_block=output_block,
        )

        self.depth = self.attr("depth")
        self.input_block = self.new("input_block")
        self.decoder_blocks = nn.ModuleList(
            self.new("decoder_blocks", i) for i in range(self.depth)
        )
        self.output_block = self.new("output_block")

    def forward(self, x):
        x = self.input_block(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.output_block(x)
        return x


class VectorToVectorDecoder(BaseDecoder):
    """VectorToVectorDecoder module.

    This module is specialized for decoding vector data to vector form. It extends the functionality of the BaseDecoder by incorporating specific layer and activation configurations suitable for vector-to-vector decoding.

    Configurables
    -------------
    - depth (int): The number of decoder blocks. (Default: Inherited from BaseDecoder)
    - input_block (template-like): Serves as the initial transformation block before the main decoding process. (Default: Inherited from BaseDecoder)
    - decoder_blocks (template-like): Constitute the main computational blocks for vector-to-vector decoding. (Default: "layer" >> "activation")
        - layer: nn.LazyLinear (Default)
        - activation: nn.ReLU (Default)
        - out_features: 64 (Default)
    - output_block (template-like): Transforms the decoded vector representation into the required shape and dimensionality. (Default: Inherited from BaseDecoder)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out)
    - Inherits other constraints from BaseDecoder.

    Evaluation
    ----------
    This section represents the internal processing of the vector input through the blocks of the VectorToVectorDecoder.
    >>> x = input_block(x)
    >>> for block in decoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values and configurables inherited from BaseDecoder
    >>> vector_decoder = VectorToVectorDecoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and decoder block's out_features
    >>> vector_decoder = VectorToVectorDecoder(depth=6, decoder_blocks=Config().decoder_blocks.out_features(128))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> vector_decoder = VectorToVectorDecoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Linear)
    >>> )

    Return Values
    -------------
    The processed tensor in vector form is returned after the forward method processes the input tensor through the input, decoder, and output blocks of the VectorToVectorDecoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the VectorToVectorDecoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of vector-to-vector decoders, refer to [External Reference](#).

    Dependencies
    ------------
    - BaseDecoder: The VectorToVectorDecoder extends the BaseDecoder to incorporate specific configurations suitable for vector-to-vector decoding.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, BaseDecoder.defaults())
            .decoder_blocks(Layer("layer") >> Layer("activation"))
            .decoder_blocks.layer(nn.LazyLinear)
            .decoder_blocks.activation(nn.ReLU)
            .decoder_blocks.out_features(64)
        )


class ImageToImageDecoder(BaseDecoder):
    """ImageToImageDecoder module.

    This module is specialized for decoding 2D image data to 2D image form. It extends the functionality of the BaseDecoder by incorporating specific configurations suitable for image-to-image decoding, including upsample layers and 2D convolutional layers.

    Configurables
    -------------
    - depth (int): The number of decoder blocks. (Default: Inherited from BaseDecoder)
    - input_block (template-like): Serves as the initial transformation block before the main decoding process. (Default: Inherited from BaseDecoder)
    - decoder_blocks (template-like): Constitute the main computational blocks for image-to-image decoding. (Default: "upsample" >> "layer" >> "activation")
        - upsample: nn.Upsample, scale_factor=2 (Default)
        - layer: nn.LazyConv2d, kernel_size=3, padding=1 (Default)
        - activation: nn.ReLU (Default)
        - out_channels: Populated based on the depth, with a lambda function. (Default: lambda i: 8 * 2 ** (3 - i), length=8)
    - output_block (template-like): Transforms the decoded image representation into the required shape and dimensionality. (Default: Inherited from BaseDecoder)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, height, width)
    - output shape: (batch_size, ch_out, height, width)
    - Inherits other constraints from BaseDecoder.

    Evaluation
    ----------
    This section represents the internal processing of the 2D image input through the blocks of the ImageToImageDecoder.
    >>> x = input_block(x)
    >>> for block in decoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values and configurables inherited from BaseDecoder
    >>> image_decoder = ImageToImageDecoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and decoder block's out_channels
    >>> image_decoder = ImageToImageDecoder(depth=6, decoder_blocks=Config().decoder_blocks.out_channels(16))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> image_decoder = ImageToImageDecoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Conv2d)
    >>> )

    Return Values
    -------------
    The processed tensor in 2D image form is returned after the forward method processes the input tensor through the input, decoder, and output blocks of the ImageToImageDecoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the ImageToImageDecoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of image-to-image decoders, refer to [External Reference](#).

    Dependencies
    ------------
    - BaseDecoder: The ImageToImageDecoder extends the BaseDecoder to incorporate specific configurations suitable for image-to-image decoding.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, BaseDecoder.defaults)
            .blocks(Layer("upsample") >> Layer("layer") >> Layer("activation"))
            .blocks.populate("layer.out_channels", lambda i: 8 * 2 ** (3 - i), length=8)
            .blocks.layer(nn.LazyConv2d, kernel_size=3, padding=1)
            .blocks.activation(nn.ReLU)
            .blocks.upsample(nn.Upsample, scale_factor=2)
        )


class VectorToImageDecoder(BaseDecoder):
    """VectorToImageDecoder module.

    This module is specialized for decoding vector data to 2D image form. It extends the functionality of the BaseDecoder by incorporating specific configurations suitable for vector-to-image decoding, including the SpatialBroadcastDecoder2d as the input block.

    Configurables
    -------------
    - depth (int): The number of decoder blocks. (Default: Inherited from BaseDecoder)
    - output_size (tuple): The size of the output image in the form (height, width) or (height, width). (Default: None)
    - input_block (template-like): Utilizes SpatialBroadcastDecoder2d to broadcast the input vector to the required spatial dimensions. (Default: SpatialBroadcastDecoder2d)
        - See `SpatialBroadcastDecoder2d` for more details on configurables.
    - decoder_blocks (template-like): Constitute the main computational blocks for vector-to-image decoding. (Default: "layer" >> "activation")
        - layer: nn.LazyConv2d, kernel_size=3, padding=1, out_channels=128 (Default)
        - activation: nn.ReLU (Default)
    - output_block (template-like): Transforms the decoded image representation into the required shape and dimensionality. (Default: Inherited from BaseDecoder)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out, height, width)
    - Inherits other constraints from BaseDecoder.

    Evaluation
    ----------
    This section represents the internal processing of the vector input through the blocks of the VectorToImageDecoder.
    >>> x = input_block(x)
    >>> for block in decoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values and configurables inherited from BaseDecoder
    >>> vector_image_decoder = VectorToImageDecoder(output_size=(64, 64))  # Uses Config and Layer for default configuration
    >>> # Customizing depth and decoder block's layer
    >>> vector_image_decoder = VectorToImageDecoder(depth=6, decoder_blocks=Config().decoder_blocks.layer(nn.Conv2d))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> vector_image_decoder = VectorToImageDecoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Conv2d)
    >>>     .output_size((64, 64))
    >>> )

    Return Values
    -------------
    The processed tensor in 2D image form is returned after the forward method processes the input tensor through the input, decoder, and output blocks of the VectorToImageDecoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the VectorToImageDecoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of vector-to-image decoders, refer to [External Reference](#).

    Dependencies
    ------------
    - BaseDecoder: The VectorToImageDecoder extends the BaseDecoder to incorporate specific configurations suitable for vector-to-image decoding.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, BaseDecoder.defaults)
            .output_size(None)
            .input_block(SpatialBroadcastDecoder2d, depth=0)
            # In case user provides output_size as (ch, height, width) instead of (height, width)
            # we need to remove the channel dimension.
            .input_block.output_size(Ref("output_size", lambda size: size[-2:]))
            # No upsample layer required since we are already broadcasting to the required size.
            .encoder_blocks(Layer("layer") >> Layer("activation"))
            .encoder_blocks.layer(
                nn.LazyConv2d, kernel_size=3, padding=1, out_channels=128
            )
            .encoder_blocks.activation(nn.ReLU)
            .output_block(Ref("output_size", lambda size: nn.LazyConv2d(size[0], 1, 1)))
        )

    def __init__(self, output_size=None, **kwargs):
        super().__init__(**kwargs)
        self.output_size = self.attr("output_size")


class _BaseSpatialBroadcastDecoder(DeeplayModule):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, BaseDecoder.defaults())
            .depth(1)
            .output_size(None)
            .input_block(nn.Identity)
            .decoder_blocks(Layer("layer") >> Layer("activation"))
        )

    def __init__(
        self,
        output_size=None,
        depth=1,
        input_block=None,
        encoding=None,
        decoder_blocks=None,
        output_block=None,
    ):
        super().__init__(
            output_size=output_size,
            depth=depth,
            input_block=input_block,
            encoding=encoding,
            decoder_blocks=decoder_blocks,
            output_block=output_block,
        )

        self.depth = self.attr("depth")
        self.output_size = self.attr("output_size")
        self.input_block = self.new("input_block")
        self.encoding = self.new("encoding")
        self.decoder_blocks = nn.ModuleList(
            self.new("decoder_blocks", i) for i in range(self.depth)
        )
        self.output_block = self.new("output_block")

    def forward(self, x):
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
        x = self.input_block(x)

        output_size = self.output_size

        x = self.broadcast(x, output_size)

        x = self.encoding(x)
        for block in self.decoder_blocks:
            x = block(x)

        x = self.output_block(x)

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
            .decoder_blocks.layer(
                nn.LazyConv1d, kernel_size=1, padding=0, out_channels=128
            )
        )


class SpatialBroadcastDecoder2d(_BaseSpatialBroadcastDecoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, _BaseSpatialBroadcastDecoder.defaults())
            .encoding(PositionalEncodingLinear2d)
            .decoder_blocks.layer(
                nn.LazyConv2d, kernel_size=1, padding=0, out_channels=128
            )
        )


class SpatialBroadcastDecoder3d(_BaseSpatialBroadcastDecoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, _BaseSpatialBroadcastDecoder.defaults())
            .encoding(PositionalEncodingLinear3d)
            .decoder_blocks.layer(
                nn.LazyConv3d, kernel_size=1, padding=0, out_channels=128
            )
        )
