from ..core.templates import Layer
from ..core.core import DeeplayModule
from ..core.config import Config, Ref

import torch.nn as nn

__all__ = [
    "ImageToVectorEncoder",
    "ImageToImageEncoder",
    "VolumeToVectorEncoder",
    "VolumeToVolumeEncoder",
]


class _BaseEncoder(DeeplayModule):
    def __init__(
        self, depth=4, input_block=None, encoder_blocks=None, output_block=None
    ):
        super().__init__(
            depth=depth,
            input_block=input_block,
            encoder_blocks=encoder_blocks,
            output_block=output_block,
        )

        self.depth = self.attr("depth")
        self.input_block = self.new("input_block")
        self.encoder_blocks = nn.ModuleList(
            self.new("encoder_blocks", i) for i in range(depth)
        )
        self.output_block = self.new("output_block")

    def forward(self, x):
        x = self.input_block(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.output_block(x)
        return x


class Base2dConvolutionalEncoder(_BaseEncoder):
    """Convolutional Encoder for 2d data.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: 4)
    - input_block (template-like): Transforms the input before the first pooling layer. (Default: "layer" >> "activation")
        - layer: nn.LazyConv2d (Default)
        - activation: nn.ReLU (Default)
    - encoder_blocks (template-like): Constitute the bulk of the computation. (Default: "pool" >> "layer" >> "activation")
        - pool: nn.MaxPool2d (Default)
        - layer: nn.LazyConv2d (Default)
        - activation: nn.ReLU (Default)
    - output_block (template-like): Transforms the downsampled representation into the required shape and dimensionality. (Default: nn.Identity)


    Constraints
    -----------
    - input shape: (batch_size, ch_in, height, width)
    - depth >= 1

    Evaluation
    ----------
    This section represents the internal processing of the input through the blocks of the BaseEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> encoder = Base2dConvolutionalEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder = Base2dConvolutionalEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder = Base2dConvolutionalEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)
    >>> )

    Return Values
    -------------
    The processed tensor is returned after the forward method processes the input tensor through the input, encoder, and output blocks of the BaseEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the BaseEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of encoders, refer to [External Reference](#).

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .depth(4)
            # templates
            .input_block(Layer("layer") >> Layer("activation"))
            .encoder_blocks(Layer("pool") >> Layer("layer") >> Layer("activation"))
            .output_block(nn.Identity)
            # Layer configs. "_" is a wildcard that matches any name (in this case "input_block", "encoder_blocks", and "output_block")
            ._.layer(nn.LazyConv2d, kernel_size=3, padding=1)
            ._.activation(nn.ReLU)
            ._.pool(nn.MaxPool2d, kernel_size=2)
            # populate out_channels for each block
            .input_block.layer.out_channels(16)
            .encoder_blocks.populate(
                "layer.out_channels", lambda i: 32 * 2**i, length=8
            )
        )


class Base3dConvolutionalEncoder(_BaseEncoder):
    """Convolutional Encoder for 3d data.

    This module serves as a base for building 3D convolutional encoder structures and extends the functionality of the BaseEncoder by incorporating 3D convolutional layers.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: 4)
    - input_block (template-like): Transforms the 3D input before the first pooling layer. (Default: "layer" >> "activation")
        - layer: nn.LazyConv3d (Default)
        - activation: nn.ReLU (Default)
    - encoder_blocks (template-like): Constitute the bulk of the 3D computation. (Default: "pool" >> "layer" >> "activation")
        - pool: nn.MaxPool3d (Default)
        - layer: nn.LazyConv3d (Default)
        - activation: nn.ReLU (Default)
    - output_block (template-like): Transforms the downsampled 3D representation into the required shape and dimensionality. (Default: nn.Identity)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, depth, height, width)
    - output shape: (batch_size, ch_out, depth, height, width)
    - depth >= 1

    Evaluation
    ----------
    This section represents the internal processing of the 3D input through the blocks of the Base3dConvolutionalEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> encoder3d = Base3dConvolutionalEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder3d = Base3dConvolutionalEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder3d = Base3dConvolutionalEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)
    >>> )

    Return Values
    -------------
    The processed tensor is returned after the forward method processes the 3D input tensor through the input, encoder, and output blocks of the Base3dConvolutionalEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the Base3dConvolutionalEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of 3D convolutional encoders, refer to [External Reference](#).

    Dependencies
    ------------
    - BaseEncoder: The Base3dConvolutionalEncoder extends the BaseEncoder to incorporate 3D convolutional layers.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .depth(4)
            # templates
            .input_block(Layer("layer") >> Layer("activation"))
            .encoder_blocks(Layer("pool") >> Layer("layer") >> Layer("activation"))
            .output_block(nn.Identity)
            # Layer configs. "_" is a wildcard that matches any name (in this case "input_block", "encoder_blocks", and "output_block")
            ._.layer(nn.LazyConv3d, kernel_size=3, padding=1)
            ._.activation(nn.ReLU)
            ._.pool(nn.MaxPool3d, kernel_size=2)
            # populate out_channels for each block
            .input_block.layer.out_channels(8)
            .encoder_blocks.populate(
                "layer.out_channels", lambda i: 16 * 2**i, length=8
            )
        )


class ImageToVectorEncoder(Base2dConvolutionalEncoder):
    """ImageToVectorEncoder module.

    This module is specialized for converting 2D image data to vector form. It extends the functionality of the Base2dConvolutionalEncoder by incorporating a flattening layer as the output block.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: Inherited from BaseEncoder)
    - input_block (template-like): Transforms the 2D input before the first pooling layer. (Default: Inherited from BaseEncoder)
        - layer: Inherited from BaseEncoder
        - activation: Inherited from BaseEncoder
    - encoder_blocks (template-like): Constitute the bulk of the 2D computation. (Default: Inherited from BaseEncoder)
        - pool: Inherited from BaseEncoder
        - layer: Inherited from BaseEncoder
        - activation: Inherited from BaseEncoder
    - output_block (template-like): Transforms the downsampled 2D representation into vector form by flattening. (Default: nn.Flatten)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, height, width)
    - output shape: (batch_size, ch_out * height * width)
    - Inherits other constraints from BaseEncoder.

    Evaluation
    ----------
    This section represents the internal processing of the 2D input through the blocks of the ImageToVectorEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)  # Flattening occurs here.
    >>> return x

    Examples
    --------
    >>> # Using default values and configurables inherited from BaseEncoder
    >>> encoder2d = ImageToVectorEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder2d = ImageToVectorEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder2d = ImageToVectorEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)  # Explicitly specifying the flattening layer
    >>> )

    Return Values
    -------------
    The processed tensor in vector form is returned after the forward method processes the 2D input tensor through the input, encoder, and output blocks of the ImageToVectorEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the ImageToVectorEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of image to vector encoders, refer to [External Reference](#).

    Dependencies
    ------------
    - Base2dConvolutionalEncoder: The ImageToVectorEncoder extends the Base2dConvolutionalEncoder to incorporate a flattening layer as the output block.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, Base2dConvolutionalEncoder.defaults())
            .output_block(nn.Flatten)
        )


class ImageToImageEncoder(Base2dConvolutionalEncoder):
    """Convolutional Encoder for 2d data.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: 4)
    - input_block (template-like): Transforms the input before the first pooling layer. (Default: "layer" >> "activation")
        - layer: nn.LazyConv2d (Default)
        - activation: nn.ReLU (Default)
    - encoder_blocks (template-like): Constitute the bulk of the computation. (Default: "pool" >> "layer" >> "activation")
        - pool: nn.MaxPool2d (Default)
        - layer: nn.LazyConv2d (Default)
        - activation: nn.ReLU (Default)
    - output_block (template-like): Transforms the downsampled representation into the required shape and dimensionality. (Default: nn.Identity)


    Constraints
    -----------
    - input shape: (batch_size, ch_in, height, width)
    - depth >= 1

    Evaluation
    ----------
    This section represents the internal processing of the input through the blocks of the BaseEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> encoder = Base2dConvolutionalEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder = Base2dConvolutionalEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder = Base2dConvolutionalEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)
    >>> )

    Return Values
    -------------
    The processed tensor is returned after the forward method processes the input tensor through the input, encoder, and output blocks of the BaseEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the BaseEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of encoders, refer to [External Reference](#).

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, Base2dConvolutionalEncoder.defaults())
            .output_block(nn.Identity)
        )


class VolumeToVectorEncoder(Base3dConvolutionalEncoder):
    """Convolutional Encoder for 3d data.

    This module serves as a base for building 3D convolutional encoder structures and extends the functionality of the BaseEncoder by incorporating 3D convolutional layers.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: 4)
    - input_block (template-like): Transforms the 3D input before the first pooling layer. (Default: "layer" >> "activation")
        - layer: nn.LazyConv3d (Default)
        - activation: nn.ReLU (Default)
    - encoder_blocks (template-like): Constitute the bulk of the 3D computation. (Default: "pool" >> "layer" >> "activation")
        - pool: nn.MaxPool3d (Default)
        - layer: nn.LazyConv3d (Default)
        - activation: nn.ReLU (Default)
    - output_block (template-like): Transforms the downsampled 3D representation into the required shape and dimensionality. (Default: nn.Identity)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, depth, height, width)
    - output shape: (batch_size, ch_out, depth, height, width)
    - depth >= 1

    Evaluation
    ----------
    This section represents the internal processing of the 3D input through the blocks of the Base3dConvolutionalEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> encoder3d = Base3dConvolutionalEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder3d = Base3dConvolutionalEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder3d = Base3dConvolutionalEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)
    >>> )

    Return Values
    -------------
    The processed tensor is returned after the forward method processes the 3D input tensor through the input, encoder, and output blocks of the Base3dConvolutionalEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the Base3dConvolutionalEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of 3D convolutional encoders, refer to [External Reference](#).

    Dependencies
    ------------
    - BaseEncoder: The Base3dConvolutionalEncoder extends the BaseEncoder to incorporate 3D convolutional layers.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, Base3dConvolutionalEncoder.defaults())
            .output_block(nn.Flatten)
        )


class VolumeToVolumeEncoder(Base3dConvolutionalEncoder):
    """ImageToVectorEncoder module.

    This module is specialized for converting 2D image data to vector form. It extends the functionality of the Base2dConvolutionalEncoder by incorporating a flattening layer as the output block.

    Configurables
    -------------
    - depth (int): The number of encoder blocks. (Default: Inherited from BaseEncoder)
    - input_block (template-like): Transforms the 2D input before the first pooling layer. (Default: Inherited from BaseEncoder)
        - layer: Inherited from BaseEncoder
        - activation: Inherited from BaseEncoder
    - encoder_blocks (template-like): Constitute the bulk of the 2D computation. (Default: Inherited from BaseEncoder)
        - pool: Inherited from BaseEncoder
        - layer: Inherited from BaseEncoder
        - activation: Inherited from BaseEncoder
    - output_block (template-like): Transforms the downsampled 2D representation into vector form by flattening. (Default: nn.Flatten)

    Constraints
    -----------
    - input shape: (batch_size, ch_in, height, width)
    - output shape: (batch_size, ch_out * height * width)
    - Inherits other constraints from BaseEncoder.

    Evaluation
    ----------
    This section represents the internal processing of the 2D input through the blocks of the ImageToVectorEncoder.
    >>> x = input_block(x)
    >>> for block in encoder_blocks:
    >>>    x = block(x)
    >>> x = output_block(x)  # Flattening occurs here.
    >>> return x

    Examples
    --------
    >>> # Using default values and configurables inherited from BaseEncoder
    >>> encoder2d = ImageToVectorEncoder()  # Uses Config and Layer for default configuration
    >>> # Customizing depth and input block activation
    >>> encoder2d = ImageToVectorEncoder(depth=6, input_block=Config().activation(nn.LeakyReLU))  # Customizing using Config
    >>> # Using from_config with custom output block
    >>> encoder2d = ImageToVectorEncoder.from_config(
    >>>     Config()
    >>>     .output_block(nn.Flatten)  # Explicitly specifying the flattening layer
    >>> )

    Return Values
    -------------
    The processed tensor in vector form is returned after the forward method processes the 2D input tensor through the input, encoder, and output blocks of the ImageToVectorEncoder.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the ImageToVectorEncoder. For more details refer to [Config Documentation](#) and [Layer Documentation](#). For a deeper understanding of image to vector encoders, refer to [External Reference](#).

    Dependencies
    ------------
    - Base2dConvolutionalEncoder: The ImageToVectorEncoder extends the Base2dConvolutionalEncoder to incorporate a flattening layer as the output block.

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, Base3dConvolutionalEncoder.defaults())
            .output_block(nn.Identity)
        )
