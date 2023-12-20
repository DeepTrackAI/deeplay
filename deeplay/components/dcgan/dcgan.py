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

from deeplay import LayerList


class ConvBlock(DeeplayModule):
    """
    Convolutional block for DCGAN discriminator. It consists of a 2D Convolutional layer, a Batch Normalization layer and a LeakyReLU activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_norm=True,
    ):
        super().__init__()
        self.blocks = LayerList()
        self.blocks.append(
            LayerActivationNormalization(
                Layer(
                    nn.Conv2d,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=not batch_norm,
                ),
                Layer(nn.BatchNorm2d, out_channels)
                if batch_norm
                else Layer(nn.Identity),
                Layer(nn.LeakyReLU, 0.2),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DcganDiscriminator(DeeplayModule):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) discriminator.

    Parameters
    ----------
    input_channels: int
        Number of input channels
    features_dim: int
        Dimension of the features. The number of features in the four ConvBlocks of the Discriminator can be controlled by this parameter. Convolutional layers = [features_dim, features_dim*2, features_dim*4, features_dim*8].
    class_conditioned_model: bool
        Whether the model is class-conditional
    embedding_dim: int
        Dimension of the label embedding
    num_classes: int
        Number of classes

    Shorthands
    ----------
    - input: `.blocks[0]`
    - hidden: `.blocks[:-1]`
    - output: `.blocks[-1]`
    - layer: `.blocks.layer`
    - activation: `.blocks.activation`

    Constraints
    -----------
    - input shape: (batch_size, ch_in, 64, 64)
    - output shape: (batch_size, 1, 1, 1)

    Examples
    --------
    >>> discriminator = DCGAN_Discriminator(input_channels=1, class_conditioned_model=False)
    >>> discriminator.build()
    >>> batch_size = 16
    >>> input = torch.randn(batch_size, 1, 64, 64)
    >>> output = discriminator(input)

    Return Values
    -------------
    The forward method returns the processed tensor.


    """

    input_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int
    blocks: LayerList[Layer]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    def __init__(
        self,
        input_channels: int = 1,
        features_dim: int = 64,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.features_dim = features_dim
        self.class_conditioned_model = class_conditioned_model

        if class_conditioned_model:
            self.blocks = LayerList()
            self.label_embedding = Sequential(
                Layer(nn.Embedding, num_classes, embedding_dim),
                Layer(nn.Linear, embedding_dim, 64 * 64),
                Layer(nn.LeakyReLU, 0.2),
            )
            self.blocks.append(
                ConvBlock(input_channels + 1, features_dim, 4, 2, 1, batch_norm=False)
            )

        else:
            self.blocks = LayerList()
            self.blocks.append(
                ConvBlock(input_channels, features_dim, 4, 2, 1, batch_norm=False)
            )

        for i in range(3):
            self.blocks.append(
                ConvBlock(
                    features_dim * (2**i),
                    features_dim * (2 ** (i + 1)),
                    4,
                    2,
                    1,
                )
            )

        self.blocks.append(
            LayerActivation(
                Layer(
                    nn.Conv2d, features_dim * 8, 1, kernel_size=4, stride=2, padding=0
                ),
                Layer(nn.Sigmoid),
            )
        )

    def forward(self, x, y=None):
        expected_shape = (x.shape[0], self.input_channels, 64, 64)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input shape is {x.shape}, expected {expected_shape}. DCGAN discriminator expects 64x64 images. Check the input channels in the model initialization."
            )

        if self.class_conditioned_model:
            assert (
                y is not None
            ), "Class label y must be provided for class-conditional discriminator"

            y = self.label_embedding(y)
            y = y.view(-1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, y], dim=1)

        for block in self.blocks:
            x = block(x)

        return x

    def build(self):
        super().build()
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights of the model
        """
        for m in self.modules():
            if isinstance(
                m,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                    nn.BatchNorm2d,
                    nn.Embedding,
                    nn.Linear,
                ),
            ):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    @overload
    def configure(
        self,
        /,
        input_channels: int = 1,
        features_dim: int = 64,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure


class ConvTransposeBlock(DeeplayModule):
    """
    Convolutional transpose block for DCGAN generator. It consists of a 2D Convolutional transpose layer, a Batch Normalization layer and a ReLU activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_norm=True,
    ):
        super().__init__()
        self.blocks = LayerList()
        self.blocks.append(
            LayerActivationNormalization(
                Layer(
                    nn.ConvTranspose2d,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=not batch_norm,
                ),
                Layer(nn.BatchNorm2d, out_channels)
                if batch_norm
                else Layer(nn.Identity),
                Layer(nn.ReLU),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DcganGenerator(DeeplayModule):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) generator.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space
    feature_dims: int
        Dimension of the features. The number of features in the four ConvTransposeBlocks of the Generator can be controlled by this parameter. Convolutional transpose layers = [features_dim*16, features_dim*8, features_dim*4, features_dim*2].
    output_channels: int
        Number of output channels
    class_conditioned_model: bool
        Whether the model is class-conditional
    embedding_dim: int
        Dimension of the label embedding
    num_classes: int
        Number of classes

    Shorthands
    ----------
    - input: `.blocks[0]`
    - hidden: `.blocks[:-1]`
    - output: `.blocks[-1]`
    - layer: `.blocks.layer`
    - activation: `.blocks.activation`

    Constraints
    -----------
    - input shape: (batch_size, latent_dim)
    - output shape: (batch_size, ch_out, 64, 64)

    Examples
    --------
    >>> generator = DCGAN_Generator(latent_dim=100, output_channels=1, class_conditioned_model=False)
    >>> generator.build()
    >>> batch_size = 16
    >>> input = torch.randn([batch_size, 100, 1, 1])
    >>> output = generator(input)

    Return Values
    -------------
    The forward method returns the processed tensor.

    """

    latent_dim: int
    output_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    def __init__(
        self,
        latent_dim: int = 100,
        features_dim: int = 64,
        output_channels: int = 1,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
    ):
        super().__init__()

        self.class_conditioned_model = class_conditioned_model
        self.embedding_dim = embedding_dim

        if class_conditioned_model:
            self.blocks = LayerList()
            self.label_embedding = Layer(nn.Embedding, num_classes, embedding_dim)
            self.blocks.append(
                ConvTransposeBlock(
                    latent_dim + embedding_dim, features_dim * 16, 4, 1, 0
                )
            )
        else:
            self.blocks = LayerList()
            self.blocks.append(
                ConvTransposeBlock(latent_dim, features_dim * 16, 4, 1, 0)
            )

        for i in reversed(range(1, 4)):
            self.blocks.append(
                ConvTransposeBlock(
                    features_dim * (2 ** (i + 1)), features_dim * (2**i), 4, 2, 1
                )
            )

        self.blocks.append(
            LayerActivation(
                Layer(nn.ConvTranspose2d, features_dim * 2, output_channels, 4, 2, 1),
                Layer(nn.Tanh),
            )
        )

    def forward(self, x, y=None):
        if self.class_conditioned_model:
            assert (
                y is not None
            ), "Class label y must be provided for class-conditional generator"

            y = self.label_embedding(y)
            y = y.view(-1, self.embedding_dim, 1, 1)
            x = torch.cat([x, y], dim=1)

        for block in self.blocks:
            x = block(x)

        return x

    def build(self):
        super().build()
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights of the model
        """
        for m in self.modules():
            if isinstance(
                m,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                    nn.BatchNorm2d,
                    nn.Embedding,
                    nn.Linear,
                ),
            ):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    @overload
    def configure(
        self,
        /,
        latent_dim: int = 100,
        features_dim: int = 64,
        output_channels: int = 1,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
