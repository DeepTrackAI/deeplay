from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import (
    DeeplayModule,
    Layer,
    LayerList,
    Sequential,
    PoolLayerActivationNormalization,
    LayerActivationNormalizationUpsample,
)
import torch.nn as nn
import torch


class ConvolutionalEncoder2d(DeeplayModule):
    """Convolutional Encoder module in 2D.

    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass
    channels: list[int]
        Number of hidden units in the encoder layers
    out_channels: int
        Number of output features
    out_activation: template-like
        Specification for the output activation. (Default: nn.ReLU)
    pool: template-like
        Specification for the pooling of the block. Is not applied to the first block. (Default: nn.MaxPool2d)
    postprocess: postprocessing layer (Default: nn.Identity)

    Configurables
    -------------
    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass.
    - channels: list[int]: Number of hidden units in the encoder layers
    - out_channels (int): Number of output features.
    - out_activation (template-like): Specification for the output activation. (Default: nn.ReLU)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out)

    Evaluation
    ----------
    >>> for block in self.blocks:
    >>>    x = block(x)
    >>> x = self.postprocess(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> enc = ConvolutionalEncoder2d(3, [32, 64], 128)
    >>> # Customizing output activation
    >>> enc.block[-1].activation(nn.Sigmoid)

    Return Values
    -------------
    The forward method returns the processed tensor.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

    """

    in_channels: Optional[int]
    channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[PoolLayerActivationNormalization]

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
    def layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.blocks.layer

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    @property
    def normalization(self) -> LayerList[Layer]:
        """Return the normalizations of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.normalization

    @property
    def pool(self) -> LayerList[Layer]:
        """Return the pooling of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.pool

    def __init__(
        self,
        in_channels: Optional[int],
        channels: Sequence[int],
        out_channels: int,
        out_activation: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        pool: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        postprocess: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in self.channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {self.channels}"
            )

        if out_activation is None:
            out_activation = Layer(nn.ReLU)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.blocks = LayerList()

        for i, c_out in enumerate(self.channels + [self.out_channels]):
            c_in = self.in_channels if i == 0 else self.channels[i - 1]

            if i == 0:
                pool_layer = Layer(nn.Identity)
            else:
                if pool is None:
                    pool_layer = Layer(nn.MaxPool2d, kernel_size=2)
                elif isinstance(pool, type) and issubclass(pool, nn.Module):
                    pool_layer = Layer(pool)
                elif isinstance(pool, DeeplayModule):
                    pool_layer = pool.new()
                else:
                    pool_layer = pool

            if c_in:
                layer = Layer(nn.Conv2d, c_in, c_out, 3, 1, 1)
            else:
                layer = Layer(nn.LazyConv2d, c_out, 3, 1, 1)

            if i == len(self.channels):
                activation = out_activation
            else:
                activation = Layer(nn.ReLU)
            normalization = Layer(nn.Identity, num_features=c_out)

            block = PoolLayerActivationNormalization(
                pool=pool_layer,
                layer=layer,
                activation=activation,
                normalization=normalization,
            )

            self.blocks.append(block)

        self.postprocess = Layer(nn.Identity)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.postprocess(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        pool: Optional[Type[nn.Module]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        postprocess: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]]],
        order: Optional[Sequence[str]] = None,
        pool: Optional[Type[nn.Module]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        postprocess: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure


class ConvolutionalDecoder2d(DeeplayModule):
    """Convolutional Decoder module in 2D.

    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass
    channels: list[int]
        Number of hidden units in the decoder layers
    out_channels: int
        Number of output features
    out_activation: template-like
        Specification for the output activation. (Default: nn.Sigmoid)
    pool: template-like
        Specification for the pooling of the block. Is not applied to the first block. (Default: nn.MaxPool2d)
    preprocess: preprocessing layer (Default: nn.Identity)

    Configurables
    -------------
    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass.
    - channels: list[int]: Number of hidden units in the decoder layers
    - out_channels (int): Number of output features.
    - out_activation (template-like): Specification for the output activation. (Default: nn.ReLU)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out)

    Evaluation
    ----------
    >>>
    >>> x = self.preprocess(x)
    >>> for block in blocks:
    >>>    x = block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> dec = ConvolutionalDecoder2d(128, [128, 64, 32], 1)
    >>> # Customizing output activation
    >>> dec.block[-1].activation(nn.Identity)

    Return Values
    -------------
    The forward method returns the processed tensor.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

    """

    in_channels: Optional[int]
    channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[LayerActivationNormalizationUpsample]

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
    def layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.blocks.layer

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    @property
    def normalization(self) -> LayerList[Layer]:
        """Return the normalizations of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.normalization

    @property
    def upsample(self) -> LayerList[Layer]:
        """Return the upsampling of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.upsample

    def __init__(
        self,
        in_channels: Optional[int],
        channels: Sequence[int],
        out_channels: int,
        out_activation: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        upsample: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        preprocess: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in self.channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {self.channels}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Sigmoid)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.preprocess = Layer(nn.Identity)

        self.blocks = LayerList()

        for i, c_out in enumerate(self.channels + [self.out_channels]):
            c_in = self.in_channels if i == 0 else self.channels[i - 1]

            if i == len(self.channels):
                upsample_layer = Layer(nn.Identity)
            else:
                if upsample is None:
                    upsample_layer = Layer(
                        nn.LazyConvTranspose2d,
                        c_out,
                        kernel_size=2,
                        stride=2,
                    )
                elif isinstance(upsample, type) and issubclass(upsample, nn.Module):
                    upsample_layer = Layer(upsample)
                elif isinstance(upsample, DeeplayModule):
                    upsample_layer = upsample.new()
                else:
                    upsample_layer = upsample

            layer = Layer(nn.LazyConv2d, c_out, 3, 1, 1)
            if i == len(self.channels):
                activation = out_activation
            else:
                activation = Layer(nn.ReLU)
            normalization = Layer(nn.Identity, num_features=c_out)

            block = LayerActivationNormalizationUpsample(
                layer=layer,
                activation=activation,
                normalization=normalization,
                upsample=upsample_layer,
            )

            self.blocks.append(block)

    def forward(self, x):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
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
        upsample: Optional[Type[nn.Module]] = None,
        preprocess: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]]],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        upsample: Optional[Type[nn.Module]] = None,
        preprocess: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure


class ConvolutionalEncoderDecoder2d(DeeplayModule):
    in_channels: Optional[int]
    encoder_channels: Sequence[Optional[int]]
    decoder_channels: Sequence[Optional[int]]
    out_channels: Optional[int]

    @property
    def pool(self) -> LayerList[Layer]:
        """Return the pooling layers of the encoder. Equivalent to `.encoder.pool`."""
        return self.encoder.pool

    @property
    def upsample(self) -> LayerList[Layer]:
        """Return the upsampling layers of the decoder. Equivalent to `.decoder.upsample`."""
        return self.decoder.upsample

    def __init__(
        self,
        in_channels: Optional[int],
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        out_channels=int,
        out_activation: Optional[Type[nn.Module]] = nn.Sigmoid,
        pool: Union[Type[nn.Module], nn.Module, None] = None,
        upsample: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.channels = encoder_channels + decoder_channels
        self.out_channels = out_channels
        self.out_activation = out_activation

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in self.channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {self.channels}"
            )

        self.encoder = ConvolutionalEncoder2d(
            self.in_channels, self.encoder_channels, self.encoder_channels[-1]
        )

        self.decoder = ConvolutionalDecoder2d(
            self.encoder_channels[-1],
            self.decoder_channels,
            self.out_channels,
            self.out_activation,
        )

        self.blocks = self.encoder.blocks + self.decoder.blocks

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Cat(DeeplayModule):
    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)


class UNet2d(ConvolutionalEncoderDecoder2d):
    in_channels: Optional[int]
    channels: Sequence[Optional[int]]
    out_channels: Optional[int]
    skip: Optional[Type[nn.Module]]

    def __init__(
        self,
        in_channels: Optional[int],
        channels: Sequence[int],
        out_channels=int,
        out_activation: Optional[Type[nn.Module]] = nn.ReLU,
        pool: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        upsample: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        skip: Optional[Type[nn.Module]] = Cat(),
    ):
        super().__init__(
            in_channels=in_channels,
            encoder_channels=channels,
            decoder_channels=channels[::-1],
            out_channels=out_channels,
            out_activation=out_activation,
            pool=pool,
            upsample=upsample,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = skip

    def forward(self, x):
        acts = []
        for block in self.encoder.blocks:
            x = block(x)
            acts.append(x)
        x = self.encoder.postprocess(x)
        x = self.decoder.preprocess(x)
        for act, block in zip(acts[::-1], self.decoder.blocks):
            x = self.skip(act, x)
            x = block(x)
        return x

    @overload
    def configure(
        self,
        skip: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...
