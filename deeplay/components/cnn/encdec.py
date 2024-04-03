from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import (
    DeeplayModule,
    Layer,
    LayerList,
    Sequential,
    PoolLayerActivationNormalization,
    LayerActivationNormalizationUpsample,
)
from deeplay.components.cnn import ConvolutionalNeuralNetwork
import torch.nn as nn
import torch


class ConvolutionalEncoder2d(ConvolutionalNeuralNetwork):
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
    hidden_channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[PoolLayerActivationNormalization]

    @property
    def channel(self) -> Sequence[int]:
        import warnings

        warnings.warn(
            "The `channel` property is deprecated. Use `hidden_channels` instead.",
            DeprecationWarning,
        )
        return self.hidden_channels

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        pool: Optional[Union[Type[nn.Module], nn.Module, None]] = Layer(
            nn.MaxPool2d, kernel_size=2, stride=2
        ),
        postprocess: Layer = Layer(nn.Identity),
        channels: Optional[Sequence[int]] = None,
    ):
        if channels is not None:
            import warnings

            hidden_channels = channels
            warnings.warn(
                "The `channels` parameter is deprecated. Use `hidden_channels` instead.",
                DeprecationWarning,
            )

        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            out_activation=out_activation,
        )

        if pool is None:
            pool = Layer(nn.MaxPool2d, kernel_size=2, stride=2)
        elif isinstance(pool, type) and issubclass(pool, nn.Module):
            pool = Layer(pool, kernel_size=2, stride=2)
        elif isinstance(pool, nn.Module) and not isinstance(pool, Layer):
            pool = Layer(lambda: pool)

        self.pooled(pool)
        self.postprocess = postprocess.new()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.postprocess(x)
        return x

    def strided(
        self,
        stride: int = 2,
        apply_to_first_layer: bool = False,
        apply_to_last_layer: bool = True,
    ):

        if apply_to_first_layer:
            self.blocks[0].layer.configure(stride=stride)

        for block in self.blocks[1:]:
            block.layer.configure(stride=stride)

        if apply_to_last_layer:
            self.blocks[-1].layer.configure(stride=stride)

        self["blocks", :].remove("pool")

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    configure = DeeplayModule.configure


class ConvolutionalDecoder2d(ConvolutionalNeuralNetwork):
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

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        upsample: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        preprocess: Union[Type[nn.Module], nn.Module] = Layer(nn.Identity),
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            out_activation=out_activation,
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.preprocess = preprocess

        for block in self.blocks[:-1]:
            if upsample is None:
                block.append(Layer(nn.Upsample, scale_factor=2), name="upsample")
            elif isinstance(upsample, type) and issubclass(upsample, nn.Module):
                block.append(Layer(upsample), name="upsample")
            else:
                block.append(upsample, name="upsample")

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
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    configure = ConvolutionalNeuralNetwork.configure


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
        bottleneck_channels: Sequence[int] = [],
        decoder_channels: Optional[Sequence[int]] = None,
        out_channels: int = None,
        out_activation: Layer = Layer(nn.Identity),
    ):
        if out_channels is None:
            raise ValueError("The `out_channels` parameter must be specified.")

        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = (
            decoder_channels
            if decoder_channels is not None
            else encoder_channels[1::-1]
        )
        self.hidden_channels = list(self.encoder_channels) + list(self.decoder_channels)
        self.out_channels = out_channels
        self.out_activation = out_activation

        self.encoder = ConvolutionalEncoder2d(
            self.in_channels,
            self.encoder_channels,
            self.encoder_channels[-1],
            out_activation=Layer(nn.ReLU),
        )

        if len(bottleneck_channels) > 0:
            self.bottleneck = ConvolutionalNeuralNetwork(
                self.encoder_channels[-1],
                bottleneck_channels[:-1],
                bottleneck_channels[-1],
                out_activation=Layer(nn.ReLU),
            )
        else:
            self.bottleneck = Layer(nn.Identity)

        self.decoder = ConvolutionalDecoder2d(
            self.encoder_channels[-1],
            self.decoder_channels,
            self.out_channels,
            self.out_activation,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
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
    out_activation: Optional[Type[nn.Module]]
    skip: Optional[Type[nn.Module]]

    def __init__(
        self,
        in_channels: Optional[int],
        channels: Sequence[int],
        out_channels=int,
        out_activation: Optional[Type[nn.Module]] = nn.Identity,
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
        self.decoder.blocks.layer.configure(nn.LazyConv2d)

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
    ) -> None: ...
