from __future__ import annotations
from os import remove
from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union
import warnings

from deeplay import (
    DeeplayModule,
    Layer,
    LayerList,
)
from deeplay.components.cnn import ConvolutionalNeuralNetwork
from deeplay.blocks.conv import Conv2dBlock
from deeplay.ops import Cat
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
    blocks: LayerList[Conv2dBlock]

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
            pool=pool,
        )
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
            self.blocks[0].strided(stride, remove_pool=True)

        for block in self.blocks[1:]:
            block.strided(stride, remove_pool=True)

        if apply_to_last_layer:
            self.blocks[-1].strided(stride, remove_pool=True)

        return self

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
    blocks: LayerList[Conv2dBlock]

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        preprocess: Union[Type[nn.Module], nn.Module] = None,
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
        self.preprocess = preprocess if preprocess is not None else Layer(nn.Identity)

        for block in self.blocks[:-1]:
            block.upsampled()

    def forward(self, x):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        return x

    def upsampled(
        self,
        upsample: Layer = Layer(nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0),
        apply_to_last_layer: bool = False,
        mode="append",
        after=None,
    ):
        for block in self.blocks[:-1]:
            block.upsampled(upsample, mode=mode, after=after)
        if apply_to_last_layer:
            self.blocks[-1].upsampled(upsample, mode=mode, after=after)

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[List[int]] = None,
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

    @property
    def blocks(self) -> LayerList[Layer]:
        """Return the blocks of the encoder and decoder. Equivalent to `.encoder.blocks + .bottleneck.blocks + .decoder.blocks`."""
        if isinstance(self.bottleneck, Layer):
            return self.encoder.blocks + self.decoder.blocks
        return self.encoder.blocks + self.bottleneck.blocks + self.decoder.blocks

    @property
    def normalization(self) -> LayerList[Layer]:
        """Return the normalization layers of the encoder and decoder. Equivalent to `.encoder.normalization + .bottleneck.normalization + .decoder.normalization`."""
        return self.blocks.normalization

    def __init__(
        self,
        in_channels: Optional[int],
        encoder_channels: Sequence[int],
        bottleneck_channels: Optional[Union[Sequence[int], int]] = None,
        decoder_channels: Optional[Sequence[int]] = None,
        out_channels: int = None,
        out_activation: Optional[Layer] = None,
    ):
        if out_channels is None:
            raise ValueError("The `out_channels` parameter must be specified.")

        self.decoder_channels = (
            decoder_channels
            if decoder_channels is not None
            else encoder_channels[::-1][1:]
        )
        if bottleneck_channels is None:
            bottleneck_channels = [encoder_channels[-1]]
        elif isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels]

        super().__init__()

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels

        self.hidden_channels = list(self.encoder_channels) + list(self.decoder_channels)
        self.out_channels = out_channels

        self.encoder = ConvolutionalEncoder2d(
            self.in_channels,
            self.encoder_channels[:-1],
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
            self.bottleneck.blocks[0].pooled()
            self.bottleneck.blocks[-1].upsampled()
        else:
            self.bottleneck = Layer(nn.Identity)

        self.decoder = ConvolutionalDecoder2d(
            (encoder_channels + bottleneck_channels)[-1],
            self.decoder_channels,
            self.out_channels,
            out_activation=(
                Layer(nn.Identity) if out_activation is None else out_activation
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class UNet2d(ConvolutionalEncoderDecoder2d):
    in_channels: Optional[int]
    channels: Sequence[Optional[int]]
    out_channels: Optional[int]
    out_activation: Optional[Type[nn.Module]]
    skip: Optional[Type[nn.Module]]

    def __init__(
        self,
        in_channels: Optional[int],
        encoder_channels: Optional[List[int]] = None,
        bottleneck_channels: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
        out_channels: int = 1,
        out_activation: Union[Type[nn.Module], Layer] = nn.Identity,
        # pool: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        # upsample: Optional[Union[Type[nn.Module], nn.Module, None]] = None,
        skip: DeeplayModule = Cat(1),
        channels: Optional[Sequence[int]] = None,
    ):
        if channels is not None:
            encoder_channels = channels

        out_activation = (
            Layer(out_activation)
            if not isinstance(out_activation, Layer)
            else out_activation
        )
        out_activation = out_activation.new()
        super().__init__(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            out_activation=out_activation,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = skip.new()

        if isinstance(self.skip, Cat):
            for idx, block in enumerate(self.decoder.blocks):
                block.configure(
                    in_channels=block.in_channels
                    + self.encoder.blocks[
                        len(self.encoder.blocks) - idx - 1
                    ].out_channels
                )
        else:
            self.decoder.blocks.layer.configure(nn.LazyConv2d)

    def forward(self, x):
        acts = []
        for block in self.encoder.blocks:
            x = block(x)
            acts.append(x)
        x = self.encoder.postprocess(x)

        x = self.bottleneck(x)

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
