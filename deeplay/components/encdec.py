from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from .. import DeeplayModule, Layer, LayerList, PoolLayerActivationNormalization
import torch.nn as nn


class EncoderDecoder(DeeplayModule):
    """Encoder Decoder module.

    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass
    encoder_channels: list[int]
        Number of hidden units in the encoder layers
    decoder_channels: list[int]
        Number of hidden units in the decoder layers
    out_channels: int
        Number of output features
    out_activation: template-like
        Specification for the output activation. (Default: nn.ReLU)
    pool: template-like
        Specification for the pooling of the block. Is not applied to the first block. (Default: nn.MaxPool2d)
    unpool: template-like
        Specification for the unpooling of the block. (Default: nn.ConvTranspose2d)


    Configurables
    -------------
    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass.
    - encoder_channels: list[int]: Number of hidden units in the encoder layers
    - decoder_channels: list[int]: Number of hidden units in the decoder layers
    - out_channels (int): Number of output features.
    - bottleneck_blocks
    - output_blocks
    - out_activation (template-like): Specification for the output activation. (Default: nn.ReLU)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out)

    Evaluation
    ----------
    >>> for block in encdec.blocks:
    >>>    x = block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> encdec = EncoderDecoder(3, [32, 64, 128], [64, 32] 1)
    >>> # Customizing output activation
    >>> encdec.output_block.activation(nn.Sigmoid)



    Return Values
    -------------
    The forward method returns the processed tensor.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

    """

    in_channels: Optional[int]
    encoder_channels: Sequence[Optional[int]]
    decoder_channels: Sequence[Optional[int]]
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
    def encoder_layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.encoder_blocks.layer

    @property
    def bottleneck_layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.bottleneck_blocks

    @property
    def decoder_layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.decoder_blocks.layer

    @property
    def output_layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.output_blocks

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    @property
    def normalization(self) -> LayerList[Layer]:
        """Return the normalizations of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.normalization

    def __init__(
        self,
        in_channels: Optional[int],
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        out_channels: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
        pool: Union[Type[nn.Module], nn.Module, None] = None,
        unpool: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in self.encoder_channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {self.encoder_channels}"
            )
        if any(h <= 0 for h in self.decoder_channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {self.decoder_channels}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.encoder_blocks = LayerList()

        for i, c_out in enumerate(self.encoder_channels + [self.encoder_channels[-1]]):
            c_in = self.in_channels if i == 0 else self.encoder_channels[i - 1]

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

            if i < len(self.encoder_channels):
                if c_in:
                    layer = Layer(nn.Conv2d, c_in, c_out, 3, 1, 1)
                else:
                    layer = Layer(nn.LazyConv2d, c_out, 3, 1, 1)
            else:
                layer = Layer(nn.Identity, num_features=c_out)

            activation = (
                Layer(nn.ReLU)
                if i < len(self.encoder_channels)
                else Layer(nn.Identity, num_features=c_out)
            )
            normalization = Layer(nn.Identity, num_features=c_out)

            block = PoolLayerActivationNormalization(
                pool=pool_layer,
                layer=layer,
                activation=activation,
                normalization=normalization,
            )

            self.encoder_blocks.append(block)

        self.bottleneck_blocks = Layer(nn.Identity, num_features=c_out)

        self.decoder_blocks = LayerList()
        for i, c_out in enumerate(self.decoder_channels + [out_channels]):
            # c_in = c_in if i == 0 else self.decoder_channels[i - 1]

            if unpool is None:
                unpool_layer = Layer(
                    nn.LazyConvTranspose2d,
                    c_out,
                    kernel_size=2,
                    stride=2,
                )
            elif isinstance(unpool, type) and issubclass(unpool, nn.Module):
                unpool_layer = Layer(unpool)
            elif isinstance(unpool, DeeplayModule):
                unpool_layer = unpool.new()
            else:
                unpool_layer = unpool

            layer = Layer(nn.LazyConv2d, c_out, 3, 1, 1)
            activation = (
                Layer(nn.ReLU) if i < len(self.decoder_channels) else out_activation
            )
            normalization = Layer(nn.Identity, num_features=c_out)

            block = PoolLayerActivationNormalization(
                pool=unpool_layer,
                layer=layer,
                activation=activation,
                normalization=normalization,
            )

            self.decoder_blocks.append(block)

        self.output_blocks = Layer(nn.Identity, num_features=c_out)

        self.blocks = (
            self.encoder_blocks
            + LayerList(self.bottleneck_blocks)
            + self.decoder_blocks
            + LayerList(self.output_blocks)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        encoder_channels: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
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
        bottleneck_blocks: Optional[Type[nn.Module]] = None,
        output_blocks: Optional[Type[nn.Module]] = None,
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
        bottleneck_blocks: Optional[Type[nn.Module]] = None,
        output_blocks: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure