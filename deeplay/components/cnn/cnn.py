from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import DeeplayModule, Layer, LayerList, PoolLayerActivationNormalization

import torch.nn as nn


class ConvolutionalNeuralNetwork(DeeplayModule):
    """Convolutional Neural Network (CNN) module.

    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass
    hidden_channels: list[int]
        Number of hidden units in each layer
    out_channels: int
        Number of output features
    out_activation: template-like
        Specification for the output activation of the MLP. (Default: nn.Identity)
    pool: template-like
        Specification for the pooling of the block. Is not applied to the first block. (Default: nn.Identity)


    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass
    hidden_channels: list[int]
        Number of hidden units in each layer
    out_channels: int
        Number of output features
    out_activation: template-like
        Specification for the output activation of the MLP. (Default: nn.Identity)
    pool: template-like
        Specification for the pooling of the block. Is not applied to the first block. (Default: nn.Identity)


    Configurables
    -------------
    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass.
    - hidden_channels (list[int]): Number of hidden units in each layer.
    - out_channels (int): Number of output features.
    - blocks (template-like): Specification for the blocks of the CNN. (Default: "layer" >> "activation" >> "normalization" >> "dropout")
        - pool (template-like): Specification for the pooling of the block. (Default: nn.Identity)
        - layer (template-like): Specification for the layer of the block. (Default: nn.Linear)
        - activation (template-like): Specification for the activation of the block. (Default: nn.ReLU)
        - normalization (template-like): Specification for the normalization of the block. (Default: nn.Identity)
        - dropout (template-like): Specification for the dropout of the block. (Default: nn.Identity)
    - out_activation (template-like): Specification for the output activation of the MLP. (Default: nn.Identity)

    Constraints
    -----------
    - input shape: (batch_size, ch_in)
    - output shape: (batch_size, ch_out)

    Evaluation
    ----------
    >>> for block in mlp.blocks:
    >>>    x = block(x)
    >>> return x

    Examples
    --------
    >>> # Using default values
    >>> cnn = ConvolutionalNeuralNetwork(3, [32, 64, 128], 1)
    >>> # Customizing output activation
    >>> cnn.output_block.activation(nn.Sigmoid)
    >>> # Changing the kernel size of the first layer
    >>> cnn.input_block.layer.kernel_size(5)


    Return Values
    -------------
    The forward method returns the processed tensor.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the MLP. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

    """

    in_channels: Optional[int]
    hidden_channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[PoolLayerActivationNormalization]

    @property
    def input_block(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden_blocks(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output_block(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
        pool: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in hidden_channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_channels}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.blocks = LayerList()

        c_out = in_channels

        for i, c_out in enumerate([*self.hidden_channels, out_channels]):
            c_in = self.in_channels if i == 0 else self.hidden_channels[i - 1]

            if i == 0:
                pool_layer = Layer(nn.Identity)
            elif pool is None:
                pool_layer = Layer(nn.Identity)
            elif isinstance(pool, type) and issubclass(pool, nn.Module):
                pool_layer = Layer(pool)
            elif isinstance(pool, DeeplayModule):
                pool_layer = pool.new()
            else:
                pool_layer = pool

            layer = (
                Layer(nn.Conv2d, c_in, c_out, 3, 1, 1)
                if c_in
                else Layer(nn.LazyConv2d, c_out, 3, 1, 1)
            )
            activation = (
                Layer(nn.ReLU) if i < len(self.hidden_channels) else out_activation
            )
            normalization = Layer(nn.Identity, num_features=out_channels)

            block = PoolLayerActivationNormalization(
                pool=pool_layer,
                layer=layer,
                activation=activation,
                normalization=normalization,
            )

            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[List[int]] = None,
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
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
