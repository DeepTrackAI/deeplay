from typing import List, Optional, Literal, Any, Sequence, Type, overload

from ... import DeeplayModule, Layer, LayerList, PoolLayerActivationNormalizationBlock

import torch.nn as nn


class ConvolutionalNeuralNetwork(DeeplayModule):
    """Convolutional Neural Network (CNN) module.

    Configurables
    -------------

    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass. (Default: None)
    - hidden_channels (list[int]): Number of hidden units in each layer. (Default: [32, 32])
    - out_channels (int): Number of output features. (Default: 1)
    - blocks (template-like): Specification for the blocks of the MLP. (Default: "layer" >> "activation" >> "normalization" >> "dropout")
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
    blocks: LayerList[PoolLayerActivationNormalizationBlock]

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

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Type[nn.Module] | nn.Module | None = None,
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

        for i, c_out in enumerate(self.hidden_channels):
            c_in = self.in_channels if i == 0 else self.hidden_channels[i - 1]

            self.blocks.append(
                PoolLayerActivationNormalizationBlock(
                    Layer(nn.Identity),
                    Layer(nn.Conv2d, c_in, c_out, 3, 1, 1)
                    if c_in
                    else Layer(nn.LazyConv2d, c_out, 3, 1, 1),
                    Layer(nn.ReLU),
                    # We can give num_features as an argument to nn.Identity
                    # because it is ignored. This means that users do not have
                    # to specify the number of features for nn.Identity.
                    Layer(nn.Identity, num_features=out_channels),
                )
            )

        self.blocks.append(
            PoolLayerActivationNormalizationBlock(
                Layer(nn.Identity),
                Layer(nn.Conv2d, c_out, self.out_channels, 3, 1, 1),
                out_activation,
                Layer(nn.Identity, num_channels=self.out_channels),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: int | None = None,
        hidden_channels: List[int] | None = None,
        out_channels: int | None = None,
        out_activation: Type[nn.Module] | nn.Module | None = None,
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
        index: int | slice | List[int | slice],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
