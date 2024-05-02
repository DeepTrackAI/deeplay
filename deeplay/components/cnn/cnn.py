from __future__ import annotations

from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

import torch.nn as nn

from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.external.layer import Layer
from deeplay.list import LayerList
from deeplay.module import DeeplayModule


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
    blocks: LayerList[Conv2dBlock]

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
        elif isinstance(out_activation, nn.Module) and not isinstance(
            out_activation, Layer
        ):
            prev_out_activation = out_activation
            out_activation = Layer(lambda: prev_out_activation)

        self.blocks = LayerList()

        c_out = in_channels

        for i, c_out in enumerate([*self.hidden_channels, out_channels]):
            c_in = self.in_channels if i == 0 else self.hidden_channels[i - 1]

            activation = (
                Layer(nn.ReLU) if i < (len(self.hidden_channels)) else out_activation
            )

            block = Conv2dBlock(
                c_in,
                c_out,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=activation.new(),
            )

            self.blocks.append(block)

        if pool is not None:
            if isinstance(pool, type) and issubclass(pool, nn.Module):
                self.pooled(Layer(pool))
            elif isinstance(pool, nn.Module) and not isinstance(pool, Layer):
                for block in self.blocks[1:]:
                    block.configure(pool=pool, order=["pool"] + block.order)
            else:
                self.pooled(pool)

    def forward(self, x):
        idx = 0
        for block in self.blocks:
            x = block(x)
            idx += 1
        return x

    # Configuration shorthands
    def pooled(
        self,
        layer: Layer = Layer(nn.MaxPool2d, 2),
        before_first: bool = False,
    ):

        for block in self.blocks[1:]:
            block.pooled(layer.new())

        if before_first:
            self.blocks[0].pooled(layer.new())

        return self

    def normalized(
        self,
        normalization: Layer = Layer(nn.BatchNorm2d),
        after_last_layer: bool = True,
        mode: Literal["append", "prepend", "insert"] = "append",
        after=None,
    ):
        for idx in range(len(self.blocks) - 1):
            self.blocks[idx].normalized(normalization, mode=mode, after=after)

        if after_last_layer:
            self.blocks[-1].normalized(normalization, mode=mode, after=after)

        return self

    def strided(
        self,
        stride: int | tuple[int, ...],
        apply_to_first: bool = False,
    ):
        for block in self.blocks[1:]:
            block.strided(stride)

        if apply_to_first:
            self.blocks[0].strided(stride)

        return self

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
        **kwargs: Any,
    ) -> None: ...

    configure = DeeplayModule.configure
