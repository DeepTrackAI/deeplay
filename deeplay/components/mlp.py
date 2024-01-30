from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from .. import DeeplayModule, Layer, LayerList, LayerActivationNormalization

import torch.nn as nn


class MultiLayerPerceptron(DeeplayModule):
    """Multi-layer perceptron module.

    Also commonly known as a fully-connected neural network, or a dense neural network.

    Configurables
    -------------

    - in_features (int): Number of input features. If None, the input shape is inferred in the first forward pass. (Default: None)
    - hidden_features (list[int]): Number of hidden units in each layer.
    - out_features (int): Number of output features. (Default: 1)
    - blocks (template-like): Specification for the blocks of the MLP. (Default: "layer" >> "activation" >> "normalization" >> "dropout")
        - layer (template-like): Specification for the layer of the block. (Default: nn.Linear)
        - activation (template-like): Specification for the activation of the block. (Default: nn.ReLU)
        - normalization (template-like): Specification for the normalization of the block. (Default: nn.Identity)
    - out_activation (template-like): Specification for the output activation of the MLP. (Default: nn.Identity)

    Shorthands
    ----------
    - `input`: Equivalent to `.blocks[0]`.
    - `hidden`: Equivalent to `.blocks[:-1]`.
    - `output`: Equivalent to `.blocks[-1]`.
    - `layer`: Equivalent to `.blocks.layer`.
    - `activation`: Equivalent to `.blocks.activation`.
    - `normalization`: Equivalent to `.blocks.normalization`.

    Evaluation
    ----------
    >>> for block in mlp.blocks:
    >>>    x = block(x)
    >>> return x

    Examples
    --------
    >>> mlp = MultiLayerPerceptron(28 * 28, [128, 128], 10)
    >>> mlp.hidden.normalization.configure(nn.BatchNorm1d)
    >>> mlp.output.activation.configure(nn.Softmax)
    >>> mlp.layer.configure(bias=False)
    >>> mlp.build()


    Return Values
    -------------
    The forward method returns the processed tensor.

    """

    in_features: Optional[int]
    hidden_features: Sequence[Optional[int]]
    out_features: int
    blocks: LayerList[LayerActivationNormalization]

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
        in_features: Optional[int],
        hidden_features: Sequence[Optional[int]],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
        flatten_input: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.flatten_input = flatten_input

        if out_features <= 0:
            raise ValueError(
                f"Number of output features must be positive, got {out_features}"
            )

        if in_features is not None and in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        f_out = in_features

        self.blocks = LayerList()
        for i, f_out in enumerate(self.hidden_features):
            f_in = self.in_features if i == 0 else self.hidden_features[i - 1]

            self.blocks.append(
                LayerActivationNormalization(
                    Layer(nn.Linear, f_in, f_out)
                    if f_in
                    else Layer(nn.LazyLinear, f_out),
                    Layer(nn.ReLU),
                    # We can give num_features as an argument to nn.Identity
                    # because it is ignored. This means that users do not have
                    # to specify the number of features for nn.Identity.
                    Layer(nn.Identity, num_features=f_out),
                )
            )

        self.blocks.append(
            LayerActivationNormalization(
                Layer(nn.Linear, f_out, self.out_features),
                out_activation,
                Layer(nn.Identity, num_features=self.out_features),
            )
        )

    def forward(self, x):
        x = nn.Flatten()(x) if self.flatten_input else x
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[List[int]] = None,
        out_features: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]], None] = None,
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
