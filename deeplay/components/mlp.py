from ..core.templates import Layer
from ..core.core import DeeplayModule
from ..core.config import Config, Ref

import torch.nn as nn


class MultiLayerPerceptron(DeeplayModule):
    """Multi-layer perceptron module.

    Also commonly known as a fully-connected neural network, or a dense neural network.

    Configurables
    -------------
    - in_features (int): Number of input features. If None, the input shape is inferred from the first forward pass. (Default: None)
    - hidden_dims (list[int]): Number of hidden units in each layer. (Default: [32, 32])
    - out_features (int): Number of output features. (Default: 1)
    - blocks (template-like): Specification for the blocks of the MLP. (Default: "layer" >> "activation" >> "normalization" >> "dropout")
        - layer (template-like): Specification for the layer of the block. (Default: nn.Linear)
        - activation (template-like): Specification for the activation of the block. (Default: nn.ReLU)
        - normalization (template-like): Specification for the normalization of the block. (Default: nn.Identity)
        - dropout (template-like): Specification for the dropout of the block. (Default: nn.Identity)
    

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
    >>> mlp = MultiLayerPerceptron(28 * 28, [128], 10)
    >>> # Customizing output activation
    >>> mlp = MultiLayerPerceptron(28 * 28, [128], 1, nn.Sigmoid)
    >>> # Using from_config with custom normalization
    >>> mlp = MultiLayerPerceptron.from_config(
    >>>     Config()
    >>>     .in_features(28 * 28)
    >>>     .hidden_dims([128])
    >>>     .out_features(1)
    >>>     .out_activation(nn.Sigmoid)
    >>>     .blocks[0].normalization(nn.BatchNorm1d, num_features=128)
    >>> )

    Return Values
    -------------
    The forward method returns the processed tensor.

    Additional Notes
    ----------------
    The `Config` and `Layer` classes are used for configuring the blocks of the MLP. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

    """

    @staticmethod
    def defaults():
        return (
            Config()
            .in_features(None)
            .depth(Ref("hidden_dims", lambda s: len(s) + 1))
            .blocks(
                Layer("layer")
                >> Layer("normalization")
                >> Layer("activation")
                >> Layer("dropout")
            )
            .blocks.layer(nn.Linear)
            .blocks.activation(nn.ReLU)
            .blocks.normalization(nn.Identity)
            .blocks.dropout(nn.Identity)
            .out_layer(nn.Linear)
            .out_layer.in_features(Ref("hidden_dims", lambda s: s[-1]))
            .out_layer.out_features(Ref("out_features"))
            .out_activation(nn.Identity)
            # If in_features is not specified, we do lazy initialization
        )

    def __init__(
        self,
        in_features: int or None,
        hidden_dims: list[int],
        out_features: int,
        out_activation=None,
        blocks=None,
    ):
        """Multi-layer perceptron module.

        Also commonly known as a fully-connected neural network, or a dense neural network.

        Configurables
        -------------
        - depth (int): Number of layers in the MLP. (Default: 2)
        - blocks (template-like): Specification for the blocks of the MLP. (Default: "layer" >> "activation")
            - layer (template-like): Specification for the layer of the block. (Default: nn.LazyLinear)
            - activation (template-like): Specification for the activation of the block. (Default: nn.ReLU)

        Constraints
        -----------
        - input shape: (batch_size, ch_in)
        - output shape: (batch_size, ch_out)
        - depth >= 1

        Evaluation
        ----------
        >>> for block in mlp.blocks:
        >>>    x = block(x)
        >>> return x

        Examples
        --------
        >>> # Using default values
        >>> mlp = MultiLayerPerceptron()
        >>> # Customizing depth and activation
        >>> mlp = MultiLayerPerceptron(depth=4, blocks=Config().activation(nn.Sigmoid))
        >>> # Using from_config with custom normalization
        >>> mlp = MultiLayerPerceptron.from_config(
        >>>     Config()
        >>>     .blocks(Layer("layer") >> Layer("activation") >> Layer("normalization"))
        >>>     .blocks.normalization(nn.LazyBatchNorm1d)
        >>> )

        Return Values
        -------------
        The forward method returns the processed tensor.

        Additional Notes
        ----------------
        The `Config` and `Layer` classes are used for configuring the blocks of the MLP. For more details refer to [Config Documentation](#) and [Layer Documentation](#).

        """
        super().__init__(
            in_features=in_features,
            hidden_dims=hidden_dims,
            out_features=out_features,
            out_activation=out_activation,
            blocks=blocks,
        )

        self.in_features = self.attr("in_features")
        self.hidden_dims = self.attr("hidden_dims")
        self.out_features = self.attr("out_features")
        self.depth = self.attr("depth")

        blocks = nn.ModuleList()
        for i, out_features in enumerate(self.hidden_dims):
            in_features = self.in_features if i == 0 else self.hidden_dims[i - 1]

            if in_features is None:
                kwargs = {
                    "layer": nn.LazyLinear,
                    "layer.out_features": out_features,
                }
            else:
                kwargs = {
                    "layer.in_features": in_features,
                    "layer.out_features": out_features,
                }

            block = self.new(
                "blocks",
                i,
                extra_kwargs=kwargs,
                now=True,
            )
            blocks.append(block)

        self.blocks = blocks

        # Underscored to represent that it is not a configurable attribute
        self.out_layer = self.new("out_layer")

        self.out_activation = self.new("out_activation")

    def forward(self, x):
        x = nn.Flatten()(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        x = self.out_activation(x)
        return x


class MLPTiny(MultiLayerPerceptron):
    @staticmethod
    def defaults():
        return MultiLayerPerceptron.defaults().hidden_dims([32, 32])


class MLPSmall(MultiLayerPerceptron):
    @staticmethod
    def defaults():
        return MultiLayerPerceptron.defaults().hidden_dims([64, 128, 64])


class MLPMedium(MultiLayerPerceptron):
    @staticmethod
    def defaults():
        return (
            MultiLayerPerceptron.defaults()
            .hidden_dims([128, 256, 512, 1024])
            .blocks.normalization(nn.LazyBatchNorm1d)
        )


class MLPLarge(MultiLayerPerceptron):
    @staticmethod
    def defaults():
        return (
            MultiLayerPerceptron.defaults()
            .hidden_dims([256, 512, 1024, 1024, 1024])
            .blocks.normalization(nn.LazyBatchNorm1d)
        )


class MLPMassive(MultiLayerPerceptron):
    @staticmethod
    def defaults():
        return (
            MultiLayerPerceptron.defaults()
            .hidden_dims([512, 1024, 1024, 1024, 1024, 1024])
            .blocks.normalization(nn.LazyBatchNorm1d)
        )
