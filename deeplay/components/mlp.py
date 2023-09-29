from ..templates import Layer
from ..core import DeeplayModule
from ..config import Config, Ref

import torch.nn as nn


class MultiLayerPerceptron(DeeplayModule):
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

    @staticmethod
    def defaults():
        return (
            Config()
            .depth(2)
            .blocks(Layer("layer") >> Layer("activation"))
            .blocks.layer(nn.LazyLinear)
            .blocks.activation(nn.ReLU)
        )

    def __init__(self, depth=2, blocks=None):
        super().__init__(depth=depth, blocks=blocks)

        self.depth = self.attr("depth")
        self.blocks = nn.ModuleList(self.new("blocks", i) for i in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
