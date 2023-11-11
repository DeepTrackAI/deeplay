from typing import List, Optional, Literal, Any, Sequence

from deeplay import DeeplayModule, Layer, LayerList, LayerActivationNormalizationBlock

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

    in_features: Optional[int]
    hidden_dims: List[int]
    out_features: int
    blocks: LayerList[LayerActivationNormalizationBlock]
    out_layer: LayerActivationNormalizationBlock

    def __init__(
        self,
        in_features: Optional[int],
        hidden_features: Sequence[Optional[int]],
        out_features: int,
        out_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_dims = hidden_features
        self.out_features = out_features

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, nn.Module):
            out_activation = Layer(out_activation)
        elif isinstance(out_activation, DeeplayModule):
            ...
        else:
            raise ValueError(
                f"out_activation must be a nn.Module or a DeeplayModule or None, got {out_activation}"
            )

        self.blocks = LayerList()
        for i, f_out in enumerate(self.hidden_dims):
            f_in = self.in_features if i == 0 else self.hidden_dims[i - 1]

            self.blocks.append(
                LayerActivationNormalizationBlock(
                    Layer(nn.Linear, f_in, f_out)
                    if f_in
                    else Layer(nn.Linear, f_in, f_out),
                    Layer(nn.ReLU),
                    # We can give num_features as an argument to nn.Identity
                    # because it is ignored. This means that users do not have
                    # to specify the number of features for nn.Identity.
                    Layer(nn.Identity, num_features=out_features),
                )
            )

        self.out_layer = LayerActivationNormalizationBlock(
            Layer(nn.Linear, self.hidden_dims[-1], self.out_features),
            Layer(out_activation or nn.Identity),
            Layer(nn.Identity, num_features=self.out_features),
        )

    def forward(self, x):
        x = nn.Flatten()(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x
