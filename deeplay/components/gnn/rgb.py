"""Recurrent Graph Block Module

This module defines the `RecurrentGraphBlock` class, a component designed for
recurrent graph-based computations. It employs a recurrent structure to process
graph data iteratively using specified combine, layer, and head modules.
"""

import torch
from torch_geometric.data import Data
from deeplay import DeeplayModule
from deeplay.components.dict import CatDictElements


class RecurrentGraphBlock(DeeplayModule):
    """Recurrent graph processing block for iterative feature transformation.

    This module combines graph data features and hidden states, processes
    them through a recurrent structure, and generates outputs using a
    specified head module. It supports modular design for flexibility in
    defining the combine, layer, and head operations.

    Parameters
    ----------
    layer : DeeplayModule
        Module that applies transformations to the graph data at each
        iteration.
    head : DeeplayModule
        Module that processes the output from the layer and generates
        final predictions.
    hidden_features : int
        The number of hidden features for the recurrent block.
    num_iter : int
        The number of recurrent iterations.
    combine : DeeplayModule, optional
        The module responsible for combining graph features with hidden
        states. Default is `CatDictElements(("x", "hidden"))`.

    Returns
    -------
    list
        A list of outputs generated at each recurrent iteration.

    Raises
    ------
    AttributeError
        If the `combine` module does not contain `source` and `target`
        attributes. Graph `combine` modules compatible with
        `RecurrentGraphBlock` must have `source` and `target` attributes to
        specify the keys to concatenate. A catalog of `combine` operations
        is available in the `deeplay.components.dict` module.

    Example
    -------
    >>> combine = CatDictElements(("x", "hidden"))
    >>> layer = dl.MessagePassingNeuralNetwork([], 128)
    >>> head = MultiLayerPerceptron(128, [], 128)
    >>> block = RecurrentGraphBlock(
    ...     layer=layer,
    ...     head=head,
    ...     combine=combine,
    ...     hidden_features=128,
    ...     num_iter=10
    ... )

    >>> # Set maps for input and output keys
    >>> block.head.set_input_map("x")
    >>> block.head.set_output_map("x")
    >>> block = block.create()

    >>> # Create input graph data
    >>> data = Data(
    ...     x=torch.randn(3, 128),
    ...     edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    ...     edge_attr=torch.randn(4, 128)
    ... )
    >>> outputs = block(data)
    >>> len(outputs)
    10
    """

    combine: DeeplayModule
    layer: DeeplayModule
    head: DeeplayModule

    def __init__(
        self,
        layer: DeeplayModule,
        head: DeeplayModule,
        hidden_features: int,
        num_iter: int,
        combine: DeeplayModule = CatDictElements(("x", "hidden")),
    ):
        super().__init__()
        self.combine = combine
        self.layer = layer
        self.head = head
        self.hidden_features = hidden_features
        self.num_iter = num_iter

        if not all(hasattr(self.combine, attr) for attr in ("source", "target")):
            raise AttributeError(
                "The 'combine' module must have 'source' and 'target' attributes. "
                "These specify the keys to concatenate. Ensure the 'combine' "
                "module is initialized with valid 'source' and 'target' keys. "
                "Refer to the `CatDictElements` class in the `deeplay.components.dict` "
                "module for guidance."
            )
        self.hidden_variables_name = self.combine.target

    def initialize_hidden(self, x):
        """Initialize hidden states for the graph nodes if not already provided
        in the input data.

        Parameters
        ----------
        x : Data or dict
            The input graph data or dictionary-like structure.

        Returns
        -------
        Data or dict
            The input graph data with initialized hidden states.
        """
        x = x.clone() if isinstance(x, Data) else x.copy()
        for source, hidden_variable_name in zip(
            self.combine.source, self.hidden_variables_name
        ):
            if hidden_variable_name not in x:
                x.update(
                    {
                        hidden_variable_name: torch.zeros(
                            x[source].size(0), self.hidden_features
                        ).to(x[source].device)
                    }
                )
        return x

    def forward(self, x):
        """Forward pass to process the graph data through recurrent iterations.

        Parameters
        ----------
        x : Data or dict
            The input graph data or dictionary-like structure.

        Returns
        -------
        list
            A list of outputs generated at each recurrent iteration.
        """
        x = self.initialize_hidden(x)
        outputs = []
        for _ in range(self.num_iter):
            x = self.combine(x)
            x = self.layer(x)
            outputs.append(self.head(x))

        return outputs
