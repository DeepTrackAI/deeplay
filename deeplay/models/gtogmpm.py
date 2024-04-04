from typing import Sequence, Type, Union

from . import MPM
from deeplay import DeeplayModule, FromDict

import torch.nn as nn
from torch_geometric.nn import global_mean_pool

import itertools


class GlobalMeanPool(DeeplayModule):
    """
    Global mean pooling layer for Graph Neural Networks.

    Constraints
    -----------
    - Inputs:
        - x: torch.Tensor of shape (num_nodes, num_features)
        - batch: torch.Tensor of shape (num_nodes,)

        Inputs can be passed to the forward method as a tuple or as separate arguments.

    - Output: torch.Tensor of shape (batch_size, num_features)

    Examples
    --------
    >>> # Global mean pooling layer
    >>> layer = GlobalMeanPool().create()
    >>> # Define input as a tuple of node features and batch
    >>> x = torch.randn(10, 16)
    >>> batch = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]).long()
    >>> out1 = layer((x, batch))
    >>> # Define input as separate arguments
    >>> out2 = layer(x, batch)
    >>> torch.allclose(out1, out2)
    True
    >>> out1.shape
    torch.Size([3, 16])

    """

    def forward(self, x):
        x = tuple(itertools.chain(*x)) if isinstance(x[0], tuple) else x
        return global_mean_pool(*x)


class GraphToGlobalMPM(MPM):
    """Graph-to-Global Message Passing Neural Network (MPN) model.

    Parameters
    ----------
    hidden_features: list[int]
        Number of hidden units in each Message Passing Layer.
    out_features: int
        Number of output features.
    pool: template-like
        Specification for the pooling of the model. Default: nn.Identity.
    out_activation: template-like
        Specification for the output activation of the model. Default: nn.Identity.


    Configurables
    -------------
    - hidden_features (list[int]): Number of hidden units in each Message Passing Layer.
    - out_features (int): Number of output features.
    - pool (template-like): Specification for the pooling of the model. Default: nn.Identity.
    - out_activation (template-like): Specification for the output activation of the model. Default: nn.Identity.
    - encoder (template-like): Specification for the encoder of the model. Default: dl.Parallel consisting of two MLPs to process node and edge features.
    - backbone (template-like): Specification for the backbone of the model. Default: dl.MessagePassingNeuralNetwork.
    - selector (template-like): Specification for the selector of the model. Default: dl.FromDict("x") selecting the node features.
    - head (template-like): Specification for the head of the model. Default: dl.MultiLayerPerceptron.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_in_features).
        - edge_index: torch.Tensor of shape (2, num_edges).
        - edge_attr: torch.Tensor of shape (num_edges, edge_in_features).
        - batch: torch.Tensor of shape (num_nodes,).

        NOTE: node_in_features and edge_in_features are inferred from the input data.

    - output: torch.Tensor of shape (batch_size, out_features)

    Examples
    --------
    >>> model = GraphToGlobalMPM([64, 64], 1).create()
    >>> inp = {}
    >>> inp["x"] = torch.randn(10, 16)
    >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
    >>> inp["edge_attr"] = torch.randn(20, 8)
    >>> inp["batch"] = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).long()
    >>> model(inp).shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        pool: Union[Type[nn.Module], nn.Module] = GlobalMeanPool,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(
            hidden_features,
            out_features,
            pool,
            out_activation,
        )

        # Configures selector to retrieve the node features (x)
        # and the batch tensor as inputs to the pool layer
        self.replace("selector", FromDict("x", "batch"))
