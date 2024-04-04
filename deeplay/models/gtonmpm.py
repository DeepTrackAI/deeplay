from typing import Sequence, Type, Union

from . import MPM

import torch.nn as nn


class GraphToNodeMPM(MPM):
    """Graph-to-Node Message Passing Neural Network (MPN) model.

    Parameters
    ----------
    hidden_features: list[int]
        Number of hidden units in each Message Passing Layer.
    out_features: int
        Number of output features.
    out_activation: template-like
        Specification for the output activation of the model. Default: nn.Identity.


    Configurables
    -------------
    - hidden_features (list[int]): Number of hidden units in each Message Passing Layer.
    - out_features (int): Number of output features.
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

        NOTE: node_in_features and edge_in_features are inferred from the input data.

    - output: torch.Tensor of shape (num_nodes, out_features)

    Examples
    --------
    >>> model = GraphToNodesMPM([64, 64], 1).create()
    >>> inp = {}
    >>> inp["x"] = torch.randn(10, 16)
    >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
    >>> inp["edge_attr"] = torch.randn(20, 8)
    >>> model(inp).shape
    torch.Size([10, 1])
    """

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(
            hidden_features=hidden_features,
            out_features=out_features,
            out_activation=out_activation,
        )
