from typing import Sequence, Type, Union, Optional, overload, List, Any

from deeplay import (
    Layer,
    DeeplayModule,
    Parallel,
    MultiLayerPerceptron,
    MessagePassingNeuralNetwork,
    FromDict,
)

import torch.nn as nn


class MPM(DeeplayModule):
    """Message Passing Neural Network (MPN) model.

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

        NOTE: node_in_features and edge_in_features are inferred from the input data.

    - output: torch.Tensor of shape (num_nodes, out_features)

    Examples
    --------
    >>> # MPN with 2 hidden layers of 64 units each and 1 output feature
    >>> model = MPN([64, 64], 1).create()
    >>> # Define input as a dictionary with node features, edge index and edge features
    >>> inp = {}
    >>> inp["x"] = torch.randn(10, 16)
    >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
    >>> inp["edge_attr"] = torch.randn(20, 8)
    >>> out = model(inp)
    >>> print(out.shape)
    torch.Size([10, 1])

    """

    hidden_features: Sequence[int]
    out_features: int

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        pool: Union[Type[nn.Module], nn.Module, None] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.out_features = out_features

        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        if len(hidden_features) == 0:
            raise ValueError("hidden_features must contain at least one element")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_features must be positive, got {hidden_features}"
            )

        if pool is None:
            pool_layer = Layer(nn.Identity)
        elif isinstance(pool, type) and issubclass(pool, nn.Module):
            pool_layer = Layer(pool)
        elif isinstance(pool, DeeplayModule):
            pool_layer = pool
        else:
            raise ValueError(
                f"Invalid pool layer {pool}. Expected a nn.Module, DeeplayModule or None. Found {type(pool)}"
            )

        self.encoder = Parallel(
            **{
                key: MultiLayerPerceptron(
                    in_features=None,
                    hidden_features=[],
                    out_features=hidden_features[0],
                    flatten_input=False,
                ).set_input_map(key)
                for key in ("x", "edge_attr")
            }
        )

        self.backbone = MessagePassingNeuralNetwork(
            hidden_features=hidden_features[:-1],
            out_features=hidden_features[-1],
            out_activation=nn.ReLU,
        )

        self.selector = FromDict("x")

        self.pool = pool_layer

        self.head = MultiLayerPerceptron(
            in_features=hidden_features[-1],
            hidden_features=[hidden_features[-1] // 2, hidden_features[-1] // 4],
            out_features=out_features,
            out_activation=out_activation,
            flatten_input=False,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        x = self.selector(x)
        x = self.pool(x)
        x = self.head(x)
        return x
