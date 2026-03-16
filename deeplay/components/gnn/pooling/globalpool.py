from typing import Optional

import torch.nn as nn
import torch
from deeplay.module import DeeplayModule

class GlobalGraphPooling(DeeplayModule):
    """
    Pools all the nodes of the graph to a single cluster.

    (Inspired by MinCut-pooling ('Spectral Clustering with Graph Neural Networks for Graph Pooling'):
    but with the assignment matrix S being deterministic (all nodes are pooled into one cluster))

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)

    - output: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_clusters, node_features)
        - s: torch.Tensor of shape (num_nodes, num_clusters)

    Examples
    --------
    >>> global_pool = GlobalGraphPooling().build()
    >>> inp = {}
    >>> inp["x"] = torch.randn(3, 2)
    >>> inp["batch"] = torch.zeros(3, dtype=int)
    >>> inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> out = global_pool(inp)
    """

    def __init__(
            self,
            ):
        super().__init__()

        class Select(DeeplayModule):
            def forward(self, x):
                return torch.ones((x.shape[0], 1))  
            
        class ClusterMatrixForBatch(DeeplayModule):
            def forward(self, S, B):
                unique_graphs = torch.unique(B)
                num_graphs = len(unique_graphs)

                S_ = torch.zeros(S.shape[0] * num_graphs)

                row_indices = torch.arange(S.shape[0])
                col_indices = B

                S_[row_indices * num_graphs + col_indices] = S.view(-1)
                B_ = torch.arange(num_graphs)

                return S_.reshape([S.shape[0], -1]), B_
            

        class Reduce(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s.transpose(-2,-1), x)

        self.select = Select()
        self.select.set_input_map('x')
        self.select.set_output_map('s')

        self.batch_compatible = ClusterMatrixForBatch()
        self.batch_compatible.set_input_map('s', 'batch')
        self.batch_compatible.set_output_map('s', 'batch')

        self.reduce = Reduce()
        self.reduce.set_input_map('x', 's')
        self.reduce.set_output_map('x')

    def forward(self, x):
        x = self.select(x)
        x = self.batch_compatible(x)
        x = self.reduce(x)
        return (x)

    
class GlobalGraphUpsampling(DeeplayModule):
    """
    Reverse of GlobalGraphPooling.
    Only upsampling the node features.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_clusters, node_features)
        - s: torch.Tensor of shape (num_nodes, num_clusters)

    - output: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        
    Examples
    --------
    >>> global_upsampling = GlobalGraphUpsampling()
    >>> global_upsampling = global_upsampling.build()

    >>> inp = {}
    >>> inp["x"] = torch.randn(1, 2)
    >>> inp["s"] = torch.ones((3, 1))
    >>> out = global_upsampling(inp)
    """

    def __init__(
            self,
            ):
        super().__init__()
    
        class Upsample(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s, x)
            
        self.upsample = Upsample()
        self.upsample.set_input_map('x', 's')
        self.upsample.set_output_map('x')

    def forward(self, x):
        x = self.upsample(x)
        return x