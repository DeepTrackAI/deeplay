from .cla import CombineLayerActivation


class GetEdgeFeatures(CombineLayerActivation):
    """"""

    def get_forward_args(self, x):
        """Get the node features of neighboring nodes for each edge.
        - node features of sender nodes (x[edge_index[0]])
        - node features of receiver nodes (x[edge_index[1]])
        
        edge_index denote the connectivity of the graph.
        """
        x, edge_index = x
        return x[edge_index[0]], x[edge_index[1]]
