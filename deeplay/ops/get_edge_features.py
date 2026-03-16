
from deeplay import DeeplayModule

class GetEdgeFeaturesNew(DeeplayModule):
    """"""

    def forward(self, x, edge_index):
        """Get the node features of neighboring nodes for each edge.
        - node features of sender nodes (x[edge_index[0]])
        - node features of receiver nodes (x[edge_index[1]])
        
        edge_index denote the connectivity of the graph.
        """
        return x[edge_index[0]], x[edge_index[1]]
