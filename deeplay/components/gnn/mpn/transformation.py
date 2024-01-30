from .ct import CombineTransform


class Transform(CombineTransform):
    """Transform module for MPN."""

    def get_forward_args(self, x):
        """Get the arguments for the Transform module.
        An MPN Transform module takes the following arguments:
        - node features of sender nodes (x[A[0]])
        - node features of receiver nodes (x[A[1]])
        - edge features (edgefeat)
        A is the adjacency matrix of the graph.
        """
        x, A, edgefeat = x
        return x[A[0]], x[A[1]], edgefeat
