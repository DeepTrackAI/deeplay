from .cla import CombineLayerActivation


class Update(CombineLayerActivation):
    """Update module for MPN."""

    def get_forward_args(self, x):
        """Get the arguments for the Update module.
        An MPN Update module takes the following arguments:
        - node features (x)
        - aggregated edge features (aggregates)
        """
        x, aggregates = x
        return x, aggregates
