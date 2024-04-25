from deeplay import DeeplayModule

import torch


class Sum(DeeplayModule):
    """Sums the edge features of each receiver node."""

    def forward(self, x, edge_index, edge_attr):
        _sum = torch.zeros(
            x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device
        )
        indices = edge_index[1].unsqueeze(1).expand_as(edge_attr)
        return _sum.scatter_add_(0, indices, edge_attr)


class WeightedSum(DeeplayModule):
    """Sums the edge features of each receiver node with weights."""

    def forward(self, x, edge_index, edge_attr, weights):
        _sum = torch.zeros(
            x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device
        )
        indices = edge_index[1].unsqueeze(1).expand_as(edge_attr)
        return _sum.scatter_add_(0, indices, edge_attr * weights)


class Mean(DeeplayModule):
    """Averages the edge features of each receiver node."""

    Sum = Sum()

    def forward(self, x, edge_index, edge_attr):
        _sum = self.Sum(x, edge_index, edge_attr)
        return _sum / (
            torch.bincount(edge_index[1], minlength=x.size(0)).unsqueeze(1).clamp(min=1)
        )


class Prod(DeeplayModule):
    """Product of the edge features of each receiver node."""

    def forward(self, x, edge_index, edge_attr):
        # returns prod * mask so not connected nodes have prod = 0
        prod = torch.ones(
            x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device
        )
        indices = edge_index[1].unsqueeze(1).expand_as(edge_attr)
        prod = prod.scatter_reduce_(0, indices, edge_attr, "prod")

        mask = (
            torch.bincount(edge_index[1], minlength=x.size(0)).unsqueeze(1).clamp(max=1)
        )
        return prod * mask


class Min(DeeplayModule):
    """Minimum of the edge features of each receiver node."""

    def forward(self, x, edge_index, edge_attr):
        _min = torch.ones(
            x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device
        ) * float("inf")
        indices = edge_index[1].unsqueeze(1).expand_as(edge_attr)
        _min = _min.scatter_reduce_(0, indices, edge_attr, "min")

        # remove inf values
        _min[_min == float("inf")] = 0
        return _min


class Max(DeeplayModule):
    """Maximum of the edge features of each receiver node."""

    def forward(self, x, edge_index, edge_attr):
        _max = torch.ones(
            x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device
        ) * float("-inf")
        indices = edge_index[1].unsqueeze(1).expand_as(edge_attr)
        _max = _max.scatter_reduce_(0, indices, edge_attr, "max")

        # remove -inf values
        _max[_max == float("-inf")] = 0
        return _max
