from deeplay import DeeplayModule

import torch


class Sum(DeeplayModule):
    """Sums the edge features of each receiver node."""

    def forward(self, x, A, edgefeat):
        return torch.zeros(
            (x.shape[0], edgefeat.shape[1]),
            dtype=edgefeat.dtype,
            device=edgefeat.device,
        ).scatter_reduce_(
            dim=0,
            index=A[1].unsqueeze(1).repeat(1, edgefeat.shape[1]),
            src=edgefeat,
            reduce="sum",
        )


class Mean(DeeplayModule):
    """Averages the edge features of each receiver node."""

    Sum = Sum()

    def forward(self, x, A, edgefeat):
        _sum = self.Sum(x, A, edgefeat)
        counts = self.Sum(x, A, torch.ones_like(edgefeat))
        return _sum / counts.clamp(min=1)


class Prod(DeeplayModule):
    """Product of the edge features of each receiver node."""

    Sum = Sum()

    def forward(self, x, A, edgefeat):
        # returns prod * mask so not connected nodes have prod = 0
        return torch.ones(
            (x.shape[0], edgefeat.shape[1]),
            dtype=edgefeat.dtype,
            device=edgefeat.device,
        ).scatter_reduce_(
            dim=0,
            index=A[1].unsqueeze(1).repeat(1, edgefeat.shape[1]),
            src=edgefeat,
            reduce="prod",
        ) * self.Sum(
            x, A, torch.ones_like(edgefeat)
        ).clamp(
            max=1
        )


class Min(DeeplayModule):
    """Minimum of the edge features of each receiver node."""

    def forward(self, x, A, edgefeat):
        _min = (
            torch.ones(
                (x.shape[0], edgefeat.shape[1]),
                dtype=edgefeat.dtype,
                device=edgefeat.device,
            )
            * float("inf")
        ).scatter_reduce_(
            dim=0,
            index=A[1].unsqueeze(1).repeat(1, edgefeat.shape[1]),
            src=edgefeat,
            reduce="min",
        )
        # remove inf values
        _min[_min == float("inf")] = 0

        return _min


class Max(DeeplayModule):
    """Maximum of the edge features of each receiver node."""

    def forward(self, x, A, edgefeat):
        _max = (
            torch.ones(
                (x.shape[0], edgefeat.shape[1]),
                dtype=edgefeat.dtype,
                device=edgefeat.device,
            )
            * float("-inf")
        ).scatter_reduce_(
            dim=0,
            index=A[1].unsqueeze(1).repeat(1, edgefeat.shape[1]),
            src=edgefeat,
            reduce="max",
        )

        # remove -inf values
        _max[_max == float("-inf")] = 0

        return _max
