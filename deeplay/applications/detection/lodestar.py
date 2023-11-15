from ..application import Application
from ...components import ConvolutionalNeuralNetwork

import torch
import torch.nn as nn


class LodeSTAR(Application):
    def __init__(
        self,
        model=None,
        num_outputs=2,
        intra_consistency_loss=None,
        inter_consistency_loss=None,
        **kwargs
    ):
        self.num_outputs = num_outputs
        self.model = model or self._build_default_model()
        self.intra_consistency_loss = intra_consistency_loss or nn.L1Loss(
            reduction="mean"
        )
        self.inter_consistency_loss = inter_consistency_loss or nn.L1Loss(
            reduction="mean"
        )

        super().__init__(**kwargs)

    def _build_default_model(self):
        cnn = ConvolutionalNeuralNetwork(
            None,
            [32, 32, 64, 64, 64, 64, 64, 64, 64],
            self.num_outputs,
        )
        cnn.blocks[2].pool.configure(nn.MaxPool2d, kernel_size=2)

    def forward(self, x):
        B, C, Hx, Wx = x.shape

        y = self.model(x)

        _, _, Hy, Wy = y.shape

        x_range = torch.arange(Hy, device=x.device) * Hx / Hy
        y_range = torch.arange(Wy, device=x.device) * Wx / Wy

        Y, X = torch.meshgrid(y_range, x_range)

        delta_x = y[:, 0]
        delta_y = y[:, 1]
        weights = y[:, -1].sigmoid()

        X = X + delta_x
        Y = Y + delta_y

        return torch.cat([X[:, None], Y[:, None], y[:, 2:-1], weights[:, None]], dim=1)

    def normalize(self, weights):
        weights = weights + 1e-6
        return weights / weights.sum(dim=(2, 3), keepdim=True)

    def reduce(self, X, weights):
        return (X * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

    def compute_loss(self, y_hat, y):
        y_pred, weights = y_hat[:, :-1], y_hat[:, -1]

        weights = self.normalize(weights)
        y_reduced = self.reduce(y_pred, weights)

        consistency = (y_pred - y_reduced[..., None, None]) * weights
        consistency_loss = self.intra_consistency_loss(
            consistency, torch.zeros_like(consistency)
        )

        y_reduced_on_initial = self.apply_inverse(y_reduced, y)

        average_on_initial = y_reduced_on_initial.mean(dim=0, keepdim=True)

        inter_consistency_loss = self.inter_consistency_loss(
            y_reduced_on_initial, average_on_initial
        )

        return {
            "consistency_loss": consistency_loss,
            "inter_consistency_loss": inter_consistency_loss,
        }

    def apply_inverse(self, y_reduced, y):
        for f in reversed(y):
            y_reduced = f(y_reduced)
        return y_reduced
