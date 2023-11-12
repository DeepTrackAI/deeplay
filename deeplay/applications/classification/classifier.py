from typing import Optional, Sequence

from ..application import Application
from ...external import External, Optimizer, Adam


import torch
import torch.nn.functional as F
import torchmetrics as tm


class Classifier(Application):
    model: torch.nn.Module
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer=None,
        make_targets_one_hot: bool = False,
        **kwargs,
    ):
        super().__init__(loss=loss, **kwargs)
        self.model = model
        self.optimizer = optimizer or Adam(lr=1e-3)
        self.make_targets_one_hot = make_targets_one_hot

        @self.optimizer.params
        def params():
            return self.model.parameters()

    def compute_loss(self, y_hat, y):
        if self.make_targets_one_hot:
            y = F.one_hot(y, num_classes=y_hat.size(1)).float()

        return self.loss(y_hat, y)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)
