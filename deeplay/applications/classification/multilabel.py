from typing import Optional, Sequence

from deeplay.applications import Application
from deeplay.external import Optimizer, Adam


import torch
import torch.nn.functional as F
import torchmetrics as tm

from .classifier import Classifier


class MultiLabelClassifier(Application):
    model: torch.nn.Module
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module = torch.nn.BCELoss(),
        optimizer=None,
        **kwargs,
    ):
        if kwargs.get("metrics", None) is None:
            kwargs["metrics"] = [tm.Accuracy("binary")]

        super().__init__(loss=loss, **kwargs)

        self.model = model
        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.model.parameters()

    def compute_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)
