from typing import Optional, Sequence

from deeplay.applications import Application
from deeplay.external import Optimizer, Adam


import torch
import torch.nn.functional as F
import torchmetrics as tm

from .classifier import Classifier


class BinaryClassifier(Application):
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

    def compute_loss(self, y_hat, y):
        if isinstance(self.loss, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
            y = y.float()
        return super().compute_loss(y_hat, y)

    def metrics_preprocess(self, y_hat, y: torch.Tensor):
        if y.is_floating_point():
            y = y > 0.5
        return super().metrics_preprocess(y_hat, y)

    def forward(self, x):
        return self.model(x)
