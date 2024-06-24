from typing import Optional, Sequence

from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam


import torch
import torch.nn.functional as F
import torchmetrics as tm


class Classifier(Application):

    model: torch.nn.Module
    metrics: list

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer=None,
        make_targets_one_hot: bool = False,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        if num_classes is not None and kwargs.get("metrics", None) is None:
            kwargs["metrics"] = [tm.Accuracy("multiclass", num_classes=num_classes)]
        super().__init__(loss=loss, optimizer=optimizer or Adam(lr=1e-3), **kwargs)

        self.model = model
        self.make_targets_one_hot = make_targets_one_hot
        self.num_classes = num_classes

    def compute_loss(self, y_hat, y):
        if self.make_targets_one_hot:
            y = F.one_hot(y, num_classes=self.num_classes or y_hat.size(1)).float()

        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)
