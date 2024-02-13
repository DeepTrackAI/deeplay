from typing import Optional, Sequence

from ..application import Application
from ...external import External, Optimizer, Adam


import torch
import torch.nn.functional as F
import torchmetrics as tm


class Regressor(Application):
    model: torch.nn.Module
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module = torch.nn.L1Loss(),
        optimizer=None,
        **kwargs,
    ):
        super().__init__(loss=loss, **kwargs)

        self.model = model
        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.model.parameters()

    def forward(self, x):
        return self.model(x)
