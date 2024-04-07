from typing import Any
import torch
import torch.nn as nn

from deeplay.module import DeeplayModule


class FromLogs(DeeplayModule):

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.logs[self.key]
