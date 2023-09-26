import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..components import (UNet, )
from ..config import (Config)

from .applications import DeeplayLightningModule

# from ..backbones.encoders import Encoder2d
# from ..backbones.decoders import Decoder2d
# from ..connectors import FlattenDenseq

__all__ = [
    "ImageSegmentor",
]



class ImageSegmentor(DeeplayLightningModule):
    @staticmethod
    def defaults():
        return (
            Config()
            .backbone(UNet)
            .head()
        )

    def __init__(self, hidden_dim=2, encoder=None, decoder=None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )

        self.backbone = self.new("backbone")
        self.head = self.new("head")

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def segment(self, x, th=0.5):
        return self.forward(x) > th

