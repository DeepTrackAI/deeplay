import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..components import (
    UNet,
    ImageSegmentationHead,
)
from ..config import Config
from ..utils import center_pad_to_largest, center_crop
from .applications import Application

# from ..backbones.encoders import Encoder2d
# from ..backbones.decoders import Decoder2d
# from ..connectors import FlattenDenseq

__all__ = [
    "ImageSegmentor",
]


class ImageSegmentor(Application):
    @staticmethod
    def defaults():
        return Config().backbone(UNet).head(ImageSegmentationHead).loss(nn.BCELoss)

    def __init__(self, backbone=None, head=None, loss=None):
        super().__init__(
            backbone=backbone,
            head=head,
            loss=loss,
        )

        self.backbone = self.new("backbone")
        self.head = self.new("head")

        self.loss = self.new("loss")

    def forward(self, x, pad_output=True):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        pad_output : bool
            Whether to pad the output to the input size.
            Default is False.
        """
        y = self.backbone(x)
        y = self.head(y)
        if pad_output:
            (y, x) = center_pad_to_largest([y, x])

        return y

    def segment(self, x, th=0.5, pad_output=True):
        return self.forward(x, pad_output=pad_output) > th

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, pad_output=False)

        # Take the center crop of y_hat to match the size of y
        y = center_crop(y, y_hat.shape[2:])
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, pad_output=False)

        # Take the center crop of y_hat to match the size of y
        y = center_crop(y, y_hat.shape[2:])
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, pad_output=False)

        # Take the center crop of y_hat to match the size of y
        y = center_crop(y, y_hat.shape[2:])
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
