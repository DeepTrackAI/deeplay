import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import (
    Config,
    DeeplayModule,
)


class Application(DeeplayModule, pl.LightningModule):
    config = Config().optimizer(torch.optim.Adam, lr=1e-3)

    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat
