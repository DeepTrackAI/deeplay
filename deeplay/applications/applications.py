import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import (
    Config,
    DeeplayModule,
)

class Application(DeeplayModule, pl.LightningModule):

    config = (
        Config()
        .optimizer(torch.optim.Adam, lr=1e-3)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        raise NotImplementedError
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat