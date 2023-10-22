import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from .. import (
    Config,
    DeeplayModule,
)


class Application(DeeplayModule, L.LightningModule):
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
        return self.new(
            "optimizer", extra_kwargs={"params": self.parameters()}, now=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        y_hat = self(batch)
        return y_hat
