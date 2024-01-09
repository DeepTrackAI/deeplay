from typing import Optional, Sequence, Callable, List

from ...components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from ..application import Application
from ...external import External, Optimizer, Adam


import torch
import torch.nn as nn


class VariationalAutoEncoder(Application):
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        input_size: Optional[Sequence[int]] = (28, 28),
        channels: Optional[List[int]] = [32, 64],
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        reconstruction_loss: Optional[Callable] = nn.BCELoss(reduction="sum"),
        latent_dim=int,
        optimizer=None,
        **kwargs,
    ):
        red_size = [int(dim / (2 ** len(channels))) for dim in input_size]
        self.encoder = encoder or self._get_default_encoder(channels)
        self.fc_mu = nn.Linear(
            channels[-1] * red_size[0] * red_size[1],
            latent_dim,
        )
        self.fc_var = nn.Linear(
            channels[-1] * red_size[0] * red_size[1],
            latent_dim,
        )
        self.fc_dec = nn.Linear(
            latent_dim,
            channels[-1] * red_size[0] * red_size[1],
        )
        self.decoder = decoder or self._get_default_decoder(channels[::-1], red_size)
        self.reconstruction_loss = reconstruction_loss or nn.BCELoss(reduction="sum")
        self.latent_dim = latent_dim

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params():
            return self.parameters()

    def configure_optimizers(self):
        return self.optimizer

    def _get_default_encoder(self, channels):
        encoder = ConvolutionalEncoder2d(
            1,
            channels,
            channels[-1],
        )
        encoder.post.configure(nn.Flatten)
        return encoder

    def _get_default_decoder(self, channels, red_size):
        decoder = ConvolutionalDecoder2d(
            channels[0],
            channels,
            1,
        )
        decoder.pre.configure(
            nn.Unflatten,
            dim=1,
            unflattened_size=(channels[0], red_size[0], red_size[1]),
        )
        return decoder

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat, mu, log_var = self(x)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        loss = {"rec_loss": rec_loss, "KL": KLD}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return sum(loss.values())

    def compute_loss(self, y_hat, y, mu, log_var):
        rec_loss = self.reconstruction_loss(y_hat, y)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec_loss, KLD

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decode(z)
        return y_hat, mu, log_var
