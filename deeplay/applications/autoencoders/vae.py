from typing import Optional, Sequence, Callable, List

from deeplay.components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam


import torch
import torch.nn as nn


class VariationalAutoEncoder(Application):
    input_size: tuple
    channels: list
    latent_dim: int
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    beta: float
    reconstruction_loss: torch.nn.Module
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
        beta=1,
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
        self.beta = beta

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def _get_default_encoder(self, channels):
        encoder = ConvolutionalEncoder2d(
            1,
            channels,
            channels[-1],
        )
        encoder.postprocess.configure(nn.Flatten)
        return encoder

    def _get_default_decoder(self, channels, red_size):
        decoder = ConvolutionalDecoder2d(
            channels[0],
            channels,
            1,
            out_activation=nn.Sigmoid,
        )
        # for block in decoder.blocks[:-1]:
        #     block.upsample.configure(nn.ConvTranspose2d, kernel_size=3, stride=2, in_channels=block.in_channels, out_channels=block.out_channels)

        decoder.preprocess.configure(
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

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat, mu, log_var = self(x)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + self.beta * KLD
        loss = {"rec_loss": rec_loss, "KL": KLD, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss

    def compute_loss(self, y_hat, y, mu, log_var):
        rec_loss = self.reconstruction_loss(y_hat, y)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec_loss, KLD

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decode(z)
        return y_hat, mu, log_var
