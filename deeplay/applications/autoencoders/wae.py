from typing import Optional, Sequence, Callable, List

from deeplay.components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam, Layer

import torch
import torch.nn as nn


class WassersteinAutoEncoder(Application):
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    discriminator: torch.nn.Module
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        input_size: Optional[Sequence[int]] = (28, 28),
        channels: Optional[List[int]] = [32, 64, 128],
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        reconstruction_loss: Optional[Callable] = nn.MSELoss(reduction="mean"),
        latent_dim=int,
        optimizer=None,
        **kwargs,
    ):
        red_size = [int(dim / (2 ** (len(channels) - 1))) for dim in input_size]
        self.encoder = encoder or self._get_default_encoder(channels)
        self.fc_enc = nn.Linear(
            channels[-1] * red_size[0] * red_size[1],
            latent_dim,
        )
        self.fc_dec = nn.Linear(
            latent_dim,
            channels[-1] * red_size[0] * red_size[1],
        )
        self.decoder = decoder or self._get_default_decoder(channels[::-1], red_size)
        self.reconstruction_loss = reconstruction_loss or nn.MSELoss(reduction="mean")
        self.latent_dim = latent_dim
        self.reg_weight = 1.0
        self.z_var = 1.0

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=5e-3)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def _get_default_encoder(self, channels):
        encoder = ConvolutionalEncoder2d(
            1,
            channels[:-1],
            channels[-1],
        )
        encoder.postprocess.configure(nn.Flatten)
        encoder.blocks[1:].layer.configure(stride=2)
        encoder["blocks", :].all.normalized(nn.BatchNorm2d).remove(
            "pool", allow_missing=True
        )
        return encoder

    def _get_default_decoder(self, channels, red_size):
        decoder = ConvolutionalDecoder2d(
            channels[0],
            channels[1:],
            1,
            out_activation=nn.Sigmoid,
        )
        decoder.preprocess.configure(
            nn.Unflatten,
            dim=1,
            unflattened_size=(channels[0], red_size[0], red_size[1]),
        )
        decoder[..., "layer"].all.configure(
            nn.ConvTranspose2d,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        decoder["blocks", :-1].all.normalized(
            nn.BatchNorm2d,
        )
        decoder.blocks[-1].layer.configure(
            nn.Conv2d,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        decoder["blocks", :].all.remove("upsample", allow_missing=True)
        return decoder

    def encode(self, x):
        x = self.encoder(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat, z = self(x)
        rec_loss, mmd_loss = self.compute_loss(y_hat, y, z)
        loss = {
            "rec_loss": rec_loss,
            "mmd_loss": mmd_loss,
        }
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

    def compute_IMQ(self, x1, x2):
        # Inverse MultiQuadratic kernel
        C = 2 * self.latent_dim * self.z_var
        kernel = C / (1e-8 + C + (x1 - x2).pow(2).sum(-1))
        return kernel

    def compute_mmd(self, z):
        batch_size = z.size(0)
        qz = torch.randn_like(z)

        qz_kernel = self.compute_kernel(qz, qz)
        z_kernel = self.compute_kernel(z, z)
        qz_z_kernel = self.compute_kernel(qz, z)

        qz_kernel = (qz_kernel.sum() - qz_kernel.diag().sum()) / (
            batch_size * (batch_size - 1)
        )
        z_kernel = (z_kernel.sum() - z_kernel.diag().sum()) / (
            batch_size * (batch_size - 1)
        )
        qz_z_kernel = qz_z_kernel.sum() / (batch_size * batch_size)

        mmd = self.reg_weight * (qz_kernel + z_kernel - 2 * qz_z_kernel)
        return mmd

    def compute_kernel(self, x1, x2):
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(1).expand(N, N, D)
        x2 = x2.unsqueeze(0).expand(N, N, D)

        result = self.compute_IMQ(x1, x2)
        return result

    def compute_loss(self, y_hat, y, z):
        rec_loss = self.reconstruction_loss(y_hat, y)
        mmd_loss = self.compute_mmd(z)
        return rec_loss, mmd_loss

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x, z
