from ..applications import Application
from ..classification import Classifier
from ...core.config import Config
from ...core.templates import Layer
from ....deeplay.components import (
    SpatialBroadcastDecoder2d,
    PositionalEncodingSinusoidal2d,
    PositionalEncodingLinear2d,
    ImageRegressionHead,
)
import torch
import torch.nn as nn
from torch.optim import Adam


class VanillaGAN(Application):
    @staticmethod
    def defaults():
        return (
            Config()
            .hidden_dim(10)
            .generator(Layer("backbone") >> Layer("head"))
            .generator.backbone(SpatialBroadcastDecoder2d)
            .generator.backbone.encoding(PositionalEncodingLinear2d)
            .generator.head(ImageRegressionHead)
            .discriminator(Classifier)
            .discriminator.num_classes(1)
            .discriminator.on_first_forward(
                "generator.backbone.output_size", lambda _, x: x.shape[2:]
            )
            .discriminator_loss(nn.MSELoss)
            .generator_loss(nn.MSELoss)
            .discriminator_optimizer(Adam, lr=1e-4, betas=(0.5, 0.999))
            .generator_optimizer(Adam, lr=2e-4, betas=(0.5, 0.999))
        )

    def __init__(
        self,
        hidden_dim=128,
        generator=None,
        discriminator=None,
        sampler=None,
        discriminator_loss=None,
        generator_loss=None,
        discriminator_optimizer=None,
        generator_optimizer=None,
    ):
        Application.__init__(
            self,
            hidden_dim=hidden_dim,
            generator=generator,
            discriminator=discriminator,
            sampler=sampler,
            discriminator_loss=discriminator_loss,
            generator_loss=generator_loss,
            discriminator_optimizer=discriminator_optimizer,
            generator_optimizer=generator_optimizer,
        )

        self.hidden_dim = self.attr("hidden_dim")

        self.generator = self.new("generator")
        self.discriminator = self.new("discriminator")

        self.discriminator_loss = self.new("discriminator_loss")
        self.generator_loss = self.new("generator_loss")

    def configure_optimizers(self):
        return (
            self.new(
                "generator_optimizer",
                extra_kwargs={"params": self.generator.parameters()},
            ),
            self.new(
                "discriminator_optimizer",
                extra_kwargs={"params": self.discriminator.parameters()},
            ),
        )

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def forward(self, x=None):
        if x is None:
            x = self.sample_noise(1)
            x = x.to(self.device).float()
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self._get_image(batch)
        z = self.sample_noise(x.size(0))

        if optimizer_idx == 0:
            # store generated images for the discriminator optimizer
            self._generated_imgs = self(z)

            valid = torch.ones(x.size(0), 1).type_as(x)

            g_loss = self.generator_loss(
                self.discriminator(self._generated_imgs), valid
            )

            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)

            return g_loss

        if optimizer_idx == 1:
            valid = torch.ones(x.size(0), 1).type_as(x)
            fake = torch.zeros(x.size(0), 1).type_as(x)

            real_loss = self.discriminator_loss(self.discriminator(x), valid)
            fake_loss = self.discriminator_loss(
                self.discriminator(self.generator(z).detach()), fake
            )
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
            # if d_loss.item() < 0.25:
            #     return d_loss * 0

            return d_loss

    def _get_image(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch
