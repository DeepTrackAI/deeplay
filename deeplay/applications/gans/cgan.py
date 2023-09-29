from ..applications import DeeplayLightningModule
from ..classification import ImageClassifier
from ...config import Config, Ref
from ...templates import Layer
from ...components import (
    SpatialBroadcastDecoder2d,
    PositionalEncodingSinusoidal2d,
    PositionalEncodingLinear2d,
    ImageRegressionHead,
)
import torch
import torch.nn as nn
from torch.optim import Adam


class ClassConditionedGAN(DeeplayLightningModule):
    @staticmethod
    def defaults():
        return (
            Config()
            .num_classes(10)
            .hidden_dim(10)
            .generator(Layer("backbone") >> Layer("head"))
            .generator.backbone(SpatialBroadcastDecoder2d)
            .generator.backbone.encoding(PositionalEncodingLinear2d)
            .generator.head(ImageRegressionHead)
            .discriminator(ImageClassifier)
            .discriminator.num_classes(1)
            .discriminator.on_first_forward(
                "generator.backbone.output_size", lambda _, x: x.shape[2:]
            )
            .embedding(nn.Embedding, num_embeddings=Ref("num_classes"), embedding_dim=4)
            .discriminator_loss(nn.MSELoss)
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
        DeeplayLightningModule.__init__(
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
        self.embedding = self.new("embedding")

        self.discriminator_loss = self.new("discriminator_loss")

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

    def sample_noise(self, size):
        batch_size, *spatials = size
        return torch.randn(batch_size, self.hidden_dim, *spatials).to(self.device)

    def sample_conditioned_latent(self, x, z=None):
        if z is None:
            z = self.sample_noise(x.size())
            z = z.to(self.device).float()

        embed = self.embedding(x.long())
        conditioned_latent = torch.cat([embed, z], dim=1)
        return conditioned_latent

    def forward(self, x, z=None):
        image = self.generate(x, z=z)
        valid = self.discriminate(image, x)
        return image, valid

    def discriminate(self, x, condition):
        embed = self.embedding(condition)
        while len(embed.shape) < len(x.shape):
            embed = embed.unsqueeze(-1)
        embed = embed.repeat(1, 1, *x.shape[2:])

        x = torch.cat([x, embed], dim=1)
        return self.discriminator(x)

    def generate(self, condition, z=None):
        conditioned_latent = self.sample_conditioned_latent(condition, z=z)
        return self.generator(conditioned_latent)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, condition = batch

        z = self.sample_noise(condition.size())
        z = z.to(self.device).float()

        if optimizer_idx == 0:
            # store generated images for the discriminator optimizer
            y_hat = self.generate(condition, z=z)

            valid = torch.ones(x.size(0), 1).type_as(x)

            g_loss = self.discriminator_loss(self.discriminate(y_hat, condition), valid)

            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)

            return g_loss

        if optimizer_idx == 1:
            valid = torch.ones(x.size(0), 1).type_as(x)
            fake = torch.zeros(x.size(0), 1).type_as(x)
            fake_x = self.generate(condition, z=z).detach()
            real_loss = self.discriminator_loss(self.discriminate(x, condition), valid)
            fake_loss = self.discriminator_loss(
                self.discriminate(fake_x, condition), fake
            )
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
            return d_loss

    def _get_image(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch
