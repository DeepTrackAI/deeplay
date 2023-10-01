import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import (
    Config,
    Ref,
    ImageToImageEncoder,
    VectorToImageDecoder,
    SpatialBroadcastDecoder2d,
    ImageToImageDecoder,
    ImageRegressionHead,
    Bottleneck,
    VariationalBottleneck,
    Layer,
)

from .applications import Application

# from ..backbones.encoders import Encoder2d
# from ..backbones.decoders import Decoder2d
# from ..connectors import FlattenDenseq

__all__ = [
    "SimpleAutoencoder",
    "Autoencoder",
    "FullAutoencoder",
    "VariationalAutoencoder",
    "BetaVAE",
]


def _prod(x):
    p = x[0]
    for i in x[1:]:
        p *= i
    return p


class SimpleAutoencoder(Application):
    @staticmethod
    def defaults():
        return (
            Config()
            .hidden_dim(2)
            .encoder(Layer("flatten") >> Layer("layer") >> Layer("activation"))
            .encoder.flatten(nn.Flatten)
            .encoder.layer(nn.LazyLinear, out_features=Ref("hidden_dim"))
            .encoder.activation(nn.LeakyReLU, negative_slope=0.2)
            .decoder(Layer("layer") >> Layer("activation") >> Layer("unflatten"))
            .decoder.layer(nn.LazyLinear, out_features=Ref("input_size", _prod))
            .decoder.activation(nn.Sigmoid)
            .decoder.unflatten(nn.Unflatten, dim=1, unflattened_size=Ref("input_size"))
            .loss(nn.MSELoss, reduction="sum")
            # hooks
            .on_first_forward("input_size", lambda _, x: x.shape[1:])
        )

    def __init__(self, hidden_dim=2, encoder=None, decoder=None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )

        self.encoder = self.new("encoder")
        self.decoder = self.new("decoder")

        self.loss = self.new("loss")

    def forward(self, x, return_latent=False):
        latent = self.encode(x)
        x = self.decode(latent)
        if return_latent:
            return x, latent
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def _get_image(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch

    def _compute_loss(self, batch, batch_idx):
        x = self._get_image(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("test_loss", loss)
        return loss


class Autoencoder(SimpleAutoencoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .hidden_dim(2)
            .encoder(ImageToImageEncoder, depth=2)
            .bottleneck(Bottleneck, hidden_dim=Ref("hidden_dim"))
            .decoder(
                SpatialBroadcastDecoder2d,
                depth=2,
                output_size=Ref("input_size", lambda s: s[1:]),
            )
            .head(ImageRegressionHead, output_size=Ref("input_size"))
            .head.output.activation(nn.Sigmoid)
            .loss(nn.MSELoss, reduction="sum")
            # hooks
            .on_first_forward("input_size", lambda _, x: x.shape[1:])
        )

    def __init__(
        self, hidden_dim=2, encoder=None, bottleneck=None, decoder=None, head=None
    ):
        Application.__init__(
            self,
            hidden_dim=hidden_dim,
            encoder=encoder,
            decoder=decoder,
        )

        self.encoder = self.new("encoder")
        self.bottleneck = self.new("bottleneck")
        self.decoder = self.new("decoder")
        self.head = self.new("head")

        self.loss = self.new("loss")

    def encode(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.head(x)
        return x


class FullAutoencoder(Autoencoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .hidden_dim(2)
            .encoder(ImageToImageEncoder, depth=2)
            .connector_to_vector(nn.Flatten)
            .bottleneck(Bottleneck, hidden_dim=Ref("hidden_dim"))
            .connector_to_image(nn.Unflatten, dim=1, unflattened_size=Ref("base_size"))
            .decoder(ImageToImageDecoder, depth=2)
            .head(ImageRegressionHead)
            .head.output.activation(nn.Sigmoid)
            # hooks
            .on_first_forward("head.output_size", lambda _, x: x.shape[1:])
            .bottleneck.on_first_forward(
                "connector_to_image.base_size", lambda _, x: x.shape[1:]
            )
        )


class VariationalAutoencoder(Autoencoder):
    @staticmethod
    def defaults():
        return (
            Config()
            .merge(None, Autoencoder.defaults())
            .bottleneck(VariationalBottleneck)
            .loss(nn.BCELoss, reduction="sum")
        )

    def encode(self, x, return_posteriors=False):
        x = self.encoder(x)
        x, posts = self.bottleneck(x)

        if return_posteriors:
            return x, posts
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.head(x)
        return x

    def forward(self, x, return_posteriors=False):
        x, posts = self.encode(x, return_posteriors=True)
        x = self.decode(x)
        if return_posteriors:
            return x, posts
        return x

    def _compute_loss(self, batch, batch_idx):
        x = self._get_image(batch)
        y_hat, posts = self(x, return_posteriors=True)

        reconstruction_loss = self.loss(y_hat.squeeze(), x.squeeze())
        kl_loss = torch.cat(
            [
                torch.distributions.kl_divergence(post, prior)
                for post, prior in zip(posts, self.bottleneck.priors)
            ]
        ).sum()

        self.log(
            "reconstruction_loss", reconstruction_loss, prog_bar=True, on_step=True
        )
        self.log("kl_loss", kl_loss, prog_bar=True, on_step=True)

        return reconstruction_loss * x[0].numel() + kl_loss


class BetaVAE(VariationalAutoencoder):
    @staticmethod
    def defaults():
        return Config().merge(None, VariationalAutoencoder.defaults()).beta(1)

    def __init__(self, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.beta = self.attr("beta")

    def _compute_loss(self, batch, batch_idx):
        x = self._get_image(batch)
        y_hat, posts = self(x, return_posteriors=True)

        reconstruction_loss = self.loss(y_hat, x)

        kl_loss = (
            torch.cat(
                [
                    torch.distributions.kl_divergence(post, prior)
                    for post, prior in zip(posts, self.bottleneck.priors)
                ]
            )
            .sum(-1)
            .mean()
        )

        self.log(
            "reconstruction_loss", reconstruction_loss, prog_bar=True, on_step=True
        )
        self.log("kl_loss", kl_loss, prog_bar=True, on_step=True)

        return reconstruction_loss + self.beta * kl_loss
