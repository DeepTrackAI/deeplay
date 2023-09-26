import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import (
    Config,
    Ref,
    DeeplayModule,
    Layer,
    ImageToImageEncoder,
    ConvolutionalDecoder,
)

# from ..backbones.encoders import Encoder2d
# from ..backbones.decoders import Decoder2d
# from ..connectors import FlattenDenseq

__all__ = ["Autoencoder", "EncoderDecoder", "ImageToImageEncoderDecoder"]


class EncoderDecoder(DeeplayModule):
    defaults = (
        Config()
        .depth(4)
        .encoder.depth(Ref("depth"))
        .encoder.layer.padding(1)
        .decoder.depth(Ref("depth"))
        .bottleneck(nn.Identity)
    )

    def __init__(self, depth=4, encoder=None, bottleneck=None, decoder=None):
        super().__init__(
            depth=depth, encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )

        self.encoder = self.new("encoder")
        self.decoder = self.new("decoder")

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ImageToImageEncoderDecoder(EncoderDecoder):
    defaults = (
        Config()
        .merge(None, EncoderDecoder.defaults)
        .encoder(ImageToImageEncoder)
        .decoder(ConvolutionalDecoder)
    )


class Autoencoder(DeeplayModule, pl.LightningModule):
    defaults = (
        Config()
        .backbone(EncoderDecoder)
        .head(nn.LazyConv2d, out_channels=1, kernel_size=1, stride=1)
    )

    def __init__(self, backbone=None, head=None):
        super().__init__(backbone=backbone, head=head)

        self.backbone = self.new("backbone")
        self.head = self.new("head")

    def forward(self, x):
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
