import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import Default, default
from ..backbones.encoders import Encoder2d
from ..backbones.decoders import Decoder2d
from ..connectors import FlattenDense


class ImageAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        latent_dim=32,
        encoder=default,
        encoder_connector=default,
        decoder_connector=default,
        decoder=default,
    ):
        """
        Image autoencoder.
        Evaluated as:
           encoder -> encoder_connector -> decoder_connector -> decoder
        Where encoder transforms the image to a downsampled representation,
        encoder_connector transforms the representation to a latent space,
        decoder_connector transforms the latent space to a spatial representation,
        and decoder transforms the spatial representation to a reconstructed image.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space.
        encoder : None, Dict, nn.Module, optional
            Encoder config. If None, a default Encoder2d backbone is used.
            If Dict, it is used as kwargs for the backbone class.
            If nn.Module, it is used as the backbone.
        encoder_connector : None, Dict, nn.Module, optional
            Encoder connector config. If None, a default FlattenDense connector is used.
            If Dict, it is used as kwargs for the connector class.
            If nn.Module, it is used as the connector.
        decoder_connector : None, Dict, nn.Module, optional
            Decoder connector config. If None, a default DenseUnflatten connector is used.
            If Dict, it is used as kwargs for the connector class.
            If nn.Module, it is used as the connector.
        decoder : None, Dict, nn.Module, optional
            Decoder config. If None, a default Decoder2d backbone is used.
            If Dict, it is used as kwargs for the backbone class.
            If nn.Module, it is used as the backbone.
        """

        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Default(encoder, Encoder2d, channels_out=[16, 32, 64])
        self.encoder_connector = Default(
            encoder_connector, FlattenDense, out_features=128
        )
        self.decoder_connector = Default(
            decoder_connector, DenseUnflatten, out_features=128
        )
        self.decoder = Default(decoder, Decoder2d, channels_in=[64, 32, 16])
