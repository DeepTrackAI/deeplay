import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy
from ..core import DeepTorchModule
from ..config import Config, Ref
from ..templates import Layer
from ..components import ConvolutionalEncoder, DenseEncoder, CategoricalClassificationHead

class ImageClassifier(DeepTorchModule, pl.LightningModule):

    defaults = (
        Config()
        .num_classes(2)
        .backbone(ConvolutionalEncoder)
        .connector(nn.Flatten)
        .head(CategoricalClassificationHead, num_classes=Ref("num_classes"))
    )

    def __init__(self, num_classes, backbone=None, connector=None, head=None):
        """Image classifier.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        backbone : None, Config, nn.Module, optional
            Backbone config. If None, a default Encoder backbone is used.
            If nn.Module, it is used as the backbone.
        connector : None, Dict, nn.Module, optional
            Connector config. Connects the backbone to the head by reducing the
            to 1D (channels).
        head : None, Dict, nn.Module, optional
            Head config. If None, a default CategoricalClassificationHead head is used.
            If Dict, it is used as kwargs for the head class.
            If nn.Module, it is used as the head.
        """
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            connector=connector,
            head=head,
        )

        self.num_classes = self.attr("num_classes")
        self.backbone = self.create("backbone")
        self.connector = self.create("connector")
        self.head = self.create("head")

        self.val_accuracy = Accuracy()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, (batch_size, channels, width, height)

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.backbone(x)
        x = self.connector(x)
        x = self.head(x)
        return x

    def classify(self, x):
        """Classify input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, (batch_size, channels, width, height)

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        y = self.forward(x)
        y = torch.argmax(y, dim=1)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # one-hot encode y
        y = F.one_hot(y, num_classes=self.num_classes).float()

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # one-hot encode y
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss = F.cross_entropy(y_hat, y_one_hot)

        classification = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(classification, y)

        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # one-hot encode y
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss = F.cross_entropy(y_hat, y_one_hot)

        classification = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(classification, y)

        self.log("test_acc", self.val_accuracy, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss
