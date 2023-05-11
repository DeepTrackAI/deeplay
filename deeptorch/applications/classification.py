import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy

from .. import Default, default
from ..backbones.encoders import Encoder2d
from ..heads.classification import CategoricalClassificationHead
from ..connectors import FlattenDense


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, backbone=default, connector=default, head=default):
        """Image classifier.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        backbone : None, Dict, nn.Module, optional
            Backbone config. If None, a default Encoder2d backbone is used.
            If Dict, it is used as kwargs for the backbone class.
            If nn.Module, it is used as the backbone.
        connector : None, Dict, nn.Module, optional
            Connector config. Connects the backbone to the head by reducing the
            dimensionality of the output of the backbone from 3D (width, height, channels)
            to 1D (channels).
        head : None, Dict, nn.Module, optional
            Head config. If None, a default CategoricalClassificationHead head is used.
            If Dict, it is used as kwargs for the head class.
            If nn.Module, it is used as the head.
        """
        super().__init__()
        self.num_classes = num_classes

        self.backbone = Default(backbone, Encoder2d, channels_out=[16, 32, 64])
        self.connector = Default(connector, FlattenDense, out_features=128)
        self.head = Default(
            head, CategoricalClassificationHead, num_classes=num_classes
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

    def build(self, *args):
        """Build the image classifier.


        Parameters
        ----------
        args : torch.Tensor or tuple
            Example input. Should contain batch dimension.
            If tuple, it's interpreted as the shape of the input tensor and
            a random tensor is generated.
            Can be multiple arguments if the backbone requires multiple inputs.

        """
        backbone = self.backbone.build()
        connector = self.connector.build()
        head = self.head.build()

        self.classifier = nn.Sequential(backbone, connector, head)

        # Conduct a dry run to initialize the model
        model_inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.to(self.device).to(self.dtype)

            elif isinstance(arg, tuple):
                arg = torch.rand(*arg).to(self.device).to(self.dtype)

            else:
                raise ValueError(f"Invalid input type {type(arg)}")

            model_inputs.append(arg)

        self.classifier(*model_inputs)

        return self

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
        return self.classifier(x)

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
