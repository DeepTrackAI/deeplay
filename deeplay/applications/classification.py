import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy
from ..core import DeeplayModule
from ..config import Config, Ref
from ..templates import Layer
from ..components import (
    ImageToVectorEncoder,
    CategoricalClassificationHead,
)

__all__ = ["ImageClassifier"]


class ImageClassifier(DeeplayModule, pl.LightningModule):
    defaults = (
        Config()
        .backbone(ImageToVectorEncoder)
        .head(CategoricalClassificationHead, num_classes=Ref("num_classes"))
        .head.output.activation(
            Ref(
                "num_classes",
                lambda num_classes: nn.Sigmoid()
                if num_classes == 1
                else nn.LogSoftmax(),
            )
        )
        .optimizer(torch.optim.Adam, lr=1e-3)
        .loss(nn.NLLLoss)
    )

    def __init__(
        self, num_classes, backbone=None, head=None, optimizer=None, loss=None, **kwargs
    ):
        """Image classifier.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        backbone : None, Config, nn.Module, optional
            Backbone config. If None, a default Encoder backbone is used.
            If nn.Module, it is used as the backbone.
        head : None, Dict, nn.Module, optional
            Head config. If None, a default CategoricalClassificationHead head is used.
            If Dict, it is used as kwargs for the head class.
            If nn.Module, it is used as the head.
        optimizer : None, Config, nn.Module, optional
            Optimizer config. If None, a default Adam optimizer is used.
            If nn.Module, it is used as the optimizer.
        loss : None, Config, nn.Module, optional
            Loss config. If None, a default CrossEntropyLoss loss is used.
            If nn.Module, it is used as the loss.

        """
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            optimizer=optimizer,
            loss=loss,
            **kwargs
        )

        self.num_classes = self.attr("num_classes")
        self.backbone = self.new("backbone")
        self.head = self.new("head")

        self.loss = self.new("loss")
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
        optimizer = self.new("optimizer", extra_kwargs={"params": self.parameters()})
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
