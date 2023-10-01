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
    MultiLayerPerceptron,
    MLPTiny,
    MLPSmall,
    MLPMedium,
    MLPLarge,
    MLPMassive,
)
from ..applications import Application

__all__ = [
    "Classifier",
    "MLPClassifier",
    "MLPClassifierTiny",
    "MLPClassifierSmall",
    "MLPClassifierMedium",
    "MLPClassifierLarge",
    "MLPClassifierMassive",
]


class Classifier(Application):
    @staticmethod
    def defaults():
        return (
            Config()
            .make_target_onehot(False)
            .optimizer(torch.optim.Adam, lr=1e-3)
            .loss(nn.CrossEntropyLoss)
        )

    def __init__(
        self, model=None, make_target_onehot=False, optimizer=None, loss=None, **kwargs
    ):
        super().__init__(
            model=model,
            make_target_onehot=make_target_onehot,
            optimizer=optimizer,
            loss=loss,
            **kwargs
        )

        self.make_target_onehot = self.attr("make_target_onehot")

        self.model = self.new("model")
        self.loss = self.new("loss")

        # metrics
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, pred_class = y_hat.max(1)

        if self.make_target_onehot:
            loss = self.loss(y_hat, F.one_hot(y, num_classes=y_hat.size(1)).float())
        else:
            loss = self.loss(y_hat, y)

        train_accuracy = self.train_accuracy(pred_class, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": train_accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, pred_class = y_hat.max(1)

        if self.make_target_onehot:
            loss = self.loss(y_hat, F.one_hot(y, num_classes=y_hat.size(1)).float())
        else:
            loss = self.loss(y_hat, y)

        val_accuracy = self.val_accuracy(pred_class, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": val_accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, pred_class = y_hat.max(1)

        if self.make_target_onehot:
            loss = self.loss(y_hat, F.one_hot(y, num_classes=y_hat.size(1)).float())
        else:
            loss = self.loss(y_hat, y)

        val_accuracy = self.val_accuracy(pred_class, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": val_accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


class MLPClassifier(Application):
    @staticmethod
    def defaults():
        return (
            Classifier.defaults()
            .in_features(None)
            .model(MultiLayerPerceptron)
            .model.in_features(Ref("in_features"))
            .model.hidden_dims(Ref("hidden_dims"))
            .model.out_features(Ref("num_classes"))
        )

    def __init__(
        self,
        in_features: int or None,
        hidden_dims: list[int],
        num_classes: int,
        **kwargs
    ):
        self.in_features = self.attr("in_features")
        self.hidden_dims = self.attr("hidden_dims")
        self.num_classes = self.attr("num_classes")
        super().__init__(**kwargs)


class MLPClassifierTiny(MLPClassifier):
    @staticmethod
    def defaults():
        return MLPClassifier.defaults().model(MLPTiny)


class MLPClassifierSmall(MLPClassifier):
    @staticmethod
    def defaults():
        return MLPClassifier.defaults().model(MLPSmall)


class MLPClassifierMedium(MLPClassifier):
    @staticmethod
    def defaults():
        return MLPClassifier.defaults().model(MLPMedium)


class MLPClassifierLarge(MLPClassifier):
    @staticmethod
    def defaults():
        return MLPClassifier.defaults().model(MLPLarge)


class MLPClassifierMassive(MLPClassifier):
    @staticmethod
    def defaults():
        return MLPClassifier.defaults().model(MLPMassive)
