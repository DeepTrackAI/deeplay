import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics as tm
from ..core.core import DeeplayModule
from ..core.config import Config, Ref
from ..core.templates import Layer
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
    "Regressor",
    "MLPRegressor",
    "MLPRegressorTiny",
    "MLPRegressorSmall",
    "MLPRegressorMedium",
    "MLPRegressorLarge",
    "MLPRegressorMassive",
]


class Regressor(Application):
    """Trainable module for regression tasks.

    This module is a trainable application designed for regression tasks. It is structured to work seamlessly with PyTorch Lightning's training loop, providing additional configurables suitable for regression.

    Configurables
    -------------
    - model (nn.Module): The neural network model to be used for classification. It should produce un-normalized output of shape (batch_size, num_features). No activation should be applied to the output. (Required)
    - optimizer (torch.optim): The optimizer to be used for training the model. (Default: torch.optim.Adam, lr=1e-3)
    - loss (nn.Module): The loss function to be used for training. (Default: nn.L1Loss)

    Constraints
    -----------
    - model: The model should produce un-normalized output of shape (batch_size, num_classes). No activation should be applied to the output.

    """

    @staticmethod
    def defaults():
        return Config().optimizer(torch.optim.Adam, lr=1e-3).loss(nn.L1Loss)

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
        self.train_metrics = []  # self.new("train_metrics")
        self.val_metrics = []  # self.new("val_metrics")

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

        for metric in self.train_metrics:
            metric.update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, pred_class = y_hat.max(1)

        if self.make_target_onehot:
            loss = self.loss(y_hat, F.one_hot(y, num_classes=y_hat.size(1)).float())
        else:
            loss = self.loss(y_hat, y)

        for metric in self.val_metrics:
            metric.update(y_hat, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, pred_class = y_hat.max(1)

        if self.make_target_onehot:
            loss = self.loss(y_hat, F.one_hot(y, num_classes=y_hat.size(1)).float())
        else:
            loss = self.loss(y_hat, y)

        for metric in self.val_metrics:
            metric.update(y_hat, y)

        return loss

    def on_train_epoch_end(self) -> None:
        for metric in self.train_metrics:
            self.log(
                "train_" + metric.__class__.__name__,
                metric.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        for metric in self.val_metrics:
            self.log(
                "val_" + metric.__class__.__name__,
                metric.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            metric.reset()

    def on_test_epoch_end(self) -> None:
        for metric in self.val_metrics:
            self.log(
                "test_" + metric.__class__.__name__,
                metric.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            metric.reset()


class MLPRegressor(Regressor):
    @staticmethod
    def defaults():
        return (
            MLPRegressor.defaults()
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


class MLPRegressorTiny(Regressor):
    @staticmethod
    def defaults():
        return Regressor.defaults().model(MLPTiny)


class MLPRegressorSmall(Regressor):
    @staticmethod
    def defaults():
        return Regressor.defaults().model(MLPSmall)


class MLPRegressorMedium(Regressor):
    @staticmethod
    def defaults():
        return Regressor.defaults().model(MLPMedium)


class MLPRegressorLarge(Regressor):
    @staticmethod
    def defaults():
        return Regressor.defaults().model(MLPLarge)


class MLPRegressorMassive(Regressor):
    @staticmethod
    def defaults():
        return Regressor.defaults().model(MLPMassive)
