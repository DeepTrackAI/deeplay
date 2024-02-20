import copy
from typing import (
    Callable,
    Iterator,
    Tuple,
    Literal,
    Optional,
    Dict,
    TypeVar,
    Sequence,
    Union,
)

import lightning as L
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.nn.modules.module import Module

from deeplay import DeeplayModule, Optimizer

import torchmetrics as tm
import copy

T = TypeVar("T")


class Application(DeeplayModule, L.LightningModule):
    def __init__(
        self,
        loss: Optional[Union[nn.Module, Callable[..., torch.Tensor]]] = None,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Sequence[tm.Metric]] = None,
        train_metrics: Optional[Sequence[tm.Metric]] = None,
        val_metrics: Optional[Sequence[tm.Metric]] = None,
        test_metrics: Optional[Sequence[tm.Metric]] = None,
    ):
        super().__init__()
        if loss:
            self.loss = loss
        if optimizer:
            self.optimizer = optimizer
            self._provide_paramaters_if_has_none(optimizer)

        metrics = metrics or []
        self.train_metrics = tm.MetricCollection(
            [*self.clone_metrics(metrics), *(train_metrics or [])],
            prefix="train",
        )
        self.val_metrics = tm.MetricCollection(
            [*self.clone_metrics(metrics), *(val_metrics or [])],
            prefix="val",
        )
        self.test_metrics = tm.MetricCollection(
            [*self.clone_metrics(metrics), *(test_metrics or [])],
            prefix="test",
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, y_hat, y) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.loss:
            return self.loss(y_hat, y)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        try:
            return self.optimizer.create()

        except AttributeError as e:
            raise AttributeError(
                "Application has no configured optimizer. Make sure to pass optimizer=... to the constructor."
            ) from e

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if not isinstance(loss, dict):
            loss = {"loss": loss}

        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log_metrics(
            "train", y_hat, y, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        x, y = self.val_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if not isinstance(loss, dict):
            loss = {"loss": loss}

        for name, v in loss.items():
            self.log(
                f"val_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log_metrics(
            "val",
            y_hat,
            y,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return sum(loss.values())

    def test_step(self, batch, batch_idx):
        x, y = self.test_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if not isinstance(loss, dict):
            loss = {"loss": loss}

        for name, v in loss.items():
            self.log(
                f"test_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log_metrics(
            "test",
            y_hat,
            y,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return sum(loss.values())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        y_hat = self(batch)
        return y_hat

    def log_metrics(
        self, kind: Literal["train", "val", "test"], y_hat, y, **logger_kwargs
    ):
        metrics: tm.MetricCollection = getattr(self, f"{kind}_metrics")
        metrics(y_hat, y)

        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                **logger_kwargs,
            )

    @L.LightningModule.trainer.setter
    def trainer(self, trainer):
        # Call the original setter
        L.LightningModule.trainer.fset(self, trainer)

        # Overrides default implementation to do a deep search for all
        # submodules that have a trainer attribute and set it to the
        # same trainer instead for just direct children.
        for module in self.modules():
            if module is self:
                continue
            try:
                if hasattr(module, "trainer") and module.trainer is not trainer:
                    module.trainer = trainer
            except RuntimeError:
                # hasattr can raise RuntimeError if the module is not attached to a trainer
                if isinstance(module, L.LightningModule):
                    module.trainer = trainer

    @staticmethod
    def clone_metrics(metrics: T) -> T:
        return [
            metric.clone() if hasattr(metric, "clone") else copy.copy(metric)
            for metric in metrics
        ]

    def train_preprocess(self, batch):
        return batch

    def val_preprocess(self, batch):
        return batch

    def test_preprocess(self, batch):
        return batch

    def _provide_paramaters_if_has_none(self, optimizer):
        if isinstance(optimizer, Optimizer):
            print("optimizer is an instance of Optimizer")

            @optimizer.params
            def f(self):
                return self.parameters()

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        name_child_iterator = list(super().named_children())
        # optimizers last
        not_optimizers = [
            (name, child)
            for name, child in name_child_iterator
            if not isinstance(child, Optimizer)
        ]
        optimizers = [
            (name, child)
            for name, child in name_child_iterator
            if isinstance(child, Optimizer)
        ]

        yield from (not_optimizers + optimizers)

    def create_optimizer_with_params(self, optimizer, params):
        if isinstance(optimizer, Optimizer):
            optimizer.configure(params=params)
            return optimizer.build()
        else:
            return optimizer
