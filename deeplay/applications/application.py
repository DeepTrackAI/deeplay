import copy
import logging
from typing import (Callable, Dict, Iterator, Literal, Optional, Sequence,
                    Tuple, TypeVar, Union)

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
import tqdm
from lightning.pytorch.callbacks import RichProgressBar
from torch.nn.modules.module import Module

import deeplay as dl
from deeplay import DeeplayModule, Optimizer

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

T = TypeVar("T")

class LogHistory(L.Callback):
    """A keras-like history callback for lightning. Keeps track of metrics and losses during training and validation.
    
    Example:
    >>> history = LogHistory()
    >>> trainer = dl.Trainer(callbacks=[history])
    >>> trainer.fit(model, train_dataloader, val_dataloader)
    >>> history.history {"train_loss_epoch": {"value": [0.1, 0.2, 0.3], "epoch": [0, 1, 2], "step": [0, 100, 200]}}
    """

    @property
    def history(self):
        return {key: 
                {
                    "value": [item["value"] for item in value],
                    "epoch": [item["epoch"] for item in value],
                    "step": [item["step"] for item in value]
                }
                for key, value in self._history.items()}

    @property
    def step_history(self):
        return {key: 
                {
                    "value": [item["value"] for item in value],
                    "epoch": [item["epoch"] for item in value],
                    "step": [item["step"] for item in value]
                }
                for key, value in self._step_history.items()}
    
    def __init__(self):
        self._history = {}
        self._step_history = {}

    def on_train_batch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.endswith("_step"):
                self._step_history.setdefault(key, []).append(self._logitem(trainer, value))

    def on_train_epoch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))
    
    def on_validation_epoch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))

    def _logitem(self, trainer, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "value": value
        }

    def plot(self, *args, yscale="log", **kwargs):
        """Plot the history of the metrics and losses.
        """

        history = self.history
        step_history = self.step_history
        
        keys = list(history.keys())
        keys = [key.replace("val", "").replace("train", "") for key in keys]
        unique_keys = list(set(keys))
        # sort unique keys same as keys
        unique_keys.sort(key=lambda x: keys.index(x))
        keys = unique_keys

        max_width = 3
        rows = len(keys) // max_width + 1
        width = min(len(keys), max_width)

        fig, axes = plt.subplots(rows, width, figsize=(15, 5 * rows))

        if len(keys) == 1:
            axes = np.array([[axes]])

        for ax, key in zip(axes.ravel(), keys):
            train_key = "train" + key
            val_key = "val" + key
            step_key = ("train" + key).replace("epoch", "step")

            
            if step_key in step_history:
                ax.plot(step_history[step_key]["step"], step_history[step_key]["value"], label=step_key, color="C1", alpha=0.25)
            if train_key in history:
                step = np.array(history[train_key]["step"])
                step[1:] = step[1:] - (step[1:] - step[:-1]) / 2
                step[0] /= 2
                marker_kwargs = dict(marker="o", markerfacecolor="white", markeredgewidth=1.5) if len(step) < 20 else {}
                ax.plot(step, history[train_key]["value"], label=train_key, color="C1", **marker_kwargs)

            if val_key in history:
                marker_kwargs = dict(marker="d", markerfacecolor="white", markeredgewidth=1.5) if len(step) < 20 else {}
                ax.plot(history[val_key]["step"], history[val_key]["value"], label=val_key, color="C3", linestyle="--", **marker_kwargs)
            
            ax.set_title(key.replace("_", " ").replace("epoch", "").strip().capitalize())
            ax.set_xlabel("Step")

            ax.legend()
            ax.set_yscale(yscale)
        
        return fig, axes


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

    def fit(self, 
            train_data,
            val_data=None,
            max_epochs=None, 
            batch_size=32, 
            steps_per_epoch=100,
            replace=False, 
            val_batch_size=None, 
            val_steps_per_epoch=10,
            callbacks=[],
            **kwargs) -> LogHistory:
        """Train the model on the training data.

        Train the model on the training data, with optional validation data.

        Parameters
        ----------
        max_epochs : int
            The maximum number of epochs to train the model.
        batch_size : int
            The batch size to use for training.
        steps_per_epoch : int
            The number of steps per epoch (used if train_data is a Feature).
        replace : bool or float
            Whether to replace the data after each epoch (used if train_data is a Feature).
            If a float, the data is replaced with the given probability.
        val_batch_size : int
            The batch size to use for validation. If None, the training batch size is used.
        val_steps_per_epoch : int
            The number of steps per epoch for validation.
        callbacks : list
            A list of callbacks to use during training.
        permute_target_channels : bool or "auto"
            Whether to permute the target channels to channel first. If "auto", the model will
            attempt to infer the correct permutation based on the input and target shapes.
        **kwargs
            Additional keyword arguments to pass to the trainer.
        """        

        val_batch_size = val_batch_size or batch_size
        val_steps_per_epoch = val_steps_per_epoch or 10
        train_data = self.create_data(train_data, batch_size, steps_per_epoch, replace)
        val_data = self.create_data(val_data, val_batch_size, val_steps_per_epoch, False) if val_data else None
        
        history = LogHistory()
        progressbar = RichProgressBar()
        
        callbacks = callbacks + [history, progressbar]
        trainer = dl.Trainer(max_epochs=max_epochs, callbacks=callbacks, **kwargs)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        if not self._has_built:
            self.build()

        self.train()
        if val_data:
            val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False) if val_data else None
            trainer.fit(self, train_dataloader, val_dataloader)
        else:
            trainer.fit(self, train_dataloader)
        self.eval()
        return history

    def test(self, 
             data, 
             metrics: Union[tm.Metric, Tuple[str, tm.Metric], Sequence[Union[tm.Metric, Tuple[str, tm.Metric]]], Dict[str, tm.Metric]],
             batch_size: int = 32):
        """Test the model on the given data.

        Test the model on the given data, using the given metrics. Metrics can be
        given as a single metric, a tuple of name and metric, a sequence of metrics
        (or tuples of name and metric) or a dictionary of metrics. In the case of
        tuples, the name is used as the key in the returned dictionary. In the case
        of metrics, the name of the metric is used as the key in the returned dictionary.

        Parameters
        ----------
        data : data-like
            The data to test the model on. Can be a Feature, a torch.utils.data.Dataset, a tuple of tensors, a tensor or a numpy array.
        metrics : metric-like
            The metrics to use for testing. Can be a single metric, a tuple of name and metric, a sequence of metrics (or tuples of name and metric) or a dictionary of metrics.
        batch_size : int
            The batch size to use for testing.
        """

        device = self.trainer.strategy.root_device
        self.to(device)
        test_data = self.create_data(data)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        dict_metrics: Dict[str, tm.Metric]
        if isinstance(metrics, tm.Metric):
            dict_metrics = {metrics._get_name(): metrics}
        elif isinstance(metrics, tuple):
            dict_metrics = {metrics[0]: metrics[1]}
        elif isinstance(metrics, list):
            m = {}
            for metric in metrics:
                if isinstance(metric, tm.Metric):
                    m[metric._get_name()] = metric
                elif isinstance(metric, tuple):
                    m[metric[0]] = metric[1]
            dict_metrics = m
        else:
            dict_metrics = {k: v for k, v in metrics.items()}

        for (x, y) in tqdm.tqdm(test_dataloader):
            y_hat = self(x.to(device))
            for metric in dict_metrics.values():
                metric.to(device)
                metric.update(y_hat.to(device), y.to(device))
        
        return {name: dict_metrics[name].compute() for name in dict_metrics}

    def __call__(self, *args, **kwargs):
        args = [self._maybe_to_channel_first(arg) for arg in args]
        return super().__call__(*args, **kwargs)

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
        x, y = batch
        x = self._maybe_to_channel_first(x)
        y = self._maybe_to_channel_first(y)
        
        return x, y

    val_preprocess = train_preprocess
    test_preprocess = train_preprocess

    def _provide_paramaters_if_has_none(self, optimizer):
        if isinstance(optimizer, Optimizer):
            if "params" in optimizer.kwargs:
                return
            else:

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
    
    def create_data(self, data, batch_size=32, steps_per_epoch=100, replace=False, **kwargs):
        """Create a torch Dataset from data. If data is a Feature, it will create a Dataset with the feature.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            return self._maybe_to_channel_first(data)
        
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, torch.utils.data.Dataset):
            return data

        if isinstance(data, (tuple, list)):
            datas = [self.create_data(d) for d in data]
            return torch.utils.data.TensorDataset(*datas)
        
        # check if deeptrack object
        if hasattr(data, "__module__") and data.__module__.startswith("deeptrack"):
            import deeptrack as dt
            if isinstance(data, dt.Feature):
                return dt.pytorch.Dataset(data, 
                                          length=batch_size*steps_per_epoch, replace=replace)
            elif isinstance(data, dt.pytorch.Dataset):
                return data
        
        raise ValueError(f"Data type {type(data)} not supported")
    
    @staticmethod
    def _maybe_to_channel_first(x, other=None):
        
        if not isinstance(x, np.ndarray):
            return x
        
        if x.ndim <= 2:
            return x
        
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return x
        
        return np.moveaxis(x, -1, 1)