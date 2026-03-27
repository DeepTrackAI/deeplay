import copy

import logging
from pickle import PicklingError
from typing import (
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Any,
)
from warnings import warn

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torchmetrics as tm
import tqdm
from torch.nn.modules.module import Module
from torch_geometric.data import Data

import deeplay as dl
from deeplay import DeeplayModule, Optimizer
from deeplay.callbacks import RichProgressBar, LogHistory
import dill

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

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

    def fit(
        self,
        train_data,
        val_data=None,
        max_epochs=None,
        batch_size=32,
        steps_per_epoch=100,
        replace=False,
        val_batch_size=None,
        val_steps_per_epoch=10,
        callbacks=[],
        **kwargs,
    ) -> LogHistory:
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
        val_data = (
            self.create_data(val_data, val_batch_size, val_steps_per_epoch, False)
            if val_data
            else None
        )

        history = LogHistory()
        aux_callbacks = [history]

        callbacks = callbacks + aux_callbacks
        trainer = dl.Trainer(max_epochs=max_epochs, callbacks=callbacks, **kwargs)

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        if not self._has_built:
            self.build()

        self.train()
        if val_data:
            val_dataloader = (
                torch.utils.data.DataLoader(
                    val_data, batch_size=val_batch_size, shuffle=False
                )
                if val_data
                else None
            )
            trainer.fit(self, train_dataloader, val_dataloader)
        else:
            trainer.fit(self, train_dataloader)
        self.eval()
        return history

    def test(
        self,
        data,
        metrics: Union[
            tm.Metric,
            Tuple[str, tm.Metric],
            Sequence[Union[tm.Metric, Tuple[str, tm.Metric]]],
            Dict[str, tm.Metric],
            None,
        ] = None,
        batch_size: int = 32,
        reset_metrics: bool = True,
    ):
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
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

        dict_metrics: Dict[str, tm.Metric]
        if metrics is None:
            return self.trainer.test(self, test_dataloader)[0]

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

        if reset_metrics:
            for value in dict_metrics.values():
                value.reset()

        for x, y in tqdm.tqdm(test_dataloader):
            y_hat = self(x.to(device))
            for metric in dict_metrics.values():
                metric.to(device)
                metric.update(y_hat.to(device), y.to(device))

        out = {name: dict_metrics[name].compute() for name in dict_metrics}

        return out

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

            return self.create_optimizer_with_params(self.optimizer, self.parameters())

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

        assert "loss" in loss, "the output of compute_loss should contain a 'loss' key"

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

        return loss["loss"]

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
        return loss["loss"] if "loss" in loss else 0

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

        return loss["loss"] if "loss" in loss else 0

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        y_hat = self(batch)
        return y_hat

    def log_metrics(
        self, kind: Literal["train", "val", "test"], y_hat, y, **logger_kwargs
    ):
        ys = self.metrics_preprocess(y_hat, y)

        metrics: tm.MetricCollection = getattr(self, f"{kind}_metrics")
        metrics(*ys)

        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                **logger_kwargs,
            )

    def metrics_preprocess(self, y_hat, y) -> Tuple[torch.Tensor, torch.Tensor]:
        return y_hat, y

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
                if isinstance(module, L.LightningModule) or hasattr(module, "trainer"):

                    module.trainer = trainer

            except RuntimeError:
                # hasattr can raise RuntimeError if the module is not attached to a trainer
                if isinstance(module, L.LightningModule):
                    print("Battaching trainer to", module)
                    module.trainer = trainer
                    print("Battached trainer to", module)

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

    def create_data(
        self, data, batch_size=32, steps_per_epoch=100, replace=False, **kwargs
    ):
        """Create a torch Dataset from data. If data is a Feature, it will create a Dataset with the feature."""

        if isinstance(data, dl.DataLoader):
            return data

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if data.dtype in [torch.float16, torch.float, torch.float32, torch.float64]:
                data = data.to(self.dtype)
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
                return dt.pytorch.Dataset(
                    data, length=batch_size * steps_per_epoch, replace=replace
                )
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

    def create_optimizer_with_params(self, optimizer, params):
        if isinstance(optimizer, Optimizer):
            optimizer.configure(params=params)
            return optimizer.create()
        else:
            return optimizer

    def _apply_batch_transfer_handler(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0
    ) -> Any:
        batch = super()._apply_batch_transfer_handler(batch, device, dataloader_idx)
        return self._configure_batch(batch)

    def _configure_batch(self, batch: Any) -> Any:
        if isinstance(batch, (dict, Data)):
            assert "y" in batch, (
                "The batch should contain a 'y' key corresponding to the labels."
                "Found {}".format([key for key, _ in batch.items()])
            )
            self._infer_batch_size_from_batch_indices(batch)
            y = batch.pop("y")
            return batch, y

        return batch

    def _infer_batch_size_from_batch_indices(self, batch):
        if not hasattr(self, "_batch_indices_key"):
            alias = ["batch", "batch_index", "batch_indices"]
            key = next((key for key in alias if key in batch), None)
            if key:
                self._batch_indices_key = key
            else:
                raise ValueError(
                    "The batch should contain a key with the batch indices",
                    "Supported key names are {}".format(alias),
                )

        self._current_batch_size = (
            torch.max(
                batch[self._batch_indices_key],
            ).item()
            + 1
        )

    def log(self, name, value, **kwargs):
        if (not "batch_size" in kwargs) and hasattr(self, "_current_batch_size"):
            kwargs.update({"batch_size": self._current_batch_size})

        super().log(name, value, **kwargs)

    def build(self, *args, **kwargs):
        if self.root_module is self:
            try:
                self._store_hparams(*args, **kwargs)
            except PicklingError:
                warn("Could not store hparams, checkpointing might not be available.")

        return super().build(*args, **kwargs)

    def _store_hparams(self, *args, **kwargs):
        _pickled_application = dill.dumps(self)
        self._set_hparams(
            {
                "__from_ckpt_application": _pickled_application,
                "__build_args": args,
                "__build_kwargs": kwargs,
            }
        )

    # @classmethod
    # def load_from_checkpoint(cls, checkpoint_path: str | Path | np.IO, map_location: torch.device | str | int | Callable[[UntypedStorage, str], UntypedStorage | None] | Dict[torch.device | str | int, torch.device | str | int] | None = None, hparams_file: str | Path | None = None, strict: bool | None = None, **kwargs: Any) -> Self:
    #     return super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
