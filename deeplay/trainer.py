"""Extension of the Lightining Trainer.

This module extends the PyTorch Lightning Trainer class to include additional
functionality for managing callbacks and progress bars. It provides a custom
`_DeeplayCallbackConnector` to configure default callbacks and a `Trainer`
subclass that offers methods to enable or disable specific callbacks like
progress bars and logging.

Key Features
------------
- **Enhanced Callback Connector**

    The `_DeeplayCallbackConnector` class extends the Lightning
    `_CallbackConnector` to add default callbacks such as `TQDMProgressBar` and
    `LogHistory` if they are not explicitly provided.

- **Custom Trainer Class**

    The `Trainer` class extends the Lightning `Trainer` to provide convenience
    methods for enabling and disabling progress bars (`tqdm` or `rich`) and
    log history callbacks.

Module Structure
----------------
Classes:

- `_DeeplayCallbackConnector`: Connector to configure default callbacks.

- `Trainer`: Extended trainer with additional methods for managing callbacks.

Examples
--------
This shows how to use the extended trainer to enable different progress bars:

```python
import deeplay as dl
import torch

# Create training dataset.
num_samples = 10 ** 2
data = torch.randn(num_samples, 2)
labels = (data.sum(dim=1) > 0).long()

dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = dl.DataLoader(dataset, batch_size=16, shuffle=True)

# Create neural network and classifier application.
mlp = dl.MediumMLP(in_features=2, out_features=2)
classifier = dl.Classifier(mlp, optimizer=dl.Adam(), num_classes=2).build()

# Train neural network with progress bar disabled.
trainer = dl.Trainer(max_epochs=100)
trainer.disable_progress_bar()
trainer.fit(classifier, dataloader)

# Return and plot training history.
history = trainer.history
history.plot()

# Retrain with TQDM progress bar enabled.
trainer.tqdm_progress_bar()
trainer.fit(classifier, dataloader)

# Retrain with rich progress bar enabled.
trainer.rich_progress_bar()
trainer.fit(classifier, dataloader)

```

"""

from __future__ import annotations

from lightning import Trainer as pl_Trainer
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.trainer.connectors.callback_connector import (
    _CallbackConnector,
)

from deeplay.callbacks import LogHistory, RichProgressBar, TQDMProgressBar


class _DeeplayCallbackConnector(_CallbackConnector):
    def _configure_progress_bar(
        self: _DeeplayCallbackConnector,
        enable_progress_bar: bool = True,
    ) -> None:

        progress_bars = [
            c for c in self.trainer.callbacks if isinstance(c, ProgressBar)
        ]
        if enable_progress_bar and not progress_bars:
            self.trainer.callbacks.append(TQDMProgressBar())

        # Not great. Should be in a separate configure method. However, this
        # is arguably more stable to api changes in lightning.
        log_histories = [c for c in self.trainer.callbacks if isinstance(c, LogHistory)]
        if not log_histories:
            self.trainer.callbacks.append(LogHistory())

        return super()._configure_progress_bar(enable_progress_bar)


class Trainer(pl_Trainer):

    @property
    def _callback_connector(self: Trainer):
        """Returns the callback connector."""

        return self._callbacks_connector_internal

    @_callback_connector.setter
    def _callback_connector(self: Trainer, value: _CallbackConnector):
        """Sets the callback connector."""

        self._callbacks_connector_internal = _DeeplayCallbackConnector(value.trainer)

    @property
    def history(self: Trainer) -> LogHistory:
        """Returns the history of the training process.

        Returns
        -------
        LogHistory
            The log history callback object.

        Raises
        ------
        ValueError
            If the history callback is not found.

        """

        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                return callback

        raise ValueError("History object not found in callbacks")

    def disable_history(self: Trainer) -> None:
        """Disables the history callback.

        Raises
        ------
        ValueError
            If the history callback is not found.

        """

        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                self.callbacks.remove(callback)
                return

        raise ValueError("History object not found in callbacks")

    def disable_progress_bar(self: Trainer) -> None:
        """Disables the progress bar.

        """

        for callback in self.callbacks:
            if isinstance(callback, ProgressBar):
                self.callbacks.remove(callback)
                return

    def tqdm_progress_bar(self: Trainer, refresh_rate: int = 1) -> None:
        """Enables the TQDM progress bar.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by detault 1. 

        """

        self.disable_progress_bar()

        self.callbacks.append(TQDMProgressBar(refresh_rate=refresh_rate))

    def rich_progress_bar(self, refresh_rate: int = 1, leave: bool = False) -> None:
        """Enables the rich progress bar.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by default 1.
        leave : bool, optional
            Whether to leave the progress bar after completion,
            by default False.

        """

        self.disable_progress_bar()

        self.callbacks.append(
            RichProgressBar(refresh_rate=refresh_rate, leave=leave),
        )
