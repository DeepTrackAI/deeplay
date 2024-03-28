from datetime import timedelta
from typing import Dict, List, Optional, Union

from lightning import Callback, LightningDataModule
from lightning import Trainer as pl_Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from deeplay.callbacks import LogHistory, RichProgressBar


class _DeeplayCallbackConnector(_CallbackConnector):
    def _configure_progress_bar(self, enable_progress_bar: bool = True) -> None:
        progress_bars = [
            c for c in self.trainer.callbacks if isinstance(c, ProgressBar)
        ]
        if enable_progress_bar and not progress_bars:
            self.trainer.callbacks.append(RichProgressBar())

        # Not great. Should be in a separate configure method. However, this
        # is arguably more stable to api changes in lightning.
        log_histories = [c for c in self.trainer.callbacks if isinstance(c, LogHistory)]
        if not log_histories:
            self.trainer.callbacks.append(LogHistory())

        return super()._configure_progress_bar(enable_progress_bar)


class Trainer(pl_Trainer):

    @property
    def _callback_connector(self):
        return self._callbacks_connector_internal

    @_callback_connector.setter
    def _callback_connector(self, value: _CallbackConnector):
        self._callbacks_connector_internal = _DeeplayCallbackConnector(value.trainer)

    @property
    def history(self) -> LogHistory:
        """Returns the history of the training process."""
        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                return callback
        raise ValueError("History object not found in callbacks")

    def disable_history(self) -> None:
        """Disables the history callback."""
        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                self.callbacks.remove(callback)
                return
        raise ValueError("History object not found in callbacks")
