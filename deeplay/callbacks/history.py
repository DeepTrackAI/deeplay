import lightning as L
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Literal


class LogHistory(L.Callback):
    """A keras-like history callback for lightning. Keeps track of metrics and losses during training and validation.

    Example:
    >>> history = LogHistory()
    >>> trainer = dl.Trainer(callbacks=[history])
    >>> trainer.fit(model, train_dataloader, val_dataloader)
    >>> history.history {"train_loss_epoch": {"value": [0.1, 0.2, 0.3], "epoch": [0, 1, 2], "step": [0, 100, 200]}}
    """

    @property
    def history(
        self,
    ) -> Dict[str, Dict[Literal["value", "epoch", "step"], List[Union[float, int]]]]:
        return {
            key: {
                "value": [item["value"] for item in value],
                "epoch": [item["epoch"] for item in value],
                "step": [item["step"] for item in value],
            }
            for key, value in self._history.items()
        }

    @property
    def step_history(
        self,
    ) -> Dict[str, Dict[Literal["value", "epoch", "step"], List[Union[float, int]]]]:
        return {
            key: {
                "value": [item["value"] for item in value],
                "epoch": [item["epoch"] for item in value],
                "step": [item["step"] for item in value],
            }
            for key, value in self._step_history.items()
        }

    def __init__(self):
        self._history = {}
        self._step_history = {}

    def on_train_batch_end(self, trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.endswith("_step"):
                self._step_history.setdefault(key, []).append(
                    self._logitem(trainer, value)
                )

    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))

    def on_validation_epoch_end(self, trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))

    def _logitem(self, trainer, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "value": value,
        }

    def plot(self, *args, yscale="log", **kwargs):
        """Plot the history of the metrics and losses."""

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

        if len(keys) == 4:
            rows = 2
            width = 2

        fig, axes = plt.subplots(rows, width, figsize=(15, 5 * rows))

        if len(keys) == 1:
            axes = np.array([[axes]])

        for ax, key in zip(axes.ravel(), keys):
            train_key = "train" + key
            val_key = "val" + key
            step_key = ("train" + key).replace("epoch", "step")

            if step_key in step_history:
                ax.plot(
                    step_history[step_key]["step"],
                    step_history[step_key]["value"],
                    label=step_key,
                    color="C1",
                    alpha=0.25,
                )
            if train_key in history:
                step = np.array(history[train_key]["step"])
                step[1:] = step[1:] - (step[1:] - step[:-1]) / 2
                step[0] /= 2
                marker_kwargs = (
                    dict(marker="o", markerfacecolor="white", markeredgewidth=1.5)
                    if len(step) < 20
                    else {}
                )
                ax.plot(
                    step,
                    history[train_key]["value"],
                    label=train_key,
                    color="C1",
                    **marker_kwargs,
                )

            if val_key in history:
                marker_kwargs = (
                    dict(marker="d", markerfacecolor="white", markeredgewidth=1.5)
                    if len(step) < 20
                    else {}
                )
                ax.plot(
                    history[val_key]["step"],
                    history[val_key]["value"],
                    label=val_key,
                    color="C3",
                    linestyle="--",
                    **marker_kwargs,
                )

            ax.set_title(
                key.replace("_", " ").replace("epoch", "").strip().capitalize()
            )
            ax.set_xlabel("Step")

            ax.legend()
            ax.set_yscale(yscale)

        return fig, axes
