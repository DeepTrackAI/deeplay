"""Enhanced progress bars for compatibility and customization.

This module defines enhanced progress bars for PyTorch Lightning, designed to
improve compatibility and usability in various environments.
The `TQDMProgressBar` and `RichProgressBar` classes extend the Lightning
default implementations and provide safe refresh rate handling for platforms
like Google Colab and Kaggle, which may crash with small refresh rates.

Key Features
------------
- **TQDM Progress Bar with Compatibility Enhancements**

    The `TQDMProgressBar` class extends Lightning's `TQDMProgressBar`,
    providing a mechanism to adjust refresh rates based on the execution
    environment. This helps avoid issues caused by small refresh rates on Colab
    and Kaggle.

- **Rich Progress Bar with Customization Options**

    The `RichProgressBar` class offers a visually appealing progress bar with
    customizable themes and console options. Similar to the `TQDMProgressBar`,
    it includes environment-based refresh rate adjustments to enhance
    stability.

Module Structure
----------------
Classes:

- `TQDMProgressBar`: Enhances Lightning TQDM progress bar.

    Automatically modifies the refresh rate if the code is executed on
    platforms like Colab or Kaggle.

- `RichProgressBar`: Enhances Lightning Rich progress bar.

    Supports configurable themes and console settings, and adjusts refresh
    rates when needed.

Examples
--------
This example demosntrate the use of the standard TQDM progress bar:

```python
import deeplay as dl
import torch

# Create training dataset.
num_samples = 10 ** 4
data = torch.randn(num_samples, 2)
labels = (data.sum(dim=1) > 0).long()

dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = dl.DataLoader(dataset, batch_size=16, shuffle=True)

# Create neural network and classifier application.
mlp = dl.MediumMLP(in_features=2, out_features=2)
classifier = dl.Classifier(mlp, optimizer=dl.Adam(), num_classes=2).build()

# Train neural network with progress bar.
tqdm_bar = dl.callbacks.TQDMProgressBar(refresh_rate=100)
trainer = dl.Trainer(max_epochs=100, callbacks=[tqdm_bar])
trainer.fit(classifier, dataloader)
```

Alternatively, you can use the rich progress bar with:

```python
rich_bar = dl.callbacks.RichProgressBar(refresh_rate=100)
trainer = dl.Trainer(max_epochs=100, callbacks=[rich_bar])
trainer.fit(classifier, dataloader)
```

"""

from __future__ import annotations

import os

from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar as LightningRichProgressBar,
    RichProgressBarTheme as RPBTheme,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import (
    TQDMProgressBar as LightningTQDMProgressBar,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_debug


class TQDMProgressBar(LightningTQDMProgressBar):
    """A progress bar for displaying training progress with TQDM.

    This class enhances the standard Lightning TQDMProgressBar by providing
    environment-specific adjustments to prevent potential crashes on platforms
    like Colab and Kaggle.

    Parameters
    ----------
    refresh_rate : int, optional
        The refresh rate of the progress bar, by default 1.

    Example
    -------
    This example demosntrate the use of the standard TQDM progress bar:

    ```python
    import deeplay as dl
    import torch

    # Create training dataset.
    num_samples = 10 ** 4
    data = torch.randn(num_samples, 2)
    labels = (data.sum(dim=1) > 0).long()

    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = dl.DataLoader(dataset, batch_size=16, shuffle=True)

    # Create neural network and classifier application.
    mlp = dl.MediumMLP(in_features=2, out_features=2)
    classifier = dl.Classifier(mlp, optimizer=dl.Adam(), num_classes=2).build()

    # Train neural network with progress bar.
    tqdm_bar = dl.callbacks.TQDMProgressBar(refresh_rate=100)
    trainer = dl.Trainer(max_epochs=100, callbacks=[tqdm_bar])
    trainer.fit(classifier, dataloader)
    ```

    """

    def __init__(
        self: TQDMProgressBar,
        refresh_rate: int = 1,
    ):
        """Initialize the progress bar with a configurable refresh rate.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by default 1.

        """

        super().__init__(refresh_rate=refresh_rate)

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        """Resolve refresh rate for compatibility with Colab and Kaggle.

        This method adjusts the refresh rate to a safe value to prevent crashes
        on platforms that are known to have issues with small refresh rates.

        Parameters
        ----------
        refresh_rate : int
            The desired refresh rate of the progress bar.

        Returns
        -------
        int
            The adjusted refresh rate.

        """

        # This should work both for Colab and Kaggle because Kaggle returns a
        # Colab session.
        if "COLAB_JUPYTER_IP" in os.environ and refresh_rate == 1:
            rank_zero_debug(
                "Small refresh rates can crash on Colab or Kaggle. "
                "Setting refresh_rate to 10.\n"
                "To manually set the refresh rate, "
                "call `trainer.tqdm_progress_bar(refresh_rate=10)`."
            )
            refresh_rate = 10

        return LightningTQDMProgressBar._resolve_refresh_rate(refresh_rate)


class RichProgressBar(LightningRichProgressBar):
    """A progress bar for displaying training progress with Rich.

    This class enhances the standard Lightning RichProgressBar by supporting
    customizable themes and console options. It includes an
    environment-specific adjustment to prevent potential crashes on platforms
    like Colab and Kaggle.

    Parameters
    ----------
    refresh_rate : int, optional
        The refresh rate of the progress bar, by default 1.
    leave : bool, optional
        Whether to leave the progress bar on the screen after completion,
        by default False.
    theme : RichProgressBarTheme, optional
        The theme used for the Rich progress bar,
        by default `RichProgressBarTheme(metrics_format=".3g")`.
    console_kwargs : dict, optional
        Additional keyword arguments for configuring the Rich console,
        by default None.

    Example
    -------
    This example demosntrate the use of the standard TQDM progress bar:

    ```python
    import deeplay as dl
    import torch

    # Create training dataset.
    num_samples = 10 ** 4
    data = torch.randn(num_samples, 2)
    labels = (data.sum(dim=1) > 0).long()

    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = dl.DataLoader(dataset, batch_size=16, shuffle=True)

    # Create neural network and classifier application.
    mlp = dl.MediumMLP(in_features=2, out_features=2)
    classifier = dl.Classifier(mlp, optimizer=dl.Adam(), num_classes=2).build()

    # Train neural network with progress bar.
    rich_bar = dl.callbacks.RichProgressBar(refresh_rate=100)
    trainer = dl.Trainer(max_epochs=100, callbacks=[rich_bar])
    trainer.fit(classifier, dataloader)
    ```

    """

    def __init__(
        self: RichProgressBar,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RPBTheme = RPBTheme(metrics_format=".3g"),
        console_kwargs=None,
    ):
        """Initialize the Rich progress bar with customizable settings.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by default 1.
        leave : bool, optional
            Whether to leave the progress bar displayed after completion,
            by default False.
        theme : RichProgressBarTheme, optional
            The theme of the progress bar,
            by default `RPBTheme(metrics_format=".3g")`.
        console_kwargs : dict, optional
            Additional keyword arguments to configure the Rich console,
            by default None.

        """

        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme,
            console_kwargs=console_kwargs,
        )

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        """Resolve refresh rate for compatibility with Colab and Kaggle.

        This method adjusts the refresh rate to a safe value to prevent crashes
        on platforms that are known to have issues with small refresh rates.

        Parameters
        ----------
        refresh_rate : int
            The desired refresh rate of the progress bar.

        Returns
        -------
        int
            The adjusted refresh rate.

        """

        # This should work both for Colab and Kaggle because Kaggle returns a
        # Colab session.
        if "COLAB_JUPYTER_IP" in os.environ and refresh_rate == 1:
            rank_zero_debug(
                "Small refresh rates can crash on Colab or Kaggle. "
                "Setting refresh_rate to 10.\n"
                "To manually set the refresh rate, "
                "call `trainer.rich_progress_bar(refresh_rate=10)`."
            )
            refresh_rate = 10

        return LightningTQDMProgressBar._resolve_refresh_rate(refresh_rate)
