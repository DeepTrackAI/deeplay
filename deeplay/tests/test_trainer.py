from deeplay import DataLoader, Regressor
from deeplay.callbacks import LogHistory, RichProgressBar

from deeplay import Regressor, DataLoader
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
import unittest
import torch.nn as nn

from deeplay.trainer import Trainer

import lightning as L
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
import torch
import torch.nn as nn
import unittest
from unittest.mock import patch


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self._patcher = patch("torch.backends.mps.is_available",
                              return_value=False)
        self._mock = self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_trainer(self):
        trainer = Trainer(callbacks=[LogHistory()])
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)
        self.assertIsInstance(
            trainer.callbacks[1], TQDMProgressBar
        )  # should be added by default

    def test_trainer(self):
        trainer = Trainer(callbacks=[LogHistory()])
        trainer.disable_progress_bar()
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)

        has_a_progress_bar = False
        for callback in trainer.callbacks:
            if isinstance(callback, ProgressBar):
                has_a_progress_bar = True
                break
        self.assertFalse(has_a_progress_bar)

    def test_trainer_tqdm_progress_bar(self):
        trainer = Trainer(callbacks=[LogHistory()])
        trainer.tqdm_progress_bar()
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)

        num_progress_bars = 0
        for callback in trainer.callbacks:
            if isinstance(callback, ProgressBar):
                num_progress_bars += 1
                self.assertIsInstance(callback, TQDMProgressBar)

        self.assertEqual(num_progress_bars, 1)

    def test_trainer_rich_progress_bar(self):
        trainer = Trainer(callbacks=[LogHistory()])
        trainer.rich_progress_bar()
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)

        num_progress_bars = 0
        for callback in trainer.callbacks:
            if isinstance(callback, ProgressBar):
                num_progress_bars += 1
                self.assertIsInstance(callback, RichProgressBar)

        self.assertEqual(num_progress_bars, 1)

    def test_trainer_explicit_progress_bar(self):
        trainer = Trainer(callbacks=[LogHistory(), RichProgressBar()])
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)
        self.assertIsInstance(trainer.callbacks[1], RichProgressBar)

    def test_trainer_explicit_progress_bar_tqdm(self):
        trainer = Trainer(callbacks=[LogHistory(), TQDMProgressBar()])
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)
        self.assertIsInstance(trainer.callbacks[1], TQDMProgressBar)

    def test_trainer_implicit_progress_bar_and_disable(self):
        trainer = Trainer(callbacks=[LogHistory()], enable_progress_bar=False)
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)

    def test_fit(self):
        trainer = Trainer(max_epochs=1)
        model = Regressor(nn.Linear(1, 1))
        X = torch.rand(100, 1)
        y = torch.rand(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        train_dataloader = DataLoader(dataset, batch_size=32)
        val_dataloader = DataLoader(dataset, batch_size=32)

        trainer.fit(model, train_dataloader, val_dataloader)

        self.assertIsInstance(trainer.history, LogHistory)
        keys = set(trainer.history.history.keys())
        self.assertTrue("train_loss_epoch" in keys)
        self.assertTrue("val_loss_epoch" in keys)
        step_keys = set(trainer.history.step_history.keys())
        self.assertTrue("train_loss_step" in step_keys)

    def test_fit_disabled_history(self):
        trainer = Trainer(max_epochs=1)
        trainer.disable_history()

        model = Regressor(nn.Linear(1, 1))
        X = torch.rand(100, 1)
        y = torch.rand(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        train_dataloader = DataLoader(dataset, batch_size=32)
        val_dataloader = DataLoader(dataset, batch_size=32)

        trainer.fit(model, train_dataloader, val_dataloader)

        with self.assertRaises(ValueError):
            trainer.history
