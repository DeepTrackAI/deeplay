from deeplay.trainer import Trainer
from deeplay.callbacks import LogHistory, RichProgressBar
from deeplay import Regressor, DataLoader
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
import unittest
import torch.nn as nn
import lightning as L
import torch


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        trainer = Trainer(callbacks=[LogHistory()])
        self.assertIsInstance(trainer, Trainer)
        self.assertIsInstance(trainer.callbacks[0], LogHistory)
        self.assertIsInstance(
            trainer.callbacks[1], RichProgressBar
        )  # should be added by default

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
