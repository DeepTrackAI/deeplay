from typing import *
from deeplay.applications import Application
from deeplay.activelearning.data import ActiveLearningDataset

import torch
import copy


class Strategy(Application):

    def __init__(
        self,
        train_pool: ActiveLearningDataset,
        val_pool: Optional[ActiveLearningDataset] = None,
        test_data: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_pool = train_pool
        self.val_pool = val_pool
        self.test_data = test_data
        self.initial_model_state: Optional[Dict[str, Any]] = None
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else batch_size
        )

    def on_train_start(self) -> None:
        # Save the initial model state before training
        # such that we can reset the model to its initial state
        # if needed.
        self.initial_model_state = copy.deepcopy(self.state_dict())
        self.train()

        return super().on_train_start()

    def reset_model(self):
        """Reset the model to its initial state.

        This is useful if you want to train the model from scratch
        after querying new samples."""
        assert self.initial_model_state is not None
        self.load_state_dict(self.initial_model_state)
        # self.trainer.strategy.setup_optimizers(self.trainer)

    def query(self, n):
        """Query the strategy for n samples to annotate."""
        if self.val_pool is not None:
            val_pool_frac = len(self.val_pool) / (
                len(self.train_pool) + len(self.val_pool)
            )
            n_val = int(n * val_pool_frac)
            n_train = n - n_val
            return self.query_train(n_train), self.query_val(n_val)
        else:
            return self.query_train(n)

    def query_train(self, n):
        """Query the strategy for n samples to annotate from the training pool."""
        return self.query_strategy(self.train_pool, n)

    def query_val(self, n):
        """Query the strategy for n samples to annotate from the validation pool."""
        return self.query_strategy(self.val_pool, n)

    def query_strategy(self, pool, n):
        """Implement the query strategy here."""
        raise NotImplementedError()

    def query_and_update(self, n):
        """Query the strategy for n samples and update the dataset."""
        self.to(self.trainer.strategy.root_device)
        self.eval()

        indices = self.query(n)
        if isinstance(indices, tuple):
            train_indices, val_indices = indices
            self.train_pool.annotate(train_indices)
            self.val_pool.annotate(val_indices)
        else:
            self.train_pool.annotate(indices)

    def train_dataloader(self):
        data = self.train_pool.get_annotated_data()
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True
        )

    # def val_dataloader(self):
    #     if self.val_pool is None:
    #         return []
    #     data = self.train_pool.get_unannotated_data()
    #     return torch.utils.data.DataLoader(
    #         data, batch_size=self.val_batch_size, shuffle=False
    #     )

    def test_dataloader(self):
        if self.test_data is None:
            return []
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.test_batch_size, shuffle=False
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_metrics(y_hat, y)
        for name, metric in self.test_metrics.items():
            self.log(name, metric)
        # return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.val_metrics(y_hat, y)
        for name, metric in self.val_metrics.items():
            self.log(name, metric)
