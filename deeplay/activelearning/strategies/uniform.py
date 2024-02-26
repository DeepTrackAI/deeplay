from deeplay.activelearning.strategies.strategy import Strategy
from deeplay.activelearning.data import ActiveLearningDataset
from deeplay.module import DeeplayModule

import numpy as np
import torch
import torch.nn.functional as F


class UniformStrategy(Strategy):

    def __init__(
        self,
        classifier: DeeplayModule,
        train_pool: ActiveLearningDataset,
        val_pool: ActiveLearningDataset = None,
        test: torch.utils.data.Dataset = None,
        batch_size: int = 32,
        val_batch_size: int = None,
        test_batch_size: int = None,
        **kwargs
    ):
        super().__init__(
            train_pool,
            val_pool,
            test,
            batch_size,
            val_batch_size,
            test_batch_size,
            **kwargs
        )
        self.classifier = classifier

    def query_strategy(self, pool, n):
        """Implement the query strategy here."""
        return np.random.choice(len(pool.get_unannotated_data()), n, replace=False)

    def training_step(self, batch, batch_idx):
        self.train()
        return super().training_step(batch, batch_idx)

    def forward(self, x):
        return self.classifier(x)
