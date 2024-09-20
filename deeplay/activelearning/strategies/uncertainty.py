from typing import Optional
from deeplay.activelearning.strategies.strategy import Strategy
from deeplay.activelearning.data import ActiveLearningDataset
from deeplay.activelearning.criterion import ActiveLearningCriterion
from deeplay.module import DeeplayModule
from deeplay.external.optimizers import Adam

import torch
import torch.nn.functional as F


class UncertaintyStrategy(Strategy):

    def __init__(
        self,
        classifier: DeeplayModule,
        criterion: ActiveLearningCriterion,
        train_pool: ActiveLearningDataset,
        val_pool: Optional[ActiveLearningDataset] = None,
        test: torch.utils.data.Dataset = None,
        batch_size: int = 32,
        val_batch_size: int = None,
        test_batch_size: int = None,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        **kwargs
    ):

        optimizer = optimizer or Adam(lr=1e-3)
        super().__init__(
            train_pool,
            val_pool,
            test,
            batch_size,
            val_batch_size,
            test_batch_size,
            loss=loss,
            optimizer=optimizer,
            **kwargs
        )
        self.classifier = classifier
        self.criterion = criterion

    def query_strategy(self, pool, n):
        """Implement the query strategy here."""
        self.eval()
        X = pool.get_unannotated_samples()

        latents = self.classifier.predict(X, batch_size=self.test_batch_size)
        probs = F.softmax(latents, dim=1)

        return self.criterion.score(probs).sort()[1][:n]

    def training_step(self, batch, batch_idx):
        self.train()
        return super().training_step(batch, batch_idx)

    def forward(self, x):
        return self.classifier(x)
