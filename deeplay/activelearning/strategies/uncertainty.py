from deeplay.activelearning.strategies.strategy import ActiveLearningStrategy
from deeplay.activelearning.data import ActiveLearningDataset, JointDataset
from deeplay.activelearning.criterion import ActiveLearningCriteria
from deeplay.module import DeeplayModule


import torch
import torch.nn.functional as F


class UncertaintyActiveLearning(ActiveLearningStrategy):

    def __init__(
        self,
        classifier: DeeplayModule,
        criteria: ActiveLearningCriteria,
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
        self.criteria = criteria


    def query_strategy(self, pool, n):
        """Implement the query strategy here."""
        self.eval()
        X = pool.get_unannotated_samples()

        latents = self.classifier.predict(X, batch_size=self.test_batch_size)
        probs = F.softmax(latents, dim=1)   

        return self.criteria.score(probs).sort()[1][:n]

    def training_step(self, batch, batch_idx):
        self.train()
        return super().training_step(batch, batch_idx)

    def forward(self, x):
        return self.classifier(x)