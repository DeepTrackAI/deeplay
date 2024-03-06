from deeplay import DeeplayModule, Layer

import math

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class PositionalEmbeddingBaseClass(DeeplayModule):
    def __init__(self, features, dropout_p=0.0, max_length=5000):
        super().__init__()
        self.features = features
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.compute_and_register_embeddings(features, max_length)
        self.dropout = Layer(nn.Dropout, dropout_p)

    def compute_and_register_embeddings(
        self,
        features,
        max_lenght,
    ):
        inv_freq = 1 / (10000 ** (torch.arange(0, features, 2).float() / features))
        positions = torch.arange(0, max_lenght).unsqueeze(1).float()
        sinusoid_inp = positions * inv_freq.unsqueeze(0)
        pos_emb = torch.zeros(max_lenght, features)
        pos_emb[:, 0::2] = torch.sin(sinusoid_inp)
        pos_emb[:, 1::2] = torch.cos(sinusoid_inp)

        self.embs = nn.Parameter(pos_emb, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "forward method not implemented for {}".format(self.__class__.__name__)
        )


class BatchedPositionalEmbedding(PositionalEmbeddingBaseClass):
    def __init__(
        self,
        features,
        dropout_p=0.0,
        max_length=5000,
        batch_first=True,
    ):
        super().__init__(features, dropout_p, max_length)

        self.batch_first = batch_first
        self.embs = nn.Parameter(
            self.embs.unsqueeze(0 if batch_first else 1), requires_grad=False
        )

    def forward(self, x):
        x = x + self.embs[: x.size(0)]
        return self.dropout(x)


class IndexedPositionalEmbedding(PositionalEmbeddingBaseClass):
    def __init__(
        self,
        features,
        dropout_p=0.0,
        max_length=5000,
    ):
        super().__init__(features, dropout_p, max_length)

    def fetch_embeddings(self, batch_index):
        """
        This method takes an array of batch indices and returns
        an array of the same size where each element is replaced
        by its relative index within its batch.

        Example:
        batch_index = [0, 0, 1, 1, 1, 2, 2]

        fetch_embeddings(batch_index) -> [0, 1, 0, 1, 2, 0, 1]
        """
        diff = torch.cat(
            (torch.ones(1, dtype=torch.int64), batch_index[1:] - batch_index[:-1])
        )
        change_points = diff.nonzero().squeeze()

        sizes = torch.diff(
            torch.cat(
                (change_points, torch.tensor([len(batch_index)], dtype=torch.int64))
            )
        )
        indices = torch.arange(len(batch_index))
        relative_indices = indices - torch.repeat_interleave(change_points, sizes)

        return self.embs[relative_indices]

    def forward(self, x, batch_index):
        x = x + self.fetch_embeddings(batch_index)
        return self.dropout(x)
