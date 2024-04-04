from deeplay import DeeplayModule, Layer

import math

import torch
import torch.nn as nn

from typing import Callable


def sinusoidal_init_(tensor: torch.Tensor):
    """
    Initialize tensor with sinusoidal positional embeddings.
    """
    lenght, features = tensor.shape

    inv_freq = 1 / (10000 ** (torch.arange(0, features, 2).float() / features))
    positions = torch.arange(0, lenght).unsqueeze(1).float()
    sinusoid_inp = positions * inv_freq.unsqueeze(0)

    with torch.no_grad():
        tensor[:, 0::2] = torch.sin(sinusoid_inp)
        tensor[:, 1::2] = torch.cos(sinusoid_inp)

        return tensor


class PositionalEmbedding(DeeplayModule):
    def __init__(
        self,
        features: int,
        max_length: int = 5000,
        initializer: Callable = sinusoidal_init_,
        learnable: bool = False,
        batch_first: bool = False,
    ):
        super().__init__()

        self.features = features
        self.max_length = max_length
        self.learnable = learnable
        self.batch_first = batch_first

        self.batched_dim = 0 if batch_first else 1
        init_embs = initializer(torch.empty(max_length, features)).unsqueeze(
            self.batched_dim
        )
        self.embs = nn.Parameter(init_embs, requires_grad=learnable)

        self.dropout = Layer(nn.Dropout, 0)

    def forward(self, x):
        seq_dim = 1 - self.batched_dim
        x = x + torch.narrow(self.embs, dim=seq_dim, start=0, length=x.size(seq_dim))
        return self.dropout(x)


class IndexedPositionalEmbedding(PositionalEmbedding):
    def __init__(
        self,
        features,
        max_length=5000,
        initializer: Callable = sinusoidal_init_,
        learnable: bool = False,
    ):
        super().__init__(features, max_length, initializer, learnable)

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
            (
                torch.ones(1, dtype=torch.int64, device=batch_index.device),
                batch_index[1:] - batch_index[:-1],
            )
        )
        change_points = diff.nonzero().squeeze().flatten()

        sizes = torch.diff(
            torch.cat(
                (
                    change_points,
                    torch.tensor(
                        [len(batch_index)], dtype=torch.int64, device=batch_index.device
                    ),
                )
            )
        )
        indices = torch.arange(len(batch_index), device=batch_index.device)
        relative_indices = indices - torch.repeat_interleave(change_points, sizes)

        return self.embs[relative_indices, 0]

    def forward(self, x, batch_index):
        x = x + self.fetch_embeddings(batch_index)
        return self.dropout(x)
