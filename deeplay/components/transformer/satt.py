from deeplay import DeeplayModule, Layer

import torch
import torch.nn as nn


class MultiheadSelfAttention(DeeplayModule):
    features: int
    num_heads: int
    projection: nn.Module
    return_attn: bool

    def __init__(
        self,
        features: int,
        num_heads: int,
        projection: nn.Module = nn.Identity(),
        return_attn: bool = False,
        batch_first: bool = False,
    ):
        super().__init__()
        self.features = features
        self.num_heads = num_heads
        self.projection = projection

        self.return_attn = return_attn

        if features <= 0:
            raise ValueError(f"Number of features must be positive, got {features}")

        self.attention = Layer(
            nn.MultiheadAttention, features, num_heads, batch_first=batch_first
        )

    def forward(self, x, batch_index=None):
        """Apply multihead self-attention to the input tensor.
        Returns (y, attn) if return_attn is True, otherwise returns y.
        y is the output of the multihead self-attention layer, attn is the
        attention matrix, and x is the input to the multihead self-attention.
        If projection is nn.Identity, then x is the same as the input to the
        multihead self-attention. Otherwise, x is the output of the projection
        layer.
        """
        attn_mask = None
        if x.ndim == 2:
            if batch_index is None:
                raise ValueError("batch_index must be provided for 2D tensor. Got None")
            attn_mask = self._fetch_attn_mask(batch_index)

        x = self.projection(x)
        y, attn = self.attention(x, x, x, attn_mask=attn_mask)

        if self.return_attn:
            return y, attn
        else:
            return y

    def _fetch_attn_mask(self, batch_index):
        """Fetch attention mask for 2D tensor. The mask is a square matrix with
        True values indicating that the corresponding element is not allowed
        to attend. This is used to deal with unbached sequences of different
        lengths.
        """
        return ~torch.eq(batch_index.unsqueeze(1), batch_index.unsqueeze(0))
