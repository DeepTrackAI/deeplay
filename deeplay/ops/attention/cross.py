from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from deeplay.external.layer import Layer
from deeplay.module import DeeplayModule


class MultiheadCrossAttention(DeeplayModule):

    features: int
    num_heads: int
    return_attn: bool

    def __init__(
        self,
        features: int,
        num_heads: int,
        return_attn: bool = False,
        queries: int | str | nn.Module = 0,
        keys: int | str | nn.Module = 1,
        values: int | str | nn.Module = 2,
        batch_first: bool = False,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()
        self.features = features
        self.num_heads = num_heads

        self.queries = queries
        self.keys = keys
        self.values = values

        self.return_attn = return_attn
        self.batch_first = batch_first

        if features <= 0:
            raise ValueError(f"Number of features must be positive, got {features}")

        self.attention = nn.MultiheadAttention(
            features,
            num_heads,
            batch_first=batch_first,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def _parse_query_key_value(self, x):
        if isinstance(x, dict):
            return x[self.queries], x[self.keys], x[self.values]
        if isinstance(x, torch.Tensor):
            return x, x, x
        if len(x) == 1 and isinstance(x[0], tuple):
            return self._parse_query_key_value(x[0])
        if len(x) == 1 and isinstance(x[0], dict):
            return self._parse_query_key_value(x[0])
        return x[self.queries], x[self.keys], x[self.values]

    def forward(self, *x, batch_index=None):
        """Apply multihead self-attention to the input tensor.
        Returns (y, attn) if return_attn is True, otherwise returns y.
        y is the output of the multihead self-attention layer, attn is the
        attention matrix, and x is the input to the multihead self-attention.
        If projection is nn.Identity, then x is the same as the input to the
        multihead self-attention. Otherwise, x is the output of the projection
        layer.
        """

        q, k, v = self._parse_query_key_value(x)

        attn_mask = None
        if q.ndim == 2:
            if batch_index is None:
                raise ValueError("batch_index must be provided for 2D tensor. Got None")
            attn_mask = self._fetch_attn_mask(batch_index)

        start_shape = v.shape

        if v.ndim > 3:
            if self.batch_first:
                q = q.view(start_shape[0], -1, start_shape[-1])
                k = k.view(start_shape[0], -1, start_shape[-1])
                v = v.view(start_shape[0], -1, start_shape[-1])
            else:
                q = q.view(-1, start_shape[-2], start_shape[-1])
                k = k.view(-1, start_shape[-2], start_shape[-1])
                v = v.view(-1, start_shape[-2], start_shape[-1])

        y, attn = self.attention(q, k, v, attn_mask=attn_mask)

        if len(start_shape) > 3:
            y = y.view(start_shape)

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
