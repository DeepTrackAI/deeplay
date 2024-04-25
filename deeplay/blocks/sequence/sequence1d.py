from __future__ import annotations
from typing import Literal, Type, Union
from typing_extensions import Self
import warnings
import torch.nn as nn

from deeplay.blocks.base import BaseBlock
from deeplay.external import Layer
from deeplay.module import DeeplayModule


class SequenceDropout(nn.Module):
    """Dropout layer for sequences.

    Ensures that the dropout mask is not applied to the hidden state.
    Also works with packed sequences. If input data is a tensor, the dropout is applied as usual.

    Parameters
    ----------
    p: float
        Probability of an element to be zeroed. Default: 0.0

    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p=self.p)

    def forward(self, x):
        if not isinstance(x, tuple):
            if isinstance(x, nn.utils.rnn.PackedSequence):
                return nn.utils.rnn.PackedSequence(self.dropout(x.data), x.batch_sizes)
            else:
                return self.dropout(x)

        if isinstance(x[0], nn.utils.rnn.PackedSequence):
            return (
                nn.utils.rnn.PackedSequence(self.dropout(x[0].data), x[0].batch_sizes),
                x[1],
            )
        return self.dropout(x[0]), x[1]


class Sequence1dBlock(BaseBlock):
    """Convolutional block with optional normalization and activation."""

    @property
    def is_recurrent(self):
        return self.mode in ["LSTM", "GRU", "RNN"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_first: bool = False,
        mode: Literal["LSTM", "GRU", "RNN", "attention"] = "LSTM",
        return_cell_state: bool = False,
        **kwargs,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.batch_first = batch_first

        cls = dict(
            LSTM=nn.LSTM, GRU=nn.GRU, RNN=nn.RNN, attention=nn.MultiheadAttention
        )[mode]
        if cls == nn.MultiheadAttention:
            raise NotImplementedError("Attention not implemented yet")

        self.mode = mode
        self.return_cell_state = return_cell_state

        layer = Layer(
            cls,
            in_features,
            out_features,
            batch_first=batch_first,
        )

        super().__init__(layer=layer, **kwargs)

    def normalized(
        self,
        normalization: Union[Type[nn.Module], DeeplayModule] = nn.LayerNorm,
        mode="append",
        after=None,
    ) -> Self:
        did_replace = mode == "replace" and "normalization" in self.order

        super().normalized(normalization, mode=mode, after=after)

        if did_replace:
            # Assume num_features is already correct
            return self

        idx = self.order.index("normalization")
        # if layer or blocks before normalization
        if any(name in self.order[:idx] for name in ["layer", "blocks"]):
            channels = self.out_features
        else:
            channels = self.in_features

        self._configure_normalization(channels)

        return self

    def _configure_normalization(self, channels):
        ntype: Type[nn.Module] = self.normalization.classtype

        if ntype == nn.BatchNorm1d:
            self.normalization.configure(num_features=channels)
        elif ntype == nn.GroupNorm:
            num_groups = self.normalization.kwargs.get("num_groups", 1)
            self.normalization.configure(num_groups=num_groups, num_channels=channels)
        elif ntype == nn.InstanceNorm1d:
            self.normalization.configure(num_features=channels)
        elif ntype == nn.LayerNorm:
            self.normalization.configure(normalized_shape=channels)

    def LSTM(self):
        self.configure(mode="LSTM")
        self.layer.configure(nn.LSTM)
        return self

    def GRU(self):
        self.configure(mode="GRU")
        self.layer.configure(nn.GRU)
        return self

    def RNN(self):
        self.configure(mode="RNN")
        self.layer.configure(nn.RNN)
        return self

    def bidirectional(self):
        self.layer.configure(bidirectional=True)
        return self

    def activated(
        self,
        activation: Type[nn.Module] | DeeplayModule = nn.ReLU,
        mode="append",
        after=None,
    ) -> Self:
        return super().activated(activation, mode, after)

    def append_dropout(self, p: float, name: str | None = "dropout"):
        self.append(Layer(SequenceDropout, p), name=name)
        return self

    def prepend_dropout(self, p: float, name: str | None = "dropout"):
        self.prepend(Layer(SequenceDropout, p), name=name)
        return self

    def insert_dropout(self, p: float, after: str, name: str | None = "dropout"):
        self.insert(Layer(SequenceDropout, p), after=after, name=name)
        return self

    def forward(self, x):
        # TODO: refactor to use super class forward
        hx = None
        for name in self.order:
            block = getattr(self, name)
            if name == "layer":
                x = block(x)
                if self.is_recurrent:
                    x, hx = x
                    if block.bidirectional:
                        x = x[:, :, : self.out_features] + x[:, :, self.out_features :]
            elif name == "shortcut_start":
                shortcut = block(x)
            elif name == "shortcut_end":
                x = block(x, shortcut)
            else:
                x = block(x)
        if self.return_cell_state:
            return x, hx
        return x
