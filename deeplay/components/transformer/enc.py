from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from .. import DeeplayModule, Layer, LayerList, MultiLayerPerceptron
from . import LayerDropoutSkipNormalization, MultiheadSelfAttention

from deeplay.blocks.sequential import SequentialBlock
from deeplay import Sequential

import torch
import torch.nn as nn

import itertools
from functools import reduce

__all__ = ["TransformerEncoderLayer"]


class Add(DeeplayModule):
    """Addition module.

    Adds input tensors element-wise.

    Examples
    --------
    >>> add = Add()
    >>> add(torch.randn(4, 4), torch.randn(4, 4))
    """

    def forward(self, *x):
        x = itertools.chain(*x) if isinstance(x[0], tuple) else x
        return reduce(lambda a, b: torch.add(a, b), x)


class TransformerEncoderLayer(DeeplayModule):
    """Transformer encoder module.

    Configurables
    -------------
    - in_features (int): Number of input features. If None, the input shape is inferred in the first forward pass. (Default: None)
    - hidden_features (list[int]): Number of hidden units in each layer.
    - out_features (int): Number of output features.
    - num_heads (int): Number of attention heads.

    Shorthands
    ----------
    - `input`: Equivalent to `.blocks[0]`.
    - `hidden`: Equivalent to `.blocks[:-1]`.
    - `output`: Equivalent to `.blocks[-1]`.
    - `multihead`: Equivalent to `.blocks.multihead`.
    - `feed_forward`: Equivalent to `.blocks.feed_forward`.

    Evaluation
    ----------
    >>> for block in tel.blocks:
    >>>    x = block(x)
    >>> return x

    Examples
    --------
    >>> tel = TransformerEncoderLayer(4, [4, 16], 4, 2)
    >>> tel.build()
    >>> seq_len, batch_size,features = 2, 10, 4
    >>> input_seq = torch.randn(seq_len, batch_size, features)
    >>> tel(input_seq).shape
    torch.Size([2, 10, 4])


    Return Values
    -------------
    The forward method returns the processed tensor.
    """

    input_features: int
    hidden_features: Sequence[Optional[int]]
    out_features: int
    num_heads: int
    blocks: LayerList[Sequential]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    @property
    def multihead(self):
        """Return the multihead attention layer of the network. Equivalent to
        `.blocks.multihead.layer`."""
        return self.blocks.multihead

    @property
    def feed_forward(self):
        """Return the feed forward layer of the network. Equivalent to
        `.blocks.feed_forward`."""
        return self.blocks.feed_forward

    def __init__(
        self,
        in_features: Optional[int],
        hidden_features: Sequence[Optional[int]],
        out_features: int,
        num_heads: int,
        batch_first: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.batch_first = batch_first

        if out_features <= 0:
            raise ValueError(
                f"Number of output features must be positive, got {out_features}"
            )

        if in_features is not None and in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )

        self.blocks = LayerList()

        for i, (f_in, f_out) in enumerate(
            zip([in_features, *hidden_features], [*hidden_features, out_features])
        ):
            self.blocks.append(
                SequentialBlock(
                    # First sub-block is multihead self-attention followed by
                    # skip connection and normalization.
                    multihead=LayerDropoutSkipNormalization(
                        layer=MultiheadSelfAttention(
                            f_out,
                            num_heads,
                            projection=(
                                Layer(nn.Linear, f_in, f_out)
                                if f_in != f_out
                                else Layer(nn.Identity)
                            ),
                            batch_first=batch_first,
                        ),
                        dropout=Layer(nn.Dropout, 0),
                        skip=Add(),
                        normalization=Layer(nn.LayerNorm, f_out),
                    ),
                    # Second sub-block is feed forward followed by skip connection
                    # and normalization
                    feed_forward=LayerDropoutSkipNormalization(
                        layer=MultiLayerPerceptron(
                            f_out, [f_out], f_out, flatten_input=False
                        ),
                        dropout=Layer(nn.Dropout, 0),
                        skip=Add(),
                        normalization=Layer(nn.LayerNorm, f_out),
                    ),
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[List[int]] = None,
        out_features: Optional[int] = None,
        num_heads: Optional[int] = None,
    ) -> None: ...

    configure = DeeplayModule.configure
