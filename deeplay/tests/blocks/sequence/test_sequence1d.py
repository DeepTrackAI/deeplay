import unittest
from itertools import product

import torch
import torch.nn as nn

from deeplay.blocks.sequence import Sequence1dBlock
from deeplay.external.layer import Layer

from deeplay.ops.attention.self import MultiheadSelfAttention
from deeplay.ops.merge import Add


class TestConv2dBlock(unittest.TestCase):

    def test_init(self):
        block = Sequence1dBlock(1, 1)
        self.assertListEqual(block.order, ["layer"])

    def test_lstm_bidirectional(self):
        block = Sequence1dBlock(1, 2).LSTM().bidirectional().build()
        x = torch.randn(10, 1, 1)
        y = block(x)
        self.assertEqual(y.shape, (10, 1, 2))

    def test_gru_bidirectional(self):
        block = Sequence1dBlock(1, 2).GRU().bidirectional().build()
        x = torch.randn(10, 1, 1)
        y = block(x)
        self.assertEqual(y.shape, (10, 1, 2))

    def test_rnn_bidirectional(self):
        block = Sequence1dBlock(1, 2).RNN().bidirectional().build()
        x = torch.randn(10, 1, 1)
        y = block(x)
        self.assertEqual(y.shape, (10, 1, 2))
