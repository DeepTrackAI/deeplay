import unittest
import torch
import torch.nn as nn
from deeplay import TransformerEncoderLayer, Layer


class TestComponentTransformerEncoder(unittest.TestCase):
    ...

    def test_tel_defaults(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 2)

        for i in range(2):
            self.assertEqual(tel.blocks[i].multihead.layer.attention.embed_dim, 4)
            self.assertEqual(tel.blocks[i].multihead.layer.attention.num_heads, 2)

            self.assertEqual(tel.blocks[i].feed_forward.layer.layer[0].in_features, 4)
            self.assertEqual(tel.blocks[i].feed_forward.layer.layer[-1].out_features, 4)

        # test on a batch of 2
        x = torch.randn(10, 2, 4)
        y = tel(x)
        self.assertEqual(y.shape, (10, 2, 4))

    def test_tel_change_depth(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.configure(hidden_features=[4, 4])
        tel.build()
        self.assertEqual(len(tel.blocks), 3)

    def test_no_hidden_layers(self):
        tel = TransformerEncoderLayer(4, [], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 1)

        self.assertEqual(tel.blocks[0].multihead.layer.attention.embed_dim, 4)
        self.assertEqual(tel.blocks[0].multihead.layer.attention.num_heads, 2)

        self.assertEqual(tel.blocks[0].feed_forward.layer.layer[0].in_features, 4)
        self.assertEqual(tel.blocks[0].feed_forward.layer.layer[-1].out_features, 4)

    def test_variable_hidden_layers(self):
        tel = TransformerEncoderLayer(4, [4, 8, 16], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 4)

        self.assertEqual(tel.blocks[0].multihead.layer.attention.embed_dim, 4)
        self.assertEqual(tel.blocks[1].multihead.layer.attention.embed_dim, 8)
        self.assertEqual(tel.blocks[2].multihead.layer.attention.embed_dim, 16)
        self.assertEqual(tel.blocks[3].multihead.layer.attention.embed_dim, 4)

    def test_tel_multihead_subcomponents(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.multihead[0].layer.configure("return_attn", True)
        tel.build()

        multihead = tel.multihead[0].layer

        # We now evaluate an input tensor of shape (10, 4) with batch index
        # (0, 0, 0, 0, 0, 0, 0, 0, 0, 1). The last element of the batch index
        # indicates that the last element of the input tensor is not allowed
        # to attend to any other element.
        x = torch.randn(10, 4)
        batch_index = torch.zeros(10, dtype=torch.long)
        batch_index[-1] = 1

        y, attn, x = multihead(x, batch_index=batch_index)
        self.assertEqual(y.shape, (10, 4))
        self.assertEqual(attn.shape, (10, 10))
        self.assertEqual(x.shape, (10, 4))

        self.assertEqual(attn.sum(dim=-1).sum(), 10)
        self.assertEqual(attn[-1, -1], 1.0)

        skip = tel.multihead[0].skip
        y1 = skip(y, x)  # as given by a dict mapping
        y2 = skip((y, x))

        self.assertEqual(y1.shape, (10, 4))
        self.assertEqual(y2.shape, (10, 4))
        self.assertEqual((y1 - y2).sum(), 0.0)
