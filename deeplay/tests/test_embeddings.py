import unittest

import torch
from deeplay import PositionalEmbedding, IndexedPositionalEmbedding


class TestPos(unittest.TestCase):
    ...

    def test_positional_embedding_default(self):
        layer = PositionalEmbedding(96)
        layer.build()

        # i.e., [batch_size, max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))

        # By default, dropout is 0.0
        self.assertEqual(layer.dropout.p, 0.0)

        # i.e., [batch_size, seq_length, features]
        x = torch.randn(10, 100, 96)
        y = layer(x)

        self.assertEqual(y.shape, (10, 100, 96))

    def test_positional_embedding_batch_first(self):
        layer = PositionalEmbedding(96, batch_first=False)
        layer.build()

        # i.e., [max_length, 1, features]
        self.assertEqual(layer.embs.shape, (5000, 1, 96))

        # i.e., [seq_length, 1, features]
        x = torch.randn(100, 10, 96)
        y = layer(x)

        self.assertEqual(y.shape, (100, 10, 96))

    def test_positional_embedding_learnable(self):
        layer = PositionalEmbedding(96, learnable=True)
        layer.build()

        # i.e., [batch_size, max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))
        self.assertTrue(layer.embs.requires_grad)

    def test_positional_embedding_initializer(self):
        layer = PositionalEmbedding(96, initializer=torch.nn.init.zeros_)
        layer.build()

        # i.e., [batch_size, max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))
        self.assertTrue(torch.all(layer.embs == 0.0))

        layer = PositionalEmbedding(96)
        layer.configure(initializer=torch.nn.init.zeros_)
        layer.build()

        # i.e., [batch_size, max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))
        self.assertTrue(torch.all(layer.embs == 0.0))

    def test_indexed_positional_embedding_default(self):
        layer = IndexedPositionalEmbedding(96)
        layer.build()

        # i.e., [max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))

        # By default, dropout is 0.0
        self.assertEqual(layer.dropout.p, 0.0)

    def test_indexed_positional_embedding_fetch(self):
        layer = IndexedPositionalEmbedding(96)
        layer.build()

        batch_indices = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        pembs = layer.fetch_embeddings(batch_indices)

        # i.e., [num_indices, features]
        self.assertEqual(pembs.shape, (10, 96))

        self.assertTrue(torch.all(pembs[0] == layer.embs[0, 0]))
        self.assertTrue(torch.all(pembs[1] == layer.embs[0, 1]))
        self.assertTrue(torch.all(pembs[2] == layer.embs[0, 0]))
        self.assertTrue(torch.all(pembs[3] == layer.embs[0, 1]))
        self.assertTrue(torch.all(pembs[4] == layer.embs[0, 2]))
        self.assertTrue(torch.all(pembs[5] == layer.embs[0, 3]))
        self.assertTrue(torch.all(pembs[6] == layer.embs[0, 0]))
        self.assertTrue(torch.all(pembs[7] == layer.embs[0, 1]))
        self.assertTrue(torch.all(pembs[8] == layer.embs[0, 2]))
        self.assertTrue(torch.all(pembs[9] == layer.embs[0, 3]))

        x = torch.randn(10, 96)
        y = layer(x, batch_indices)

        self.assertEqual(y.shape, (10, 96))

    def test_indexed_positional_embedding_fetch_with_batch_size_1(self):
        layer = IndexedPositionalEmbedding(96)
        layer.build()

        batch_indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        pembs = layer.fetch_embeddings(batch_indices)

        # i.e., [num_indices, features]
        self.assertEqual(pembs.shape, (10, 96))

        self.assertTrue(torch.all(pembs[0] == layer.embs[0, 0]))
        self.assertTrue(torch.all(pembs[1] == layer.embs[0, 1]))
        self.assertTrue(torch.all(pembs[2] == layer.embs[0, 2]))
        self.assertTrue(torch.all(pembs[3] == layer.embs[0, 3]))
        self.assertTrue(torch.all(pembs[4] == layer.embs[0, 4]))
        self.assertTrue(torch.all(pembs[5] == layer.embs[0, 5]))
        self.assertTrue(torch.all(pembs[6] == layer.embs[0, 6]))
        self.assertTrue(torch.all(pembs[7] == layer.embs[0, 7]))
        self.assertTrue(torch.all(pembs[8] == layer.embs[0, 8]))
        self.assertTrue(torch.all(pembs[9] == layer.embs[0, 9]))

        x = torch.randn(10, 96)
        y = layer(x, batch_indices)

        self.assertEqual(y.shape, (10, 96))
