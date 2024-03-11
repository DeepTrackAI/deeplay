import unittest

import torch
from deeplay import BatchedPositionalEmbedding, IndexedPositionalEmbedding


class TestPos(unittest.TestCase):
    ...

    def test_batched_positional_embedding_default(self):
        layer = BatchedPositionalEmbedding(96)
        layer.build()

        # i.e., [batch_size, max_length, features]
        self.assertEqual(layer.embs.shape, (1, 5000, 96))

        # By default, dropout is 0.0
        self.assertEqual(layer.dropout.p, 0.0)

        x = torch.randn(10, 5000, 96)
        y = layer(x)

        self.assertEqual(y.shape, (10, 5000, 96))

    def test_batched_positional_embedding_batch_first(self):
        layer = BatchedPositionalEmbedding(96, batch_first=False)
        layer.build()

        # i.e., [max_length, 1, features]
        self.assertEqual(layer.embs.shape, (5000, 1, 96))

        x = torch.randn(5000, 10, 96)
        y = layer(x)

        self.assertEqual(y.shape, (5000, 10, 96))

    def test_indexed_positional_embedding_default(self):
        layer = IndexedPositionalEmbedding(96)
        layer.build()

        # i.e., [max_length, features]
        self.assertEqual(layer.embs.shape, (5000, 96))

        # By default, dropout is 0.0
        self.assertEqual(layer.dropout.p, 0.0)

    def test_indexed_positional_embedding_fetch(self):
        layer = IndexedPositionalEmbedding(96)
        layer.build()

        batch_indices = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        pembs = layer.fetch_embeddings(batch_indices)

        # i.e., [num_indices, features]
        self.assertEqual(pembs.shape, (10, 96))

        self.assertTrue(torch.all(pembs[0] == layer.embs[0]))
        self.assertTrue(torch.all(pembs[1] == layer.embs[1]))
        self.assertTrue(torch.all(pembs[2] == layer.embs[0]))
        self.assertTrue(torch.all(pembs[3] == layer.embs[1]))
        self.assertTrue(torch.all(pembs[4] == layer.embs[2]))
        self.assertTrue(torch.all(pembs[5] == layer.embs[3]))
        self.assertTrue(torch.all(pembs[6] == layer.embs[0]))
        self.assertTrue(torch.all(pembs[7] == layer.embs[1]))
        self.assertTrue(torch.all(pembs[8] == layer.embs[2]))
        self.assertTrue(torch.all(pembs[9] == layer.embs[3]))

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

        self.assertTrue(torch.all(pembs[0] == layer.embs[0]))
        self.assertTrue(torch.all(pembs[1] == layer.embs[1]))
        self.assertTrue(torch.all(pembs[2] == layer.embs[2]))
        self.assertTrue(torch.all(pembs[3] == layer.embs[3]))
        self.assertTrue(torch.all(pembs[4] == layer.embs[4]))
        self.assertTrue(torch.all(pembs[5] == layer.embs[5]))
        self.assertTrue(torch.all(pembs[6] == layer.embs[6]))
        self.assertTrue(torch.all(pembs[7] == layer.embs[7]))
        self.assertTrue(torch.all(pembs[8] == layer.embs[8]))
        self.assertTrue(torch.all(pembs[9] == layer.embs[9]))

        x = torch.randn(10, 96)
        y = layer(x, batch_indices)

        self.assertEqual(y.shape, (10, 96))
