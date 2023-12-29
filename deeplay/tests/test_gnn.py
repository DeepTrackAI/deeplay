import unittest
import torch
import torch.nn as nn
from deeplay import (
    GraphConvolutionalNeuralNetwork,
    dense_laplacian_normalization,
    Layer,
    ToDict,
)

import itertools


class TestComponentGNN(unittest.TestCase):
    def test_gnn_defaults(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.build()
        gnn.create()

        self.assertEqual(len(gnn.blocks), 2)

        self.assertEqual(gnn.blocks[0].transform.in_features, 2)
        self.assertEqual(gnn.blocks[0].transform.out_features, 4)

        self.assertEqual(gnn.output.transform.in_features, 4)
        self.assertEqual(gnn.output.transform.out_features, 1)
        # test on a batch of 2
        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2, 1], [1, 0, 2, 1, 0]])
        out = gnn(inp)
        self.assertEqual(out["x"].shape, (3, 1))

    def test_gnn_change_depth(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 3)
        gnn.configure(hidden_channels=[4, 4])
        gnn.create()
        gnn.build()
        self.assertEqual(len(gnn.blocks), 3)

    def test_normalization_with_sparse_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(
            (
                torch.Tensor(
                    [
                        [1.0000, -0.7071, 0.0000],
                        [-0.7071, 1.0000, -0.7071],
                        [0.0000, -0.7071, 1.0000],
                    ]
                )
                - out["A"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_no_normalization_with_sparse_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.replace("normalize", Layer(nn.Identity))
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(inp["A"] == out["A"].to_dense()))
        self.assertEqual(out["x"].shape, (3, 1))

    def test_normalization_with_sparse_A_and_repd_edges(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        # edge (2, 1) is repeated
        inp["A"] = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 1]])

        out = gnn(inp)
        self.assertTrue(
            (
                torch.Tensor(
                    [
                        [1.0000, -0.7071, 0.0000],
                        [-0.7071, 1.0000, -0.7071],
                        [0.0000, -0.7071, 1.0000],
                    ]
                )
                - out["A"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_with_dense_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.normalize.configure(dense_laplacian_normalization)
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        out = gnn(inp)
        self.assertTrue(
            (
                torch.Tensor(
                    [
                        [1.0000, -0.7071, 0.0000],
                        [-0.7071, 1.0000, -0.7071],
                        [0.0000, -0.7071, 1.0000],
                    ]
                )
                - out["A"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_no_normalization_with_dense_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.replace("normalize", Layer(nn.Identity))
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        out = gnn(inp)
        self.assertTrue(torch.all(inp["A"] == out["A"].to_dense()))
        self.assertEqual(out["x"].shape, (3, 1))

    def test_numeric_output(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.output.update.set_output_map()
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.is_tensor(out))

    def test_custom_propagation(self):
        class custom_propagation(nn.Module):
            def forward(self, x, A):
                return x * 0

        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.propagate.configure(custom_propagation)
        gnn.build()
        gnn.create()

        inp = ToDict()
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(out["x"] == 0))
