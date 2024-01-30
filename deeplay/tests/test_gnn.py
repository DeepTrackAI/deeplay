import unittest
import torch
import torch.nn as nn
from deeplay import (
    GraphConvolutionalNeuralNetwork,
    MessagePassingNeuralNetwork,
    dense_laplacian_normalization,
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    Layer,
)

import itertools


class TestComponentGCN(unittest.TestCase):
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
        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2, 1], [1, 0, 2, 1, 0]])
        out = gnn(inp)
        self.assertEqual(out["x"].shape, (3, 1))

    def test_gnn_change_depth(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 3)
        gnn.configure(hidden_features=[4, 4])
        gnn.create()
        gnn.build()
        self.assertEqual(len(gnn.blocks), 3)

    def test_normalization_with_sparse_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.build()
        gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(
            (
                # Obtained using kipf's normalization (https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L24)
                torch.Tensor(
                    [
                        [0.5000, 0.4082, 0.0000],
                        [0.4082, 0.3333, 0.4082],
                        [0.0000, 0.4082, 0.5000],
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

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(inp["A"] == out["A"].to_dense()))
        self.assertEqual(out["x"].shape, (3, 1))

    def test_normalization_with_sparse_A_and_repd_edges(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.build()
        gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        # edge (2, 1) is repeated
        inp["A"] = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 1]])

        out = gnn(inp)
        self.assertTrue(
            (
                # Obtained using kipf's normalization (https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L24)
                torch.Tensor(
                    [
                        [0.5000, 0.4082, 0.0000],
                        [0.4082, 0.3333, 0.4082],
                        [0.0000, 0.4082, 0.5000],
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

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        out = gnn(inp)
        self.assertTrue(
            (
                # Obtained using kipf's normalization (https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L24)
                torch.Tensor(
                    [
                        [0.5000, 0.4082, 0.0000],
                        [0.4082, 0.3333, 0.4082],
                        [0.0000, 0.4082, 0.5000],
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

        inp = {}
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

        inp = {}
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

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["A"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(out["x"] == 0))


class TestComponentMPN(unittest.TestCase):
    def test_mpn_defaults(self):
        gnn = MessagePassingNeuralNetwork([4], 1)
        gnn.build()
        gnn.create()

        self.assertEqual(len(gnn.blocks), 2)

        self.assertEqual(gnn.transform[0].layer.out_features, 4)
        self.assertEqual(gnn.update[0].layer.out_features, 4)

        self.assertEqual(gnn.output.update.layer.out_features, 1)

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 30))
        inp["edgefeat"] = torch.ones(30, 1)

        out = gnn(inp)

        self.assertEqual(out["x"].shape, (10, 1))
        self.assertEqual(out["edgefeat"].shape, (30, 1))
        self.assertTrue(torch.all(inp["A"] == out["A"]))

    def test_gnn_change_depth(self):
        gnn = MessagePassingNeuralNetwork([4], 1)
        gnn.configure(hidden_features=[4, 4])
        gnn.create()
        gnn.build()
        self.assertEqual(len(gnn.blocks), 3)

    def test_gnn_activation_change(self):
        gnn = MessagePassingNeuralNetwork([4, 4], 1)
        gnn.configure(out_activation=nn.Sigmoid)
        gnn.create()
        gnn.build()
        
        self.assertIsInstance(gnn.output.transform.activation, nn.Sigmoid)
        self.assertIsInstance(gnn.output.update.activation, nn.Sigmoid)

    def test_gnn_default_propagation(self):
        gnn = MessagePassingNeuralNetwork([4], 1)
        gnn.build()
        gnn.create()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 30))
        inp["edgefeat"] = torch.ones(30, 1)

        # by default, the propagation is a sum
        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["A"][1], return_counts=True)
        expected = torch.zeros(10, 1)
        expected[uniques[0]] = uniques[1].unsqueeze(1).float()

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Mean(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Mean())
        gnn.blocks.propagate.set_input_map("x", "A", "edgefeat")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 3))
        inp["edgefeat"] = torch.ones(3, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["A"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Prod(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Prod())
        gnn.blocks.propagate.set_input_map("x", "A", "edgefeat")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 20))
        inp["edgefeat"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["A"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Min(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Min())
        gnn.blocks.propagate.set_input_map("x", "A", "edgefeat")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 20))
        inp["edgefeat"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["A"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Max(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Max())
        gnn.blocks.propagate.set_input_map("x", "A", "edgefeat")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["A"] = torch.randint(0, 10, (2, 20))
        inp["edgefeat"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["A"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))
