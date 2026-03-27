import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from deeplay import (
    GraphConvolutionalNeuralNetwork,
    GraphToGlobalMPM,
    GraphToNodeMPM,
    GraphToEdgeMPM,
    GraphToEdgeMAGIK,
    MessagePassingNeuralNetwork,
    ResidualMessagePassingNeuralNetwork,
    RecurrentMessagePassingModel,
    RecurrentGraphBlock,
    MultiLayerPerceptron,
    dense_laplacian_normalization,
    Sum,
    WeightedSum,
    Mean,
    Prod,
    Min,
    Max,
    Layer,
    GlobalMeanPool,
    CatDictElements,
)

import itertools


class TestComponentGCN(unittest.TestCase):
    def test_gnn_defaults(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn = gnn.create()

        self.assertEqual(len(gnn.blocks), 2)

        self.assertEqual(gnn.blocks[0].transform.in_features, 2)
        self.assertEqual(gnn.blocks[0].transform.out_features, 4)

        self.assertEqual(gnn.output.transform.in_features, 4)
        self.assertEqual(gnn.output.transform.out_features, 1)
        # test on a batch of 2
        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2, 1], [1, 0, 2, 1, 0]])
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
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

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
                - out["edge_index"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_no_normalization_with_sparse_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.replace("normalize", Layer(nn.Identity))
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(inp["edge_index"] == out["edge_index"].to_dense()))
        self.assertEqual(out["x"].shape, (3, 1))

    def test_normalization_with_sparse_A_and_repd_edges(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        # edge (2, 1) is repeated
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 1]])

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
                - out["edge_index"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_with_dense_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.normalize.configure(dense_laplacian_normalization)
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

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
                - out["edge_index"].to_dense()
            ).sum()
            < 1e-4,
        )

    def test_normalization_no_normalization_with_dense_A(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.replace("normalize", Layer(nn.Identity))
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        out = gnn(inp)
        self.assertTrue(torch.all(inp["edge_index"] == out["edge_index"].to_dense()))
        self.assertEqual(out["x"].shape, (3, 1))

    def test_numeric_output(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn.output.update.set_output_map()

        gnn = gnn.create()
        # print(gnn.output.update.output_args)
        # gnn.build()

        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        # print(out)
        self.assertTrue(torch.is_tensor(out))

    def test_custom_propagation(self):
        class custom_propagation(nn.Module):
            def forward(self, x, A):
                return x * 0

        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        # print("before", gnn.propagate)
        gnn.propagate.configure(custom_propagation)

        gnn = gnn.create()
        # # print(gnn._input_mapped)
        inp = {}
        inp["x"] = torch.randn(3, 2)
        inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        out = gnn(inp)
        self.assertTrue(torch.all(out["x"] == 0))

    def test_tg_data_input(self):
        gnn = GraphConvolutionalNeuralNetwork(2, [4], 1)
        gnn = gnn.create()

        inp = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )

        out = gnn(inp)
        self.assertEqual(out.x.shape, (3, 1))


class TestComponentMPN(unittest.TestCase):
    def test_mpn_defaults(self):
        gnn = MessagePassingNeuralNetwork([4], 1)
        gnn = gnn.create()

        self.assertEqual(len(gnn.blocks), 2)

        self.assertEqual(gnn.transform[0].layer.out_features, 4)
        self.assertEqual(gnn.update[0].layer.out_features, 4)

        self.assertEqual(gnn.output.update.layer.out_features, 1)

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 30))
        inp["edge_attr"] = torch.ones(30, 1)

        out = gnn(inp)

        self.assertEqual(out["x"].shape, (10, 1))
        self.assertEqual(out["edge_attr"].shape, (30, 1))
        self.assertTrue(torch.all(inp["edge_index"] == out["edge_index"]))

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
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 30))
        inp["edge_attr"] = torch.ones(30, 1)

        # by default, the propagation is a sum
        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1], return_counts=True)
        expected = torch.zeros(10, 1)
        expected[uniques[0]] = uniques[1].unsqueeze(1).float()

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Mean(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Mean())
        gnn.blocks.propagate.set_input_map("x", "edge_index", "edge_attr")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 3))
        inp["edge_attr"] = torch.ones(3, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Prod(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Prod())
        gnn.blocks.propagate.set_input_map("x", "edge_index", "edge_attr")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Min(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Min())
        gnn.blocks.propagate.set_input_map("x", "edge_index", "edge_attr")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Max(self):
        gnn = MessagePassingNeuralNetwork([4], 1)

        gnn.blocks[0].replace("propagate", Max())
        gnn.blocks.propagate.set_input_map("x", "edge_index", "edge_attr")
        gnn.blocks.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 2)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.ones(20, 1)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_tg_data_input(self):
        gnn = MessagePassingNeuralNetwork([4], 1)
        gnn = gnn.create()

        inp = Data(
            x=torch.randn(10, 2),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.ones(20, 1),
        )

        out = gnn(inp)

        self.assertEqual(out.x.shape, (10, 1))
        self.assertEqual(out.edge_attr.shape, (20, 1))
        self.assertTrue(torch.all(inp.edge_index == out.edge_index))


class TestComponentRMLP(unittest.TestCase):
    def test_rmpn_defaults(self):
        gnn = ResidualMessagePassingNeuralNetwork([4], 4)
        gnn = gnn.create()

        self.assertEqual(len(gnn.blocks), 2)

        self.assertEqual(gnn.transform[0].layer.out_features, 4)
        self.assertEqual(gnn.update[0].layer.out_features, 4)

        self.assertEqual(gnn.output.update.layer.out_features, 4)

        inp = {}
        inp["x"] = torch.randn(10, 4)
        inp["edge_index"] = torch.randint(0, 10, (2, 30))
        inp["edge_attr"] = torch.ones(30, 4)

        out = gnn(inp)

        self.assertEqual(out["x"].shape, (10, 4))
        self.assertEqual(out["edge_attr"].shape, (30, 4))
        self.assertTrue(torch.all(inp["edge_index"] == out["edge_index"]))

    def test_gnn_change_depth(self):
        gnn = ResidualMessagePassingNeuralNetwork([4], 4)
        gnn.configure(hidden_features=[4, 4])
        gnn.create()
        gnn.build()
        self.assertEqual(len(gnn.blocks), 3)

    def test_gnn_activation_change(self):
        gnn = ResidualMessagePassingNeuralNetwork([4, 4], 1)
        gnn.configure(out_activation=nn.Sigmoid)
        gnn.create()
        gnn.build()

        self.assertIsInstance(gnn.output.transform.activation, nn.Sigmoid)
        self.assertIsInstance(gnn.output.update.activation, nn.Sigmoid)

    def test_gnn_default_propagation(self):
        gnn = ResidualMessagePassingNeuralNetwork([4], 4)
        gnn = gnn.create()

        inp = {}
        inp["x"] = torch.randn(10, 4)
        inp["edge_index"] = torch.randint(0, 10, (2, 30))
        inp["edge_attr"] = torch.ones(30, 4)

        # by default, the propagation is a sum
        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1], return_counts=True)
        expected = torch.zeros(10, 1)
        expected[uniques[0]] = uniques[1].unsqueeze(1).float()

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_gnn_propagation_change_Mean(self):
        gnn = ResidualMessagePassingNeuralNetwork([4], 4)

        gnn.blocks[0].layer.replace("propagate", Mean())
        gnn.propagate.set_input_map("x", "edge_index", "edge_attr")
        gnn.propagate.set_output_map("aggregate")

        gnn.create()
        gnn.build()

        inp = {}
        inp["x"] = torch.randn(10, 4)
        inp["edge_index"] = torch.randint(0, 10, (2, 3))
        inp["edge_attr"] = torch.ones(3, 4)

        propagator = gnn.propagate[0]
        out = propagator(inp)

        uniques = torch.unique(inp["edge_index"][1])
        expected = torch.zeros(10, 1)
        expected[uniques] = 1.0

        self.assertTrue(torch.all(out["aggregate"] == expected))

    def test_tg_data_input(self):
        gnn = ResidualMessagePassingNeuralNetwork([4], 4)
        gnn = gnn.create()

        inp = Data(
            x=torch.randn(10, 4),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.ones(20, 4),
        )

        out = gnn(inp)

        self.assertEqual(out.x.shape, (10, 4))
        self.assertEqual(out.edge_attr.shape, (20, 4))
        self.assertTrue(torch.all(inp.edge_index == out.edge_index))


class TestModelGraphToGlobalMPM(unittest.TestCase):
    def test_gtogmpm_defaults(self):
        model = GraphToGlobalMPM([64, 64], 1)
        model = model.create()

        # node and edge encoders are defined as Linear layers by default
        self.assertEqual(len(model.encoder[0].blocks), 1)
        self.assertEqual(len(model.encoder[1].blocks), 1)

        self.assertEqual(model.encoder[0].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[0].blocks[0].layer.out_features, 64)
        self.assertEqual(model.encoder[1].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[1].blocks[0].layer.out_features, 64)

        self.assertIsInstance(model.backbone, MessagePassingNeuralNetwork)

        backbone_blocks = model.backbone.blocks
        self.assertEqual(len(backbone_blocks), 2)

        for block in backbone_blocks:

            self.assertEqual(block.transform.layer.in_features, 0)
            self.assertEqual(block.transform.layer.out_features, 64)
            self.assertIsInstance(block.transform.activation, nn.ReLU)

            self.assertIsInstance(block.propagate, Sum)

            self.assertEqual(block.update.layer.in_features, 0)
            self.assertEqual(block.update.layer.out_features, 64)
            self.assertIsInstance(block.update.activation, nn.ReLU)

        self.assertEqual(model.selector.keys, ("x", "batch"))

        self.assertIsInstance(model.pool, GlobalMeanPool)

        self.assertIsInstance(model.head, MultiLayerPerceptron)
        self.assertEqual(model.head.blocks[0].layer.in_features, 64)
        self.assertEqual(model.head.blocks[0].layer.out_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.in_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.out_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.in_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.out_features, 1)

        model = GraphToGlobalMPM([64, 64], 1).create()
        inp = {}
        inp["x"] = torch.randn(10, 16)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.randn(20, 8)
        inp["batch"] = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).long()

        out = model(inp)

        self.assertEqual(out.shape, (2, 1))

    def test_gtogmpm_change_depth(self):
        model = GraphToGlobalMPM([64, 64], 1)
        model.configure(hidden_features=[64, 64, 64])
        model.create()
        model.build()

        backbone_blocks = model.backbone.blocks
        self.assertEqual(len(backbone_blocks), 3)

        for block in backbone_blocks:

            self.assertEqual(block.transform.layer.in_features, 0)
            self.assertEqual(block.transform.layer.out_features, 64)
            self.assertIsInstance(block.transform.activation, nn.ReLU)

            self.assertIsInstance(block.propagate, Sum)

            self.assertEqual(block.update.layer.in_features, 0)
            self.assertEqual(block.update.layer.out_features, 64)
            self.assertIsInstance(block.update.activation, nn.ReLU)

    def test_gtogmpm_change_head_activation(self):
        model = GraphToGlobalMPM([64, 64], 1, out_activation=nn.Sigmoid)
        model.create()
        model.build()

        self.assertIsInstance(model.head.blocks[-1].activation, nn.Sigmoid)

    def test_gtogmpm_change_head_depth(self):
        model = GraphToGlobalMPM([64, 64], 1)
        model.head.configure(hidden_features=[64, 64, 64])
        model.create()
        model.build()

        self.assertEqual(model.head.blocks[0].layer.in_features, 64)
        self.assertEqual(model.head.blocks[0].layer.out_features, 64)
        self.assertEqual(model.head.blocks[1].layer.in_features, 64)
        self.assertEqual(model.head.blocks[1].layer.out_features, 64)
        self.assertEqual(model.head.blocks[2].layer.in_features, 64)
        self.assertEqual(model.head.blocks[2].layer.out_features, 64)
        self.assertEqual(model.head.blocks[3].layer.in_features, 64)
        self.assertEqual(model.head.blocks[3].layer.out_features, 1)


class TestModelGraphToNodesMPM(unittest.TestCase):
    def test_gtonmpm_defaults(self):
        model = GraphToNodeMPM([64, 64], 1)
        model = model.build()

        # node and edge encoders are defined as Linear layers by default
        self.assertEqual(len(model.encoder[0].blocks), 1)
        self.assertEqual(len(model.encoder[1].blocks), 1)

        self.assertEqual(model.encoder[0].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[0].blocks[0].layer.out_features, 64)
        self.assertEqual(model.encoder[1].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[1].blocks[0].layer.out_features, 64)

        self.assertIsInstance(model.backbone, MessagePassingNeuralNetwork)

        backbone_blocks = model.backbone.blocks
        self.assertEqual(len(backbone_blocks), 2)

        for block in backbone_blocks:

            self.assertEqual(block.transform.layer.in_features, 0)
            self.assertEqual(block.transform.layer.out_features, 64)
            self.assertIsInstance(block.transform.activation, nn.ReLU)

            self.assertIsInstance(block.propagate, Sum)

            self.assertEqual(block.update.layer.in_features, 0)
            self.assertEqual(block.update.layer.out_features, 64)
            self.assertIsInstance(block.update.activation, nn.ReLU)

        self.assertEqual(model.selector.keys, ("x",))
        self.assertIsInstance(model.pool, nn.Identity)

        self.assertIsInstance(model.head, MultiLayerPerceptron)
        self.assertEqual(model.head.blocks[0].layer.in_features, 64)
        self.assertEqual(model.head.blocks[0].layer.out_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.in_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.out_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.in_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.out_features, 1)

        model = GraphToNodeMPM([64, 64], 1).create()
        inp = {}
        inp["x"] = torch.randn(10, 16)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.randn(20, 8)

        out = model(inp)

        self.assertEqual(out.shape, (10, 1))


class TestModelGraphToEdgeMPM(unittest.TestCase):
    def test_gtoempm_defaults(self):
        model = GraphToEdgeMPM([64, 64], 1)
        model = model.create()

        # node and edge encoders are defined as Linear layers by default
        self.assertEqual(len(model.encoder[0].blocks), 1)
        self.assertEqual(len(model.encoder[1].blocks), 1)

        self.assertEqual(model.encoder[0].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[0].blocks[0].layer.out_features, 64)
        self.assertEqual(model.encoder[1].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[1].blocks[0].layer.out_features, 64)

        self.assertIsInstance(model.backbone, MessagePassingNeuralNetwork)

        backbone_blocks = model.backbone.blocks
        self.assertEqual(len(backbone_blocks), 2)

        for block in backbone_blocks:

            self.assertEqual(block.transform.layer.in_features, 0)
            self.assertEqual(block.transform.layer.out_features, 64)
            self.assertIsInstance(block.transform.activation, nn.ReLU)

            self.assertIsInstance(block.propagate, Sum)

            self.assertEqual(block.update.layer.in_features, 0)
            self.assertEqual(block.update.layer.out_features, 64)
            self.assertIsInstance(block.update.activation, nn.ReLU)

        self.assertEqual(model.selector.keys, ("edge_attr",))
        self.assertIsInstance(model.pool, nn.Identity)

        self.assertIsInstance(model.head, MultiLayerPerceptron)
        self.assertEqual(model.head.blocks[0].layer.in_features, 64)
        self.assertEqual(model.head.blocks[0].layer.out_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.in_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.out_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.in_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.out_features, 1)

        model = GraphToEdgeMPM([64, 64], 1).create()
        inp = {}
        inp["x"] = torch.randn(10, 16)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.randn(20, 8)

        out = model(inp)

        self.assertEqual(out.shape, (20, 1))


class TestModelGraphToEdgeMAGIK(unittest.TestCase):
    def test_gtoempm_defaults(self):
        model = GraphToEdgeMAGIK([64, 64], 1)
        model = model.create()

        # node and edge encoders are defined as Linear layers by default
        self.assertEqual(len(model.encoder[0].blocks), 1)
        self.assertEqual(len(model.encoder[1].blocks), 1)

        self.assertEqual(model.encoder[0].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[0].blocks[0].layer.out_features, 64)
        self.assertEqual(model.encoder[1].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[1].blocks[0].layer.out_features, 64)

        self.assertIsInstance(model.backbone, MessagePassingNeuralNetwork)

        backbone_blocks = model.backbone.blocks

        self.assertEqual(len(backbone_blocks), 3)

        self.assertEqual(backbone_blocks[0].sigma, 0.12)
        self.assertEqual(backbone_blocks[0].beta, 4.0)

        for block in backbone_blocks[1:]:

            self.assertEqual(block.transform.layer.in_features, 0)
            self.assertEqual(block.transform.layer.out_features, 64)
            self.assertIsInstance(block.transform.activation, nn.ReLU)

            self.assertIsInstance(block.propagate, WeightedSum)

            self.assertEqual(block.update.layer.in_features, 0)
            self.assertEqual(block.update.layer.out_features, 64)
            self.assertIsInstance(block.update.activation, nn.ReLU)

        self.assertEqual(model.selector.keys, ("edge_attr",))
        self.assertIsInstance(model.pool, nn.Identity)

        self.assertIsInstance(model.head, MultiLayerPerceptron)
        self.assertEqual(model.head.blocks[0].layer.in_features, 64)
        self.assertEqual(model.head.blocks[0].layer.out_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.in_features, 64 // 2)
        self.assertEqual(model.head.blocks[1].layer.out_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.in_features, 64 // 4)
        self.assertEqual(model.head.blocks[2].layer.out_features, 1)

        inp = {}
        inp["x"] = torch.randn(10, 16)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.randn(20, 8)
        inp["distance"] = torch.randn(20, 1)

        out = model(inp)

        self.assertEqual(out.shape, (20, 1))


class TestModelRecurrentMPM(unittest.TestCase):
    def test_recurrent_graph_block_defaults(self):
        model = RecurrentGraphBlock(
            combine=CatDictElements(("x", "hidden")),
            layer=Layer(nn.Identity),
            head=Layer(nn.Identity),
            hidden_features=64,
            num_iter=1,
        )
        model = model.create()

        self.assertIsInstance(model.combine, CatDictElements)
        self.assertIsInstance(model.layer, nn.Identity)
        self.assertIsInstance(model.head, nn.Identity)

        self.assertEqual(model.hidden_features, 64)
        self.assertEqual(model.num_iter, 1)

        # assess the case where hidden is provided
        inp = {}
        inp["x"] = torch.ones(10, 64)
        out = model(inp)

        self.assertTrue(torch.all(out[0]["hidden"][:, :64] == torch.zeros(10, 64)))

        # assess the case where hidden is provided
        inp["hidden"] = torch.ones(10, 64) * 2
        out = model(inp)

        self.assertTrue(torch.all(out[0]["hidden"][:, :64] == torch.ones(10, 64) * 2))

    def test_RMPM_defaults(self):
        model = RecurrentMessagePassingModel(96, 2, num_iter=10)
        model = model.create()

        self.assertEqual(len(model.encoder[0].blocks), 1)
        self.assertEqual(len(model.encoder[1].blocks), 1)

        self.assertEqual(model.encoder[0].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[0].blocks[0].layer.out_features, 96)
        self.assertEqual(model.encoder[1].blocks[0].layer.in_features, 0)
        self.assertEqual(model.encoder[1].blocks[0].layer.out_features, 96)

        self.assertIsInstance(model.backbone, RecurrentGraphBlock)
        self.assertIsInstance(model.backbone.combine, CatDictElements)
        self.assertIsInstance(model.backbone.layer[0][0], MultiLayerPerceptron)
        self.assertIsInstance(model.backbone.layer[0][1], MultiLayerPerceptron)
        self.assertIsInstance(model.backbone.layer[1], MessagePassingNeuralNetwork)
        self.assertIsInstance(model.backbone.head, MultiLayerPerceptron)

        self.assertEqual(model.backbone.head.in_features, 96)
        self.assertEqual(model.backbone.head.out_features, 2)

        # check default mapping
        self.assertEqual(model.encoder[0].input_args, ("x",))
        self.assertEqual(model.encoder[0].output_args.keys(), {"x"})
        self.assertEqual(model.encoder[1].input_args, ("edge_attr",))
        self.assertEqual(model.encoder[1].output_args.keys(), {"edge_attr"})

        self.assertEqual(model.backbone.combine.source, ("x", "edge_attr"))
        self.assertEqual(
            model.backbone.combine.target, ("hidden_x", "hidden_edge_attr")
        )

        self.assertEqual(model.backbone.layer[0][0].input_args, ("hidden_x",))
        self.assertEqual(model.backbone.layer[0][0].output_args.keys(), {"hidden_x"})
        self.assertEqual(model.backbone.layer[0][1].input_args, ("hidden_edge_attr",))
        self.assertEqual(
            model.backbone.layer[0][1].output_args.keys(), {"hidden_edge_attr"}
        )

        self.assertEqual(
            model.backbone.layer[1].transform[0].input_args,
            ("hidden_x", "edge_index", "hidden_edge_attr"),
        )
        self.assertEqual(
            model.backbone.layer[1].transform[0].output_args.keys(),
            {"hidden_edge_attr"},
        )
        self.assertEqual(
            model.backbone.layer[1].propagate[0].input_args,
            ("hidden_x", "edge_index", "hidden_edge_attr"),
        )
        self.assertEqual(
            model.backbone.layer[1].propagate[0].output_args.keys(), {"aggregate"}
        )

        self.assertEqual(model.backbone.layer[1].update[0].input_args, ("aggregate",))
        self.assertEqual(
            model.backbone.layer[1].update[0].output_args.keys(), {"hidden_x"}
        )

        inp = {}
        inp["x"] = torch.randn(10, 5)
        inp["edge_index"] = torch.randint(0, 10, (2, 20))
        inp["edge_attr"] = torch.randn(20, 3)

        out = model(inp)

        self.assertEqual(len(out), 10)
        for o in out:
            self.assertEqual(o.shape, (10, 2))
