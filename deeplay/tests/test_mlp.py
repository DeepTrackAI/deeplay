import unittest
import torch
import torch.nn as nn
from deeplay import MultiLayerPerceptron, Layer


class TestComponentMLP(unittest.TestCase):
    ...

    def test_mlp_defaults(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 2)

        self.assertEqual(mlp.blocks[0].layer.in_features, 2)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)

        self.assertEqual(mlp.output.layer.in_features, 4)
        self.assertEqual(mlp.output.layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp_lazy_input(self):
        mlp = MultiLayerPerceptron(None, [4], 3).build()
        self.assertEqual(len(mlp.blocks), 2)

        self.assertEqual(mlp.blocks[0].layer.in_features, 0)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)
        self.assertEqual(mlp.output.layer.in_features, 4)
        self.assertEqual(mlp.output.layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp_change_depth(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(hidden_features=[4, 4])
        mlp.build()
        self.assertEqual(len(mlp.blocks), 3)

    def test_change_activation(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=nn.Sigmoid)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 2)
        self.assertIsInstance(mlp.output.activation, nn.Sigmoid)

    def test_change_out_activation_Layer(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=Layer(nn.Sigmoid))
        mlp.build()
        self.assertEqual(len(mlp.blocks), 2)
        self.assertIsInstance(mlp.output.activation, nn.Sigmoid)

    def test_change_out_activation_instance(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=nn.Sigmoid())
        mlp.build()
        self.assertEqual(len(mlp.blocks), 2)
        self.assertIsInstance(mlp.output.activation, nn.Sigmoid)

    def test_no_hidden_layers(self):
        mlp = MultiLayerPerceptron(2, [], 3)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)
        self.assertEqual(mlp.blocks[0].layer.in_features, 2)
        self.assertEqual(mlp.blocks[0].layer.out_features, 3)

    def test_configure_layers(self):
        mlp = MultiLayerPerceptron(2, [4, 3, 5], 3)
        mlp.layer.configure(bias=False)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 4)
        for idx, block in enumerate(mlp.blocks):
            self.assertFalse(block.layer.bias)

    def test_configure_activation(self):
        mlp = MultiLayerPerceptron(2, [4, 3, 5], 3)
        mlp.activation.configure(nn.Sigmoid)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 4)
        for idx, block in enumerate(mlp.blocks):
            self.assertIsInstance(block.activation, nn.Sigmoid)

    def test_configure_activation_with_argument(self):
        mlp = MultiLayerPerceptron(2, [4, 3, 5], 3)
        mlp.activation.configure(nn.Softmax, dim=1)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 4)
        for idx, block in enumerate(mlp.blocks):
            self.assertIsInstance(block.activation, nn.Softmax)
            self.assertEqual(block.activation.dim, 1)

    def test_configure_normalization(self):
        mlp = MultiLayerPerceptron(2, [4, 3, 5], 3)
        mlp["blocks", :].all.normalized(nn.BatchNorm1d)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 4)
        for idx, block in enumerate(mlp.blocks):
            self.assertIsInstance(block.normalization, nn.BatchNorm1d)

        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_no_input_no_hidden_layers(self):
        mlp = MultiLayerPerceptron(None, [], 3)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)
        self.assertEqual(mlp.blocks[0].layer.in_features, 0)
        self.assertEqual(mlp.blocks[0].layer.out_features, 3)
