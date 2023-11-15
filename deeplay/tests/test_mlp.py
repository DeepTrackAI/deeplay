import unittest
import torch
import torch.nn as nn
from deeplay import MultiLayerPerceptron, Layer


class TestComponentMLP(unittest.TestCase):
    ...

    def test_mlp_defaults(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)

        self.assertEqual(mlp.blocks[0].layer.in_features, 2)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)

        self.assertEqual(mlp.out_layer.layer.in_features, 4)
        self.assertEqual(mlp.out_layer.layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp_lazy_input(self):
        mlp = MultiLayerPerceptron(None, [4], 3).build()
        self.assertEqual(len(mlp.blocks), 1)

        self.assertEqual(mlp.blocks[0].layer.in_features, 0)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)
        self.assertEqual(mlp.out_layer.layer.in_features, 4)
        self.assertEqual(mlp.out_layer.layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp_change_depth(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(hidden_features=[4, 4])
        mlp.build()
        self.assertEqual(len(mlp.blocks), 2)

    def test_change_activation(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=nn.Sigmoid)
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)
        self.assertIsInstance(mlp.out_layer.act, nn.Sigmoid)

    def test_change_out_activation_Layer(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=Layer(nn.Sigmoid))
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)
        self.assertIsInstance(mlp.out_layer.act, nn.Sigmoid)

    def test_change_out_activation_instance(self):
        mlp = MultiLayerPerceptron(2, [4], 3)
        mlp.configure(out_activation=nn.Sigmoid())
        mlp.build()
        self.assertEqual(len(mlp.blocks), 1)
        self.assertIsInstance(mlp.out_layer.act, nn.Sigmoid)
