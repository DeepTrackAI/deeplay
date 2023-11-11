import unittest
import torch
import torch.nn as nn
from deeplay import MultiLayerPerceptron


class TestComponentMLP(unittest.TestCase):
    def test_mlp_defaults(self):
        mlp = MultiLayerPerceptron(2, [4], 3).build()
        self.assertEqual(len(mlp.blocks), 1)

        self.assertEqual(mlp.blocks[0].layer.in_features, 2)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)

        self.assertEqual(mlp.blocks[1].layer.in_features, 4)
        self.assertEqual(mlp.blocks[1].layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp_lazy_input(self):
        mlp = MultiLayerPerceptron(None, [4], 3)
        self.assertEqual(len(mlp.blocks), 1)

        self.assertEqual(mlp.blocks[0].layer.in_features, 0)
        self.assertEqual(mlp.blocks[0].layer.out_features, 4)

        self.assertEqual(mlp.blocks[1].layer.in_features, 4)
        self.assertEqual(mlp.blocks[1].layer.out_features, 3)

        # test on a batch of 2
        x = torch.randn(2, 2)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))

    # def test_mlp_custom_depth(self):
    #     mlp = MultiLayerPerceptron(2, [4, 5, 6], 3)
    #     self.assertEqual(len(mlp.blocks), 3)

    #     first_layer = mlp.blocks[0]["layer"]
    #     self.assertEqual(first_layer.in_features, 2)
    #     self.assertEqual(first_layer.out_features, 4)

    #     second_layer = mlp.blocks[1]["layer"]
    #     self.assertEqual(second_layer.in_features, 4)
    #     self.assertEqual(second_layer.out_features, 5)

    #     third_layer = mlp.blocks[2]["layer"]
    #     self.assertEqual(third_layer.in_features, 5)
    #     self.assertEqual(third_layer.out_features, 6)

    #     output_layer = mlp.out_layer
    #     self.assertEqual(output_layer.in_features, 6)
    #     self.assertEqual(output_layer.out_features, 3)

    #     # test on a batch of 2
    #     x = torch.randn(2, 2)
    #     y = mlp(x)
    #     self.assertEqual(y.shape, (2, 3))
