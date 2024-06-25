import unittest

import torch
import torch.nn as nn
from deeplay import ConvolutionalNeuralNetwork, Layer, LayerList, Application

import itertools


class TestComponentCNN(unittest.TestCase):
    ...

    def test_cnn_defaults(self):
        cnn = ConvolutionalNeuralNetwork(3, [4], 1)
        cnn.build()
        cnn.create()

        self.assertEqual(len(cnn.blocks), 2)

        self.assertEqual(cnn.blocks[0].layer.in_channels, 3)
        self.assertEqual(cnn.blocks[0].layer.out_channels, 4)

        self.assertEqual(cnn.output.layer.in_channels, 4)
        self.assertEqual(cnn.output.layer.out_channels, 1)

        # test on a batch of 2
        x = torch.randn(2, 3, 5, 5)
        y = cnn(x)
        self.assertEqual(y.shape, (2, 1, 5, 5))

    def test_cnn_lazy_input(self):
        cnn = ConvolutionalNeuralNetwork(None, [4], 1).build()
        self.assertEqual(len(cnn.blocks), 2)

        self.assertEqual(cnn.input.layer.in_channels, 0)
        self.assertEqual(cnn.blocks[0].layer.out_channels, 4)
        self.assertEqual(cnn.output.layer.in_channels, 4)
        self.assertEqual(cnn.output.layer.out_channels, 1)

        # test on a batch of 2
        x = torch.randn(2, 3, 5, 5)
        y = cnn(x)
        self.assertEqual(y.shape, (2, 1, 5, 5))

    def test_cnn_change_depth(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(hidden_channels=[4, 4])
        cnn.create()
        cnn.build()
        self.assertEqual(len(cnn.blocks), 3)

    def test_change_act(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=nn.Sigmoid)
        cnn.create()
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output.activation, nn.Sigmoid)

    def test_change_out_act_Layer(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=Layer(nn.Sigmoid))
        cnn.create()
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output.activation, nn.Sigmoid)

    def test_change_out_act_instance(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=nn.Sigmoid())
        cnn.create()
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output.activation, nn.Sigmoid)

    def test_default_values_initialization(self):
        cnn = ConvolutionalNeuralNetwork(
            in_channels=None, hidden_channels=[32, 32], out_channels=1
        )
        self.assertIsNone(cnn.in_channels)
        self.assertEqual(cnn.hidden_channels, [32, 32])
        self.assertEqual(cnn.out_channels, 1)
        self.assertIsInstance(cnn.blocks, LayerList)

    def test_empty_hidden_channels(self):
        cnn = ConvolutionalNeuralNetwork(
            in_channels=3, hidden_channels=[], out_channels=1
        ).build()
        self.assertEqual(cnn.blocks[0].layer.in_channels, 3)
        self.assertEqual(cnn.blocks[0].layer.out_channels, 1)

        self.assertIs(cnn.blocks[0], cnn.output)
        self.assertIs(cnn.blocks[0], cnn.input)

    def test_zero_out_channels(self):
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetwork(
                in_channels=3, hidden_channels=[32, 64], out_channels=0
            )

    def test_all_cnn_blocks_are_not_same_object(self):
        cnn_with_default = ConvolutionalNeuralNetwork(3, [4, 4, 4], 1)
        cnn_with_pool_module = ConvolutionalNeuralNetwork(
            3, [4, 4, 4], 1, pool=nn.MaxPool2d
        )
        cnn_with_pool_class = ConvolutionalNeuralNetwork(
            3, [4, 4, 4], 1, pool=nn.MaxPool2d(2)
        )
        cnn_with_pool_layer = ConvolutionalNeuralNetwork(
            3, [4, 4, 4], 1, pool=Layer(nn.MaxPool2d)
        )

        # for a, b in itertools.combinations(cnn_with_default.blocks, 2):
        #     self.assertIsNot(a.pool, b.pool)

        for a, b in itertools.combinations(cnn_with_pool_module.blocks[1:], 2):
            self.assertIsNot(a.pool, b.pool)

        # for a, b in itertools.combinations(cnn_with_pool_class.blocks[1:], 2):
        #    self.assertIsNot(a.pool, b.pool)

        for a, b in itertools.combinations(cnn_with_pool_layer.blocks[1:], 2):
            self.assertIsNot(a.pool, b.pool)

    def test_cnn_configure(self):
        cnn_with_pool_module = ConvolutionalNeuralNetwork(3, [4, 4, 4], 1)
        cnn_with_pool_module.configure(pool=nn.MaxPool2d(2))
        cnn_with_pool_module.build()

        for block in cnn_with_pool_module.blocks[1:]:
            self.assertIsInstance(block.pool, nn.MaxPool2d)

    def test_create_twice(self):

        class Wrapper(Application):
            def __init__(self, model, **kwargs):
                self.model = model
                super().__init__(**kwargs)

        cnn = ConvolutionalNeuralNetwork(3, [4, 4], 1, pool=torch.nn.MaxPool2d(2))
        app_1 = Wrapper(cnn).create()
        app_2 = Wrapper(cnn).create()

        self.assertListEqual(
            app_2.model.blocks[1].order,
            ["pool", "layer", "activation"],
        )
