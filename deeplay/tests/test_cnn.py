import unittest
import torch
import torch.nn as nn
from deeplay import ConvolutionalNeuralNetwork, Layer, LayerList


class TestComponentCNN(unittest.TestCase):
    ...

    def test_cnn_defaults(self):
        cnn = ConvolutionalNeuralNetwork(3, [4], 1)
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)

        self.assertEqual(cnn.blocks[0].layer.in_channels, 3)
        self.assertEqual(cnn.blocks[0].layer.out_channels, 4)

        self.assertEqual(cnn.output_block.layer.in_channels, 4)
        self.assertEqual(cnn.output_block.layer.out_channels, 1)

        # test on a batch of 2
        x = torch.randn(2, 3, 5, 5)
        y = cnn(x)
        self.assertEqual(y.shape, (2, 1, 5, 5))

    def test_cnn_lazy_input(self):
        cnn = ConvolutionalNeuralNetwork(None, [4], 1).build()
        self.assertEqual(len(cnn.blocks), 2)

        self.assertEqual(cnn.blocks[0].layer.in_channels, 0)
        self.assertEqual(cnn.blocks[0].layer.out_channels, 4)
        self.assertEqual(cnn.output_block.layer.in_channels, 4)
        self.assertEqual(cnn.output_block.layer.out_channels, 1)

        # test on a batch of 2
        x = torch.randn(2, 3, 5, 5)
        y = cnn(x)
        self.assertEqual(y.shape, (2, 1, 5, 5))

    def test_cnn_change_depth(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(hidden_channels=[4, 4])
        cnn.build()
        self.assertEqual(len(cnn.blocks), 3)

    def test_change_activation(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=nn.Sigmoid)
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output_block.activation, nn.Sigmoid)

    def test_change_out_activation_Layer(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=Layer(nn.Sigmoid))
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output_block.activation, nn.Sigmoid)

    def test_change_out_activation_instance(self):
        cnn = ConvolutionalNeuralNetwork(2, [4], 3)
        cnn.configure(out_activation=nn.Sigmoid())
        cnn.build()
        self.assertEqual(len(cnn.blocks), 2)
        self.assertIsInstance(cnn.output_block.activation, nn.Sigmoid)

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

        self.assertIs(cnn.blocks[0], cnn.output_block)
        self.assertIs(cnn.blocks[0], cnn.input_block)

    def test_zero_out_channels(self):
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetwork(
                in_channels=3, hidden_channels=[32, 64], out_channels=0
            )
