import unittest

import torch
import torch.nn as nn
from deeplay import ConvolutionalEncoderDecoder2d, Layer, LayerList

import itertools


class TestComponentEncDec(unittest.TestCase):
    ...

    def test_encdec_defaults(self):
        encdec = ConvolutionalEncoderDecoder2d(
            in_channels=3,
            encoder_channels=[8, 16, 32],
            decoder_channels=[16, 8],
            out_channels=1,
        )
        encdec.build()
        self.assertEqual(len(encdec.encoder.blocks), 3)
        self.assertEqual(len(encdec.decoder.blocks), 3)

        self.assertEqual(encdec.encoder.blocks[0].layer.in_channels, 3)
        self.assertEqual(encdec.encoder.blocks[1].layer.out_channels, 16)
        self.assertEqual(encdec.encoder.blocks[0].layer.in_channels, 3)
        self.assertEqual(encdec.decoder.blocks[-2].layer.out_channels, 8)

        self.assertEqual(encdec.decoder.output.layer.out_channels, 1)
        # test on a batch of 2
        x = torch.randn(2, 3, 64, 64)
        y = encdec(x)
        self.assertEqual(y.shape, (2, 1, 64, 64))

    def test_change_act(self):
        encdec = ConvolutionalEncoderDecoder2d(
            in_channels=3,
            encoder_channels=[8, 16, 32],
            decoder_channels=[16, 8],
            out_channels=1,
        )
        encdec.configure(out_activation=nn.Sigmoid)
        encdec.build()

        self.assertEqual(len(encdec.blocks), 7)
        self.assertIsInstance(encdec.decoder.output.activation, nn.Sigmoid)

    def test_change_out_act_Layer(self):
        encdec = ConvolutionalEncoderDecoder2d(
            in_channels=3,
            encoder_channels=[8, 16, 32],
            decoder_channels=[16, 8],
            out_channels=1,
        )
        encdec.configure(out_activation=Layer(nn.Sigmoid))
        encdec.build()

        self.assertEqual(len(encdec.blocks), 7)
        self.assertIsInstance(encdec.decoder.output.activation, nn.Sigmoid)

    def test_change_out_act_instance(self):
        encdec = ConvolutionalEncoderDecoder2d(
            in_channels=3,
            encoder_channels=[8, 16, 32],
            decoder_channels=[16, 8],
            out_channels=1,
        )
        encdec.configure(out_activation=nn.Sigmoid())
        encdec.build()

        self.assertEqual(len(encdec.blocks), 7)
        self.assertIsInstance(encdec.decoder.output.activation, nn.Sigmoid)

    # def test_default_values_initialization(self):
    #     cnn = ConvolutionalNeuralNetwork(
    #         in_channels=None, hidden_channels=[32, 32], out_channels=1
    #     )
    #     self.assertIsNone(cnn.in_channels)
    #     self.assertEqual(cnn.hidden_channels, [32, 32])
    #     self.assertEqual(cnn.out_channels, 1)
    #     self.assertIsInstance(cnn.blocks, LayerList)

    # def test_empty_hidden_channels(self):
    #     cnn = ConvolutionalNeuralNetwork(
    #         in_channels=3, hidden_channels=[], out_channels=1
    #     ).build()
    #     self.assertEqual(cnn.blocks[0].layer.in_channels, 3)
    #     self.assertEqual(cnn.blocks[0].layer.out_channels, 1)

    #     self.assertIs(cnn.blocks[0], cnn.input)
    #     self.assertIs(cnn.blocks[0], cnn.output)

    # def test_zero_out_channels(self):
    #     with self.assertRaises(ValueError):
    #         ConvolutionalNeuralNetwork(
    #             in_channels=3, hidden_channels=[32, 64], out_channels=0
    #         )

    # def test_all_cnn_blocks_are_not_same_object(self):
    #     cnn_with_default = ConvolutionalNeuralNetwork(3, [4, 4, 4], 1)
    #     cnn_with_pool_module = ConvolutionalNeuralNetwork(
    #         3, [4, 4, 4], 1, pool=nn.MaxPool2d
    #     )
    #     cnn_with_pool_class = ConvolutionalNeuralNetwork(
    #         3, [4, 4, 4], 1, pool=nn.MaxPool2d(2)
    #     )
    #     cnn_with_pool_layer = ConvolutionalNeuralNetwork(
    #         3, [4, 4, 4], 1, pool=Layer(nn.MaxPool2d)
    #     )

    #     for a, b in itertools.combinations(cnn_with_default.blocks, 2):
    #         self.assertIsNot(a.pool, b.pool)

    #     for a, b in itertools.combinations(cnn_with_pool_module.blocks, 2):
    #         self.assertIsNot(a.pool, b.pool)

    #     for a, b in itertools.combinations(cnn_with_pool_class.blocks[1:], 2):
    #         self.assertIs(a.pool, b.pool)

    #     for a, b in itertools.combinations(cnn_with_pool_layer.blocks, 2):
    #         self.assertIsNot(a.pool, b.pool)
