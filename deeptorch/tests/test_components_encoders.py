import unittest
import torch
import torch.nn as nn
from .. import Config, Node, ConvolutionalEncoder


class TestComponentsEncoders(unittest.TestCase):

    def test_encoder_defaults(self):
        
        encoder = ConvolutionalEncoder()
        self.assertEqual(encoder.depth, 4)
        self.assertEqual(len(encoder.blocks), 4)
        self.assertListEqual(
            [encoder.blocks[i]["layer"].out_channels for i in range(encoder.depth)],
            [8, 16, 32, 64]
        )

        y = encoder(torch.randn(1, 1, 28, 28))
        self.assertEqual(y.shape, (1, 64, 1, 1))

    def test_encoder_shallow(self):

        encoder = ConvolutionalEncoder(depth=2)
        print(encoder.config)
        self.assertEqual(encoder.depth, 2)
        self.assertEqual(len(encoder.blocks), 2)
        self.assertListEqual(
            [encoder.blocks[i]["layer"].out_channels for i in range(encoder.depth)],
            [8, 16]
        )

        y = encoder(torch.randn(1, 1, 28, 28))
        self.assertEqual(y.shape, (1, 16, 7, 7))

    def test_encoder_with_padding(self):
            
            encoder = ConvolutionalEncoder(depth=2, blocks=Config().layer.padding(0))
            self.assertEqual(encoder.depth, 2)
            self.assertEqual(len(encoder.blocks), 2)
            self.assertListEqual(
                [encoder.blocks[i]["layer"].out_channels for i in range(encoder.depth)],
                [8, 16]
            )
    
            y = encoder(torch.randn(1, 1, 28, 28))
            self.assertEqual(y.shape, (1, 16, 5, 5))


