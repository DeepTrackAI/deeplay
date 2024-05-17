import unittest

import torch
import torch.nn as nn

from deeplay.models.generators.cyclegan import CycleGANResnetGenerator


class TestCycleGANResnetGenerator(unittest.TestCase):

    def test_init(self):
        generator = CycleGANResnetGenerator().build()

        # Encoder
        self.assertEqual(len(generator.encoder.blocks), 3)
        self.assertTrue(
            all(
                isinstance(generator.encoder.blocks.normalization[i], nn.InstanceNorm2d)
                for i in range(3)
            )
        )
        self.assertTrue(
            all(
                isinstance(generator.encoder.blocks.activation[i], nn.ReLU)
                for i in range(3)
            )
        )

        # Decoder
        self.assertEqual(len(generator.decoder.blocks), 3)
        self.assertTrue(
            all(
                isinstance(
                    generator.decoder.blocks[:-1].normalization[i], nn.InstanceNorm2d
                )
                for i in range(2)
            )
        )
        self.assertTrue(
            all(
                isinstance(generator.decoder.blocks.activation[i], nn.ReLU)
                for i in range(2)
            )
        )
        self.assertTrue(isinstance(generator.decoder.blocks[-1].activation, nn.Tanh))

        data = torch.randn(1, 1, 32, 32)
        output = generator(data)

        self.assertEqual(output.shape, (1, 1, 32, 32))

    def test_bottleneck_n_layers(self):
        generator = CycleGANResnetGenerator(n_residual_blocks=5).build()
        self.assertEqual(len(generator.bottleneck.blocks), 5)
        data = torch.randn(1, 1, 32, 32)
        output = generator(data)

        self.assertEqual(output.shape, (1, 1, 32, 32))
