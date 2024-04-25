import unittest

import torch
import torch.nn as nn

from deeplay.models.generators.cyclegan import CycleGANResnetGenerator


class TestCycleGANResnetGenerator(unittest.TestCase):

    def test_init(self):
        generator = CycleGANResnetGenerator().build()
        data = torch.randn(1, 1, 32, 32)
        output = generator(data)

        self.assertEqual(output.shape, (1, 1, 32, 32))

    def test_bottleneck_n_layers(self):
        generator = CycleGANResnetGenerator(n_residual_blocks=5).build()
        self.assertEqual(len(generator.bottleneck.blocks), 5)
        data = torch.randn(1, 1, 32, 32)
        output = generator(data)

        self.assertEqual(output.shape, (1, 1, 32, 32))
