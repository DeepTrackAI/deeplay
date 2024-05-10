import unittest

import torch
import torch.nn as nn

from deeplay.models.discriminators.cyclegan import CycleGANDiscriminator


class TestCycleGANDiscriminator(unittest.TestCase):

    def test_discriminator_defaults(self):

        discriminator = CycleGANDiscriminator().build()

        self.assertEqual(len(discriminator.blocks), 5)
        self.assertTrue(
            all(
                isinstance(discriminator.blocks.normalization[i], nn.InstanceNorm2d)
                for i in range(1, 4)
            )
        )
        self.assertTrue(
            all(
                isinstance(discriminator.blocks.activation[i], nn.LeakyReLU)
                for i in range(4)
            )
        )
        self.assertTrue(isinstance(discriminator.blocks[-1].activation, nn.Sigmoid))

        # Test on a batch of 2
        x = torch.rand(2, 1, 256, 256)
        output = discriminator(x)
        self.assertEqual(output.shape, (2, 1, 30, 30))
