import unittest

import torch
import torch.nn as nn

from deeplay.models.discriminators.dcgan import DCGANDiscriminator


class TestDCGANGenerator(unittest.TestCase):

    def test_init(self):
        discr = DCGANDiscriminator().build()
        data = torch.randn(1, 1, 64, 64)
        output = discr(data)

        self.assertEqual(output.shape, (1, 1, 1, 1))

    def test_conditioned(self):
        discr = DCGANDiscriminator(class_conditioned_model=True).build()
        data = torch.randn(1, 1, 64, 64)
        labels = torch.randint(0, 10, (1,))
        output = discr(data, labels)

        self.assertEqual(output.shape, (1, 1, 1, 1))
