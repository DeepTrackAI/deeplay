import unittest

import torch
import torch.nn as nn

from deeplay.models.generators.dcgan import DCGANGenerator


class TestDCGANGenerator(unittest.TestCase):

    def test_init(self):
        generator = DCGANGenerator().build()
        data = torch.randn(1, 100, 1, 1)
        output = generator(data)

        self.assertEqual(output.shape, (1, 1, 64, 64))

    def test_conditioned(self):
        generator = DCGANGenerator(class_conditioned_model=True).build()
        data = torch.randn(1, 100, 1, 1)
        labels = torch.randint(0, 10, (1,))
        output = generator(data, labels)

        self.assertEqual(output.shape, (1, 1, 64, 64))
