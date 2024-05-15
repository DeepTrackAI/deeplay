import unittest

import torch
import torch.nn as nn

from deeplay.models.generators.dcgan import DCGANGenerator


class TestDCGANGenerator(unittest.TestCase):
    ...

    def test_generator_defaults(self):

        generator = DCGANGenerator()
        generator.build()

        self.assertEqual(len(generator.blocks), 5)
        self.assertEqual(
            [generator.blocks[i].layer.kernel_size for i in range(5)], [(4, 4)] * 5
        )

        self.assertEqual(generator.blocks[0].layer.stride, (1, 1))
        self.assertEqual(
            [generator.blocks[i].layer.stride for i in range(1, 5)], [(2, 2)] * 4
        )

        self.assertEqual(generator.blocks[0].layer.padding, (0, 0))
        self.assertEqual(generator.blocks[-1].layer.padding, (1, 1))

        self.assertTrue(
            all(isinstance(generator.blocks.activation[i], nn.ReLU) for i in range(4))
        )
        self.assertTrue(isinstance(generator.blocks[-1].activation, nn.Tanh))

        self.assertTrue(
            all(
                isinstance(generator.blocks[:-1].normalization[i], nn.BatchNorm2d)
                for i in range(4)
            )
        )

        self.assertTrue(isinstance(generator.label_embedding, nn.Identity))

        # Test on a batch of 2
        x = torch.rand(2, 100, 1, 1)
        output = generator(x, y=None)
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_conditional_generator_defaults(self):

        generator = DCGANGenerator(class_conditioned_model=True)
        generator.build()

        self.assertTrue(isinstance(generator.label_embedding, nn.Embedding))
        self.assertEqual(generator.label_embedding.num_embeddings, 10)
        self.assertEqual(generator.label_embedding.embedding_dim, 100)

        # Test on a batch of 2
        x = torch.rand(2, 100, 1, 1)
        y = torch.randint(0, 10, (2,))
        output = generator(x, y)
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_weight_initialization(self):

        generator = DCGANGenerator()
        generator.build()

        for m in generator.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                self.assertAlmostEqual(m.weight.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(m.weight.data.std().item(), 0.02, places=2)

    def test_weight_initialization_conditional(self):

        generator = DCGANGenerator(class_conditioned_model=True)
        generator.build()

        for m in generator.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d, nn.Embedding)):
                self.assertAlmostEqual(m.weight.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(m.weight.data.std().item(), 0.02, places=2)
