import unittest

import torch
import torch.nn as nn

from deeplay.models.discriminators.dcgan import DCGANDiscriminator


class TestDCGANDiscriminator(unittest.TestCase):
    ...

    def test_discriminator_defaults(self):

        discriminator = DCGANDiscriminator()
        discriminator.build()

        self.assertEqual(len(discriminator.blocks), 5)
        self.assertEqual(
            [discriminator.blocks[i].layer.kernel_size for i in range(5)], [(4, 4)] * 5
        )

        self.assertEqual(
            [discriminator.blocks[i].layer.stride for i in range(5)], [(2, 2)] * 5
        )

        self.assertEqual(
            [discriminator.blocks[i].layer.padding for i in range(4)], [(1, 1)] * 4
        )
        self.assertEqual(discriminator.blocks[-1].layer.padding, (0, 0))

        self.assertTrue(
            all(
                isinstance(discriminator.blocks[i].activation, nn.LeakyReLU)
                for i in range(4)
            )
        )
        self.assertTrue(isinstance(discriminator.blocks[-1].activation, nn.Sigmoid))

        self.assertTrue(
            all(
                isinstance(discriminator.blocks[1:-1].normalization[i], nn.BatchNorm2d)
                for i in range(3)
            )
        )

        self.assertTrue(isinstance(discriminator.label_embedding, nn.Identity))

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        output = discriminator(x, y=None)
        self.assertEqual(output.shape, (2, 1, 1, 1))

    def test_conditional_discriminator_defaults(self):

        discriminator = DCGANDiscriminator(class_conditioned_model=True)
        discriminator.build()

        self.assertTrue(
            isinstance(discriminator.label_embedding.embedding, nn.Embedding)
        )
        self.assertTrue(isinstance(discriminator.label_embedding.layer, nn.Linear))
        self.assertTrue(
            isinstance(discriminator.label_embedding.activation, nn.LeakyReLU)
        )

        self.assertTrue(discriminator.label_embedding.embedding.num_embeddings, 10)
        self.assertTrue(discriminator.label_embedding.layer.in_features, 100)
        self.assertTrue(discriminator.label_embedding.layer.out_features, 64 * 64)

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        y = torch.randint(0, 10, (2,))
        output = discriminator(x, y)
        self.assertEqual(output.shape, (2, 1, 1, 1))

    def test_weight_initialization(self):

        generator = DCGANDiscriminator()
        generator.build()

        for m in generator.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                self.assertAlmostEqual(m.weight.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(m.weight.data.std().item(), 0.02, places=2)

    def test_weight_initialization_conditional(self):

        generator = DCGANDiscriminator(class_conditioned_model=True)
        generator.build()

        for m in generator.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Embedding, nn.Linear)):
                self.assertAlmostEqual(m.weight.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(m.weight.data.std().item(), 0.02, places=2)
