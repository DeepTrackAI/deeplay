import unittest
import torch
import torch.nn as nn
from .. import Config, Layer, ImageToImageEncoder


# class TestComponentsEncoders(unittest.TestCase):
#     def test_encoder_defaults(self):
#         encoder = ImageToImageEncoder()
#         self.assertEqual(encoder.depth, 4)
#         self.assertEqual(len(encoder.encoder_blocks), 4)
#         self.assertListEqual(
#             [
#                 encoder.encoder_blocks[i]["layer"].out_channels
#                 for i in range(encoder.depth)
#             ],
#             [16, 32, 64, 128],
#         )

#         y = encoder(torch.randn(1, 1, 28, 28))
#         self.assertEqual(y.shape, (1, 128, 1, 1))

#     def test_encoder_shallow(self):
#         encoder = ImageToImageEncoder(depth=2)

#         self.assertEqual(encoder.depth, 2)
#         self.assertEqual(len(encoder.encoder_blocks), 2)
#         self.assertListEqual(
#             [
#                 encoder.encoder_blocks[i]["layer"].out_channels
#                 for i in range(encoder.depth)
#             ],
#             [16, 32],
#         )

#         y = encoder(torch.randn(1, 1, 28, 28))
#         self.assertEqual(y.shape, (1, 32, 7, 7))

#     def test_encoder_with_padding(self):
#         encoder = ImageToImageEncoder(depth=2, encoder_blocks=Config().layer.padding(0))
#         self.assertEqual(encoder.depth, 2)
#         self.assertEqual(len(encoder.encoder_blocks), 2)
#         self.assertListEqual(
#             [
#                 encoder.encoder_blocks[i]["layer"].out_channels
#                 for i in range(encoder.depth)
#             ],
#             [16, 32],
#         )

#         y = encoder(torch.randn(1, 1, 28, 28))
#         self.assertEqual(y.shape, (1, 32, 4, 4))
