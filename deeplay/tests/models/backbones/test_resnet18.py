import unittest
import torch
import torch.nn as nn


from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.models.backbones.resnet18 import BackboneResnet18


class TestResnet18(unittest.TestCase):

    def test_init(self):
        model = BackboneResnet18(in_channels=3)
        model.build()

    def test_re_init(self):
        model = BackboneResnet18(in_channels=3)
        model.__construct__()
        model.build()

    def test_num_params(self):
        model = BackboneResnet18(in_channels=1)
        model.build()
        num_params = sum(p.numel() for p in model.parameters())

        self.assertEqual(num_params, 11174976)

    def test_style_resnet18_input(self):
        block = Conv2dBlock(3, 64).style("resnet18_input").build()
        self.assertEqual(block.layer.kernel_size, (7, 7))
        self.assertEqual(block.layer.stride, (2, 2))
        self.assertEqual(block.layer.padding, (3, 3))
        self.assertEqual(block.layer.bias, None)
        self.assertIsInstance(block.normalization, nn.BatchNorm2d)
        self.assertIsInstance(block.activation, nn.ReLU)
        self.assertIsInstance(block.pool, nn.MaxPool2d)

    def test_style_resnet(self):
        block = Conv2dBlock(64, 64).style("resnet", stride=1).build()
        self.assertEqual(len(block.blocks), 2)
        self.assertListEqual(block.order, ["blocks"])
        self.assertListEqual(
            block.blocks[0].order,
            ["shortcut_start", "blocks", "shortcut_end", "activation"],
        )
        self.assertListEqual(
            block.blocks[1].order,
            ["shortcut_start", "blocks", "shortcut_end", "activation"],
        )
        self.assertListEqual(
            block.blocks[0].blocks[0].order,
            ["layer", "normalization", "activation"],
        )
        self.assertListEqual(
            block.blocks[0].blocks[1].order, ["layer", "normalization"]
        )
        self.assertEqual(block.blocks[0].blocks[0].layer.in_channels, 64)
        self.assertEqual(block.blocks[0].blocks[0].layer.out_channels, 64)
        self.assertEqual(block.blocks[0].blocks[1].layer.in_channels, 64)
        self.assertEqual(block.blocks[0].blocks[1].layer.out_channels, 64)
        self.assertIsInstance(block.blocks[0].shortcut_start.layer, nn.Identity)
        # self.assertIsInstance(block.blocks[0].shortcut_start.activation, nn.Identity)
        self.assertNotIn("normalization", block.blocks[0].shortcut_start.order)

    def test_style_resnet_64_128(self):
        block = Conv2dBlock(64, 128).style("resnet", stride=2).build()
        self.assertEqual(len(block.blocks), 2)
        self.assertListEqual(block.order, ["blocks"])
        self.assertListEqual(
            block.blocks[0].order,
            ["shortcut_start", "blocks", "shortcut_end", "activation"],
        )
        self.assertListEqual(
            block.blocks[1].order,
            ["shortcut_start", "blocks", "shortcut_end", "activation"],
        )
        self.assertListEqual(
            block.blocks[0].blocks[0].order,
            ["layer", "normalization", "activation"],
        )
        self.assertListEqual(
            block.blocks[0].blocks[1].order, ["layer", "normalization"]
        )
        self.assertEqual(block.blocks[0].blocks[0].layer.in_channels, 64)
        self.assertEqual(block.blocks[0].blocks[0].layer.out_channels, 128)
        self.assertEqual(block.blocks[0].blocks[1].layer.in_channels, 128)
        self.assertEqual(block.blocks[0].blocks[1].layer.out_channels, 128)
        self.assertEqual(block.blocks[1].blocks[0].layer.in_channels, 128)
        self.assertEqual(block.blocks[1].blocks[0].layer.out_channels, 128)
        self.assertEqual(block.blocks[1].blocks[1].layer.in_channels, 128)
        self.assertEqual(block.blocks[1].blocks[1].layer.out_channels, 128)
        self.assertIsInstance(block.blocks[0].shortcut_start.layer, nn.Conv2d)
        self.assertIsInstance(
            block.blocks[0].shortcut_start.normalization, nn.BatchNorm2d
        )
        self.assertEqual(block.blocks[0].shortcut_start.layer.in_channels, 64)
        self.assertEqual(block.blocks[0].shortcut_start.layer.out_channels, 128)
        # self.assertIsInstance(block.blocks[0].shortcut_start.activation, nn.Identity)

    def test_correct_structure(self):
        model = BackboneResnet18(in_channels=3)
        model.build()
        self.assertEqual(len(model.blocks), 5)
        self.assertListEqual(
            model.blocks[0].order, ["layer", "normalization", "activation", "pool"]
        )
        self.assertEqual(model.blocks[0].layer.kernel_size, (7, 7))
        self.assertEqual(model.blocks[0].layer.stride, (2, 2))
        self.assertEqual(model.blocks[0].layer.padding, (3, 3))
        self.assertEqual(model.blocks[0].layer.bias, None)
        self.assertIsInstance(model.blocks[0].normalization, nn.BatchNorm2d)
        self.assertIsInstance(model.blocks[0].activation, nn.ReLU)
        self.assertIsInstance(model.blocks[0].pool, nn.MaxPool2d)

        # block 1
        in_channels = [3, 64, 64, 128, 256]
        out_channels = [64, 64, 128, 256, 512]
        for idx in range(1, 4):
            self.assertEqual(len(model.blocks[idx].blocks), 2)
            self.assertListEqual(model.blocks[idx].order, ["blocks"])
            self.assertListEqual(
                model.blocks[idx].blocks[0].order,
                ["shortcut_start", "blocks", "shortcut_end", "activation"],
            )
            self.assertListEqual(
                model.blocks[idx].blocks[1].order,
                ["shortcut_start", "blocks", "shortcut_end", "activation"],
            )
            self.assertListEqual(
                model.blocks[idx].blocks[0].blocks[0].order,
                ["layer", "normalization", "activation"],
            )
            self.assertListEqual(
                model.blocks[idx].blocks[0].blocks[1].order, ["layer", "normalization"]
            )
            self.assertEqual(
                model.blocks[idx].blocks[0].blocks[0].layer.in_channels,
                in_channels[idx],
            )
            self.assertEqual(
                model.blocks[idx].blocks[0].blocks[0].layer.out_channels,
                out_channels[idx],
            )
            self.assertEqual(
                model.blocks[idx].blocks[0].blocks[1].layer.in_channels,
                out_channels[idx],
            )
            self.assertEqual(
                model.blocks[idx].blocks[0].blocks[1].layer.out_channels,
                out_channels[idx],
            )
            if idx == 1:
                self.assertIsInstance(
                    model.blocks[idx].blocks[0].shortcut_start.layer, nn.Identity
                )
                self.assertFalse(
                    hasattr(model.blocks[idx].blocks[0].shortcut_start, "activation")
                )
                self.assertNotIn(
                    "normalization", model.blocks[idx].blocks[0].shortcut_start.order
                )
            else:
                self.assertIsInstance(
                    model.blocks[idx].blocks[0].shortcut_start.layer, nn.Conv2d
                )
                self.assertIsInstance(
                    model.blocks[idx].blocks[0].shortcut_start.normalization,
                    nn.BatchNorm2d,
                )
                self.assertEqual(
                    model.blocks[idx].blocks[0].shortcut_start.layer.in_channels,
                    in_channels[idx],
                )
                self.assertEqual(
                    model.blocks[idx].blocks[0].shortcut_start.layer.out_channels,
                    out_channels[idx],
                )
                self.assertFalse(
                    hasattr(model.blocks[idx].blocks[0].shortcut_start, "activation")
                )

    def test_forward(self):
        model = BackboneResnet18(in_channels=3)
        model.build()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        self.assertEqual(y.shape, (1, 512, 7, 7))
