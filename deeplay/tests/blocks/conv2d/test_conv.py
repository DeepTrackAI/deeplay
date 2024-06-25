import unittest
from itertools import product

import torch
import torch.nn as nn

from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.external.layer import Layer

from deeplay.ops.attention.self import MultiheadSelfAttention
from deeplay.ops.merge import Add


class TestConv2dBlock(unittest.TestCase):

    def test_init(self):
        block = Conv2dBlock(in_channels=1, out_channels=1)
        self.assertListEqual(block.order, ["layer"])

    def test_init_activation(self):
        activation = Layer(nn.ReLU)
        block = Conv2dBlock(in_channels=1, out_channels=1, activation=activation)
        self.assertListEqual(block.order, ["layer", "activation"])
        self.assertEqual(block.activation, activation)

    def test_init_normalization(self):
        normalization = Layer(nn.BatchNorm2d, num_features=1)
        block = Conv2dBlock(in_channels=1, out_channels=1, normalization=normalization)
        self.assertListEqual(
            block.order,
            [
                "layer",
                "normalization",
            ],
        )
        self.assertEqual(block.normalization, normalization)

    def test_init_order(self):
        order = ["normalization", "layer"]
        block = Conv2dBlock(
            in_channels=1,
            out_channels=1,
            order=order,
            normalization=Layer(nn.BatchNorm2d, num_features=1),
        )
        self.assertListEqual(block.order, order)

    def test_normalized(self):
        block = Conv2dBlock(in_channels=1, out_channels=2)
        block.normalized(normalization=nn.BatchNorm2d)
        block.build()
        self.assertEqual(block.normalization.num_features, 2)

    def test_normalized_prepend(self):
        block = Conv2dBlock(in_channels=1, out_channels=2)
        block.normalized(normalization=nn.BatchNorm2d, mode="prepend")
        block.build()
        self.assertEqual(block.normalization.num_features, 1)

    def test_pooled(self):
        block = Conv2dBlock(in_channels=1, out_channels=1).pooled().build()
        self.assertIsInstance(block.pool, nn.MaxPool2d)

    def test_strided(self):
        block = Conv2dBlock(in_channels=1, out_channels=1, padding=1).strided(2).build()
        self.assertEqual(block.layer.stride, (2, 2))

    def test_pooled_strided(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1, padding=1)
            .pooled()
            .strided(2)
            .build()
        )
        self.assertNotIn("pool", block.order)
        self.assertEqual(block.layer.stride, (2, 2))

    def test_pooled_strided_prepend(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1)
            .pooled()
            .strided(2, remove_pool=False)
            .build()
        )
        self.assertIn("pool", block.order)
        self.assertEqual(block.layer.stride, (2, 2))

    def test_shortcut(self):
        block = Conv2dBlock(in_channels=1, out_channels=1, padding=1).shortcut().build()
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start.layer, nn.Identity)

    def test_shortcut_different_num_channels(self):
        block = Conv2dBlock(in_channels=1, out_channels=2, padding=1).shortcut().build()
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start, Conv2dBlock)
        self.assertEqual(block.shortcut_start.layer.in_channels, 1)
        self.assertEqual(block.shortcut_start.layer.out_channels, 2)

    def test_strided_shortcut(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1, padding=1)
            .strided(2)
            .shortcut()
            .build()
        )
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start, Conv2dBlock)
        self.assertEqual(block.shortcut_start.layer.stride, (2, 2))

    def test_shortcut_strided(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1, padding=1)
            .shortcut()
            .strided(2)
            .build()
        )
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start, Conv2dBlock)
        self.assertEqual(block.shortcut_start.layer.stride, (2, 2))

    def test_multi_strided(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1, padding=1)
            .multi(3)
            .strided(2)
            .build()
        )
        self.assertEqual(block.blocks[0].layer.stride, (2, 2))
        for b in block.blocks[1:]:
            self.assertEqual(b.layer.stride, (1, 1))

    # def test_strided_multi(self):
    #     block = Conv2dBlock(in_channels=1, out_channels=1).strided(2).multi(3).build()
    #     for b in block.blocks:
    #         self.assertEqual(b.layer.stride, (2, 2))

    def test_multi_multi(self):
        block = Conv2dBlock(in_channels=1, out_channels=2, padding=1).multi(2)
        block.blocks[0].multi(2)
        block.blocks[1].multi(2)
        block.build()

        self.assertEqual(block.blocks[0].blocks[0].layer.in_channels, 1)
        self.assertEqual(block.blocks[0].blocks[0].layer.out_channels, 2)

        self.assertEqual(block.blocks[0].blocks[1].layer.in_channels, 2)
        self.assertEqual(block.blocks[0].blocks[1].layer.out_channels, 2)

        self.assertEqual(block.blocks[1].blocks[0].layer.in_channels, 2)
        self.assertEqual(block.blocks[1].blocks[0].layer.out_channels, 2)

        self.assertEqual(block.blocks[1].blocks[1].layer.in_channels, 2)
        self.assertEqual(block.blocks[1].blocks[1].layer.out_channels, 2)

    def test_style_residual(self):
        block = Conv2dBlock(in_channels=1, out_channels=1).style("residual").build()
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start.layer, nn.Identity)
        self.assertEqual(len(block.blocks), 2)
        self.assertEqual(
            block.blocks[0].order,
            [
                "layer",
                "activation",
                "normalization",
            ],
        )
        self.assertEqual(
            block.blocks[1].order, ["layer", "activation", "normalization"]
        )

    def test_style_residual_strided(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1)
            .style("residual")
            .strided(2)
            .build()
        )
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertIsInstance(block.shortcut_start, Conv2dBlock)
        self.assertEqual(block.shortcut_start.layer.stride, (2, 2))

    def test_style_residual_orders(self):
        l, a, n, d = "layer", "activation", "normalization", "dropout"
        orders_and_expected = [
            ("lanlan|", ([[l, a, n], [l, a, n]], [])),
            ("lanl|an", ([[l, a, n], [l]], [a, n])),
            ("lllaaa|", ([[l], [l], [l, a], [a], [a]], [])),
            ("|na", ([], [n, a])),
            ("nadlal|an", ([[n, a, d, l], [a, l]], [a, n])),
        ]
        for order, (block_orders, after_skip) in orders_and_expected:
            with self.subTest(order=order):
                block = (
                    Conv2dBlock(in_channels=1, out_channels=1)
                    .style("residual", order=order)
                    .build()
                )
                self.assertEqual(len(block.blocks), len(block_orders))
                for i, b in enumerate(block.blocks):
                    self.assertListEqual(b.order, block_orders[i])
                self.assertListEqual(
                    block.order,
                    ["shortcut_start", "blocks", "shortcut_end"] + after_skip,
                )

    def test_style_spatial_self_attention(self):
        block = (
            Conv2dBlock(in_channels=2, out_channels=2)
            .style("spatial_self_attention")
            .build(torch.randn(1, 4, 4, 2))
        )
        self.assertEqual(
            block.order,
            [
                "shortcut_start",
                "normalization",
                "layer",
                "shortcut_end",
            ],
        )

        x = torch.randn(1, 4, 4, 2)  # expects (B, H, W, C)
        output = block(x)
        self.assertEqual(output.shape, x.shape)
        self.assertIsInstance(block.normalization, nn.LayerNorm)
        self.assertIsInstance(block.layer, MultiheadSelfAttention)

    def test_style_spatial_self_attention_channel_last(self):
        block = (
            Conv2dBlock(in_channels=1, out_channels=1)
            .style("spatial_self_attention", to_channel_last=True)
            .build()
        )
        block.build()
        self.assertEqual(
            block.order,
            [
                "channel_last",
                "shortcut_start",
                "normalization",
                "layer",
                "shortcut_end",
                "channel_first",
            ],
        )

        x = torch.randn(1, 1, 4, 4)
        output = block(x)
        self.assertEqual(output.shape, x.shape)
