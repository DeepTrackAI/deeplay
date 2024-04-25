from itertools import product
from re import T
import unittest

import torch
import torch.nn as nn

from deeplay.blocks.base import BaseBlock
from deeplay.ops.merge import Add
from deeplay.external import layer
from deeplay.external.layer import Layer


class TestBaseBlock(unittest.TestCase):

    def test_base_block_init(self):
        block = BaseBlock()
        self.assertListEqual(block.order, [])

    def test_base_block_init_layer(self):
        layer = Layer(nn.Linear, 1, 1)
        block = BaseBlock(layer=layer)
        self.assertListEqual(block.order, ["layer"])
        self.assertEqual(block.layer, layer)

        block.build()

        self.assertIsInstance(block.layer, nn.Linear)

    def test_base_block_init_layer_activation(self):
        layer = Layer(nn.Linear, 1, 1)
        activation = Layer(nn.ReLU)
        block = BaseBlock(layer=layer, activation=activation)
        self.assertListEqual(block.order, ["layer", "activation"])
        self.assertEqual(block.layer, layer)
        self.assertEqual(block.activation, activation)

        block.build()

        self.assertIsInstance(block.layer, nn.Linear)
        self.assertIsInstance(block.activation, nn.ReLU)

    def test_base_block_init_layer_activation_order(self):
        layer = Layer(nn.Linear, 1, 1)
        activation = Layer(nn.ReLU)
        order = ["activation", "layer"]
        block = BaseBlock(layer=layer, activation=activation, order=order)
        self.assertListEqual(block.order, order)
        self.assertEqual(block.layer, layer)
        self.assertEqual(block.activation, activation)

        block.build()

        self.assertIsInstance(block.layer, nn.Linear)
        self.assertIsInstance(block.activation, nn.ReLU)

    def test_base_block_activated(self):
        init_with_activation = [True, False]
        mode = ["replace", "append", "prepend", "insert"]
        wrap_with_layer = [True, False]
        for iwa, m, wrap in product(init_with_activation, mode, wrap_with_layer):
            with self.subTest(init_with_activation=iwa, mode=m, wrap_with_layer=wrap):
                layer = Layer(nn.Linear, 1, 1)
                activation = Layer(nn.Identity)
                new_activation = Layer(nn.ReLU) if wrap else nn.ReLU

                if iwa:
                    block = BaseBlock(layer=layer, activation=activation)
                else:
                    block = BaseBlock(layer=layer)

                block.activated(
                    activation=new_activation,
                    mode=m,
                    after="layer" if m == "insert" else None,
                )

                block.build()

                if not iwa:
                    if m == "replace":
                        self.assertListEqual(block.order, ["layer"])
                    elif m == "append":
                        self.assertListEqual(block.order, ["layer", "activation"])
                        self.assertIsInstance(block.activation, nn.ReLU)
                    elif m == "prepend":
                        self.assertListEqual(block.order, ["activation", "layer"])
                        self.assertIsInstance(block.activation, nn.ReLU)
                    elif m == "insert":
                        self.assertListEqual(block.order, ["layer", "activation"])
                        self.assertIsInstance(block.activation, nn.ReLU)
                else:
                    self.assertListEqual(block.order, ["layer", "activation"])
                    self.assertIsInstance(block.activation, nn.ReLU)

    def test_base_block_normalized(self):
        init_with_normalization = [True, False]
        mode = ["replace", "append", "prepend", "insert"]
        wrap_with_layer = [True, False]
        for iwa, m, wrap in product(init_with_normalization, mode, wrap_with_layer):
            with self.subTest(
                init_with_normalization=iwa, mode=m, wrap_with_layer=wrap
            ):
                layer = Layer(nn.Linear, 1, 1)
                normalization = Layer(nn.Identity)
                new_normalization = Layer(nn.ReLU) if wrap else nn.ReLU

                if iwa:
                    block = BaseBlock(layer=layer, normalization=normalization)
                else:
                    block = BaseBlock(layer=layer)

                block.normalized(
                    normalization=new_normalization,
                    mode=m,
                    after="layer" if m == "insert" else None,
                )

                block.build()

                if not iwa:
                    if m == "replace":
                        self.assertListEqual(block.order, ["layer"])
                    elif m == "append":
                        self.assertListEqual(block.order, ["layer", "normalization"])
                        self.assertIsInstance(block.normalization, nn.ReLU)
                    elif m == "prepend":
                        self.assertListEqual(block.order, ["normalization", "layer"])
                        self.assertIsInstance(block.normalization, nn.ReLU)
                    elif m == "insert":
                        self.assertListEqual(block.order, ["layer", "normalization"])
                        self.assertIsInstance(block.normalization, nn.ReLU)
                else:
                    self.assertListEqual(block.order, ["layer", "normalization"])
                    self.assertIsInstance(block.normalization, nn.ReLU)

    def test_base_block_multi(self):
        n = 3
        block = BaseBlock(layer=Layer(nn.Linear, 1, 1))
        block.multi(n=n)

        # ensure all layers are not the same object
        for i in range(1, n):
            self.assertIsNot(block.blocks[i], block.blocks[i - 1])
        for i in range(1, n):
            self.assertIsNot(block.blocks[i].layer, block.blocks[i - 1].layer)

        self.assertListEqual(block.order, ["blocks"])
        self.assertEqual(len(block.blocks), n)

    def test_base_block_shortcut(self):
        block = BaseBlock(layer=Layer(nn.Linear, 1, 1))
        block.shortcut()

        self.assertListEqual(block.order, ["shortcut_start", "layer", "shortcut_end"])
        self.assertIsInstance(block.shortcut_end, Add)

    def test_base_block_multi_shortcut(self):
        n = 3
        block = BaseBlock(layer=Layer(nn.Linear, 1, 1))
        block.multi(n=n)
        block.shortcut()

        self.assertListEqual(block.order, ["shortcut_start", "blocks", "shortcut_end"])
        self.assertIsInstance(block.shortcut_end, Add)
        self.assertEqual(len(block.blocks), n)
        # ensure has correct structure
