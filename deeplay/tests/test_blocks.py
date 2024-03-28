import unittest
from deeplay import Block, SequentialBlock, Layer

import torch
import torch.nn as nn


class TestBlock(unittest.TestCase):

    def test_block(self):
        block = Block(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)

    def test_configure_block(self):
        block = Block(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.configure(b=Layer(nn.Conv2d, 1, 1, 1))

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Conv2d)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Conv2d)


class TestSequentalBlock(unittest.TestCase):

    def test_sequential_block(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertListEqual(block.order, ["a", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertListEqual(created.order, ["a", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertListEqual(built.order, ["a", "b"])

    def test_configure_sequential_block_order(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.configure(order=["b", "a"])

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertListEqual(block.order, ["b", "a"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertListEqual(created.order, ["b", "a"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertListEqual(built.order, ["b", "a"])

    def test_configure_sequence_block(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.configure(b=Layer(nn.Conv2d, 1, 1, 1))

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertListEqual(block.order, ["a", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Conv2d)
        self.assertListEqual(created.order, ["a", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Conv2d)
        self.assertListEqual(built.order, ["a", "b"])

    def test_configure_add_layer(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.configure(c=Layer(nn.Conv2d, 1, 1, 1), order=["a", "b", "c"])

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.c, Layer)
        self.assertListEqual(block.order, ["a", "b", "c"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.c, nn.Conv2d)
        self.assertListEqual(created.order, ["a", "b", "c"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.c, nn.Conv2d)
        self.assertListEqual(built.order, ["a", "b", "c"])

    def test_append(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.append(Layer(nn.Conv2d, 1, 1, 1), name="c")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.c, Layer)
        self.assertListEqual(block.order, ["a", "b", "c"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.c, nn.Conv2d)
        self.assertListEqual(created.order, ["a", "b", "c"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.c, nn.Conv2d)
        self.assertListEqual(built.order, ["a", "b", "c"])

    def test_prepend(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.prepend(Layer(nn.Conv2d, 1, 1, 1), name="c")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.c, Layer)
        self.assertListEqual(block.order, ["c", "a", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.c, nn.Conv2d)
        self.assertListEqual(created.order, ["c", "a", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.c, nn.Conv2d)
        self.assertListEqual(built.order, ["c", "a", "b"])

    def test_insert(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.insert(Layer(nn.Conv2d, 1, 1, 1), after="a", name="c")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.c, Layer)
        self.assertListEqual(block.order, ["a", "c", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.c, nn.Conv2d)
        self.assertListEqual(created.order, ["a", "c", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.c, nn.Conv2d)
        self.assertListEqual(built.order, ["a", "c", "b"])

    def test_remove(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.remove("a")

        self.assertNotIn("a", block.order)

        created = block.create()

        self.assertNotIn("a", created.order)

        built = block.build()

        self.assertNotIn("a", built.order)

    def test_remove_missing_not_ok(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        with self.assertRaises(ValueError):
            block.remove("c")

    def test_remove_missing_ok(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.remove("c", allow_missing=True)

        self.assertNotIn("c", block.order)

    def test_append_dropout(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.append_dropout(0.5)

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["a", "b", "dropout"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)
        self.assertListEqual(created.order, ["a", "b", "dropout"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)
        self.assertListEqual(built.order, ["a", "b", "dropout"])

    def test_prepend_dropout(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.prepend_dropout(0.5)

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["dropout", "a", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)
        self.assertListEqual(created.order, ["dropout", "a", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)
        self.assertListEqual(built.order, ["dropout", "a", "b"])

    def test_insert_dropout(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.insert_dropout(0.5, after="a")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["a", "dropout", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)
        self.assertListEqual(created.order, ["a", "dropout", "b"])

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)
        self.assertListEqual(built.order, ["a", "dropout", "b"])

    def test_insert_dropout_missing(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        with self.assertRaises(ValueError):
            block.insert_dropout(0.5, after="c")

    def test_set_dropout_append_if_missing(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.set_dropout(0.5, on_missing="append")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["a", "b", "dropout"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)

    def test_set_dropout_prepend_if_missing(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.set_dropout(0.5, on_missing="prepend")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["dropout", "a", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)

    def test_set_dropout_insert_if_missing(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.set_dropout(0.5, on_missing="insert", after="a")

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertIsInstance(block.dropout, Layer)
        self.assertListEqual(block.order, ["a", "dropout", "b"])

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertIsInstance(created.dropout, nn.Dropout)
        self.assertEqual(created.dropout.p, 0.5)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertIsInstance(built.dropout, nn.Dropout)
        self.assertEqual(built.dropout.p, 0.5)

    def test_remove_dropout(self):
        block = SequentialBlock(
            a=Layer(nn.Linear, 1, 2),
            b=Layer(nn.Linear, 2, 1),
            dropout=Layer(nn.Dropout, 0.5),
        )
        block.remove_dropout()

        self.assertIsInstance(block.a, Layer)
        self.assertIsInstance(block.b, Layer)
        self.assertNotIn("dropout", block.order)

        created = block.create()

        self.assertIsInstance(created.a, nn.Linear)
        self.assertIsInstance(created.b, nn.Linear)
        self.assertNotIn("dropout", created.order)

        built = block.build()

        self.assertIsInstance(built.a, nn.Linear)
        self.assertIsInstance(built.b, nn.Linear)
        self.assertNotIn("dropout", built.order)

    def test_remove_dropout_missing_not_ok(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        with self.assertRaises(ValueError):
            block.remove_dropout()

    def test_remove_dropout_missing_ok(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.remove_dropout(allow_missing=True)

        created = block.create()
        built = block.build()

        self.assertNotIn("dropout", created.order)
        self.assertNotIn("dropout", built.order)

    def test_default_name(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.append(Layer(nn.Conv2d, 1, 1, 1))

        self.assertListEqual(block.order, ["a", "b", "conv2d"])

    def test_default_name_not_layer(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))
        block.append(SequentialBlock())

        self.assertListEqual(block.order, ["a", "b", "sequentialblock"])

    def test_default_name_conflict(self):
        block = SequentialBlock(a=Layer(nn.Linear, 1, 2), b=Layer(nn.Linear, 2, 1))

        with self.assertRaises(ValueError):
            block.append(Layer(nn.Conv2d, 1, 1, 1), name="b")
