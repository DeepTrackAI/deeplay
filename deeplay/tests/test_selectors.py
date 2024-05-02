from deeplay import DeeplayModule, LayerList, LayerActivation, Layer
import unittest
import torch.nn as nn


class TestModule(DeeplayModule):
    def __init__(self):
        self.encoder = LayerList()
        self.decoder = LayerList()

        for i in range(4):
            self.encoder.append(
                LayerActivation(Layer(nn.Conv2d, 3, 3, 1, 1), Layer(nn.ReLU))
            )
            self.decoder.append(
                LayerActivation(Layer(nn.Conv2d, 3, 3, 1, 1), Layer(nn.ReLU))
            )


class TestSelectors(unittest.TestCase):
    def setUp(self) -> None:
        self.module = TestModule()

    def test_selector_str(self):
        selections = self.module["encoder"].list_names()
        self.assertListEqual(selections, [("encoder",)])

    def test_selector_str_bar(self):
        selections = self.module["encoder|decoder"].list_names()
        self.assertListEqual(selections, [("encoder",), ("decoder",)])

    def test_selector_str_comma(self):
        selections = self.module["encoder,decoder"].list_names()
        self.assertListEqual(
            selections,
            [("encoder",), ("decoder",)],
        )

    def test_selector_str_slice(self):
        selections = self.module["encoder", :2].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0"),
                ("encoder", "1"),
            ],
        )

    def test_selector_str_slice_bar(self):
        selections = self.module["encoder|decoder", :2].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0"),
                ("encoder", "1"),
                ("decoder", "0"),
                ("decoder", "1"),
            ],
        )

    def test_selector_ellipsis_first(self):
        selections = self.module[..., "layer"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
                ("encoder", "2", "layer"),
                ("encoder", "3", "layer"),
                ("decoder", "0", "layer"),
                ("decoder", "1", "layer"),
                ("decoder", "2", "layer"),
                ("decoder", "3", "layer"),
            ],
        )

    def test_selector_ellipsis_last(self):
        selections = self.module["encoder", 0, ...].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0"),
                ("encoder", "0", "layer"),
                ("encoder", "0", "activation"),
            ],
        )

    def test_selector_ellipsis_middle(self):
        selections = self.module["encoder", ..., "layer"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
                ("encoder", "2", "layer"),
                ("encoder", "3", "layer"),
            ],
        )

    def test_selector_ellipsis_middle_bar(self):
        selections = self.module["encoder|decoder", ..., "layer"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
                ("encoder", "2", "layer"),
                ("encoder", "3", "layer"),
                ("decoder", "0", "layer"),
                ("decoder", "1", "layer"),
                ("decoder", "2", "layer"),
                ("decoder", "3", "layer"),
            ],
        )

    def test_selector_hash(self):
        selections = self.module["encoder", ..., "layer#0"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
            ],
        )

    def test_selector_hash_slice(self):
        selections = self.module["encoder", ..., "layer#0:2"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
            ],
        )

    def test_selector_has_slice_2(self):
        selections = self.module["encoder", ..., "layer#::2"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "2", "layer"),
            ],
        )

    def test_selector_has_slice_3(self):
        selections = self.module["encoder", ..., "layer#1:3:2"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "1", "layer"),
            ],
        )

    def test_selector_bar_hash(self):
        selections = self.module["encoder|decoder", ..., "layer#0"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("decoder", "0", "layer"),
            ],
        )

    def test_selector_bar_hash_2(self):
        selections = self.module[..., "layer|activation#:2"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "0", "activation"),
            ],
        )

    def test_selector_bar_hash_3(self):
        selections = self.module[..., "layer#:2, activation#:2"].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
                ("encoder", "0", "activation"),
                ("encoder", "1", "activation"),
            ],
        )

    def test_selector_bar_hash_4(self):
        selections = self.module[
            "encoder|decoder", ..., "layer#:2, activation#:2"
        ].list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "1", "layer"),
                ("encoder", "0", "activation"),
                ("encoder", "1", "activation"),
                ("decoder", "0", "layer"),
                ("decoder", "1", "layer"),
                ("decoder", "0", "activation"),
                ("decoder", "1", "activation"),
            ],
        )

    def test_selector_minus_one(self):
        selections = self.module[..., "activation#-1"].list_names()
        self.assertListEqual(
            selections,
            [
                ("decoder", "3", "activation"),
            ],
        )

    def test_selector_isinstance(self):
        selections = self.module["encoder", 0, ...].isinstance(Layer).list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
                ("encoder", "0", "activation"),
            ],
        )

    def test_selector_isinstance_2(self):
        selections = self.module["encoder", 0, ...].isinstance(nn.Conv2d).list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "layer"),
            ],
        )

    def test_selector_isinstance_3(self):
        selections = self.module["encoder", 0, ...].isinstance(nn.ReLU).list_names()
        self.assertListEqual(
            selections,
            [
                ("encoder", "0", "activation"),
            ],
        )

    def test_selector_hasattr(self):
        selections = self.module["encoder", 0, ...].hasattr("append").list_names()
        self.assertListEqual(
            selections,
            [("encoder", "0")],
        )

    def test_selector_hasattr_2(self):
        selections = (
            self.module["encoder", 0, ...].hasattr("_conv_forward").list_names()
        )
        self.assertListEqual(
            selections,
            [("encoder", "0", "layer")],
        )

    def test_selector_append_all(self):
        self.module["encoder", :2, ...].hasattr("append").all.append(
            Layer(nn.Conv2d, 3, 3, 1, 1), name="conv"
        )
        created = self.module.create()
        self.assertIsInstance(created.encoder[0].conv, nn.Conv2d)
        self.assertIsInstance(created.encoder[1].conv, nn.Conv2d)

    def test_selector_append_first(self):
        self.module["encoder", 0, ...].hasattr("append").first.append(
            Layer(nn.Conv2d, 3, 3, 1, 1), name="conv"
        )
        created = self.module.create()
        self.assertIsInstance(created.encoder[0].conv, nn.Conv2d)
        self.assertFalse(hasattr(created.encoder[1], "conv"))
