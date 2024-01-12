from .. import DeeplayModule, LayerList, LayerActivation, Layer
import unittest
import torch.nn as nn


class TestModule(DeeplayModule):
    def __init__(self):
        self.encoder = LayerList()
        self.decoder = LayerList()

        for i in range(4):
            self.encoder.append(
                LayerActivation(Layer(nn.Conv2d, 3, 3, 1, 1), dl.Layer(nn.ReLU))
            )
            self.decoder.append(
                LayerActivation(Layer(nn.Conv2d, 3, 3, 1, 1), dl.Layer(nn.ReLU))
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
        self.assertListEqual(selections, [("encoder", "decoder")])
