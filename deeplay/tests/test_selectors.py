import deeplay as dl
import unittest


class TestModule(dl.DeeplayModule):
    def __init__(self):
        self.encoder = dl.LayerList()
        self.decoder = dl.LayerList()

        for i in range(4):
            self.encoder.append(
                dl.LayerActivation(dl.Layer(nn.Conv2d, 3, 3, 1, 1), dl.Layer(nn.ReLU))
            )
            self.decoder.append(
                dl.LayerActivation(dl.Layer(nn.Conv2d, 3, 3, 1, 1), dl.Layer(nn.ReLU))
            )


class TestSelectors(unittest.TestCase):
    def setUp(self) -> None:
        self.module = TestModule()

    def test_selector_str(self):
        selections = self.module["encoder"]
        self.assertListEqual(selections, [("encoder",)])

    def test_selector_str_bar(self):
        selections = self.module["encoder|decoder"]
        self.assertListEqual(selections, [("encoder",), ("decoder",)])

    def test_selector_str_comma(self):
        selections = self.module["encoder,decoder"]
        self.assertListEqual(selections, [("encoder", "decoder")])
