import unittest

from deeplay import DeeplayModule, LayerActivationBlock, Layer

import torch.nn as nn


class TestClass(DeeplayModule):
    def __init__(self):
        submodule = ChildClass()

        submodule.block.layer.configure(out_features=2)

        self.submodule = submodule


class ChildClass(DeeplayModule):
    def __init__(self):
        super().__init__()

        self.block = LayerActivationBlock(
            Layer(nn.Linear, 1, 1),
            Layer(nn.ReLU),
        )


class TestModuleSubaccess(unittest.TestCase):
    def test_subaccess(self):
        test = TestClass()
        test.build()
        self.assertEqual(test.submodule.block.layer.out_features, 2)
