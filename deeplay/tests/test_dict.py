import unittest
from deeplay import (
    FromDict,
    Sequential,
    Layer,
    LayerSkip,
    AddDict,
    Parallel,
    DeeplayModule,
)

import torch
import torch.nn as nn
from torch_geometric.data import Data


class TestComponentDict(unittest.TestCase):
    def test_FromDict(self):
        module = FromDict("a", "b")

        inp = {"a": 1, "b": 2}
        out = module(inp)

        self.assertEqual(out, (1, 2))

        module = FromDict("a")

        inp = {"a": 1, "b": 2}
        out = module(inp)

        self.assertEqual(out, 1)

        model = Sequential(FromDict("b"), nn.Linear(2, 10))
        model.build()

        inp = {"a": 1, "b": torch.ones(1, 2)}
        out = model(inp)

        class MultiInputModule(nn.Module):
            def forward(self, x):
                a, b = x
                return a * b

        model = Sequential(FromDict("a", "b"), MultiInputModule())
        model.build()

        inp = {"a": 1, "b": 2}
        out = model(inp)

        self.assertEqual(out, 2)

    def test_add_dict(self):
        inp = {}
        inp["x"] = 2
        inp["y"] = 3

        class MulBlock(DeeplayModule):
            def forward(self, x, y):
                return x * y

        layer = Parallel(
            x=MulBlock().set_input_map("x", "y"), y=MulBlock().set_input_map("y", "x")
        )
        block = LayerSkip(layer=layer, skip=AddDict("x", "y")).create()

        out = block(inp)

        self.assertEqual(out["x"], 8)
        self.assertEqual(out["y"], 9)

        inp = Data(x=torch.Tensor([2]), y=torch.Tensor([3]))

        out = block(inp)

        self.assertEqual(out.x, 8)
        self.assertEqual(out.y, 9)

    def test_add_with_base_dict(self):
        inp = Data(x=torch.Tensor([2]), y=torch.Tensor([3]))

        layer = Parallel(
            x=Layer(nn.Linear, 1, 1).set_input_map("x"),
            y=Layer(nn.Linear, 1, 10).set_input_map("y"),
        )
        block = LayerSkip(layer=layer, skip=AddDict("x")).create()

        out = block(inp)

        self.assertEqual(inp.x, 2)
        self.assertEqual(len(out.x), 1)

        # Checks that the base dict is correctly passed
        self.assertEqual(inp.y, 3)
        self.assertEqual(len(out.y), 10)
