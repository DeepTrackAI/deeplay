import unittest
from deeplay import FromDict, Sequential

import torch
import torch.nn as nn


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
            def forward(self, x, y):
                return x * y

        model = Sequential(FromDict("a", "b"), MultiInputModule())
        model.build()

        inp = {"a": 1, "b": 2}
        out = model(inp)

        self.assertEqual(out, 2)
