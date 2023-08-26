import unittest
import torch.nn as nn
from .. import Config, Layer


class MockModule(nn.Module):
    def __init__(self, a=None, b=None):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x + self.a + self.b



class TestTemplates(unittest.TestCase):

    def test_Layer(self):
        # Example usage:

        layer = Layer("foo")

        config = Config().foo(MockModule, a=1, b=2)

        built_Layer = layer.build(config)

        self.assertEqual(built_Layer.a, 1)
        self.assertEqual(built_Layer.b, 2)

        y = built_Layer(3)
        self.assertEqual(y, 6)
        
        
    def test_sequential_Layer(self):
        
        template = Layer("foo") >> Layer("bar")

        config = Config() \
                    ._(a=1, b=2) \
                    .foo(MockModule, b=3) \
                    .bar(MockModule)

        built_Layer = template.build(config)

        self.assertEqual(built_Layer["foo"].a, 1)
        self.assertEqual(built_Layer["foo"].b, 3)
        self.assertEqual(built_Layer["bar"].a, 1)
        self.assertEqual(built_Layer["bar"].b, 2)

        y = built_Layer(5)
        self.assertEqual(y, 12) # 5 + 1 + 3 + 1 + 2