import unittest
import torch.nn as nn
from .. import Config, Node


class MockModule(nn.Module):
    def __init__(self, a=None, b=None):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x + self.a + self.b



class TestTemplates(unittest.TestCase):

    def test_node(self):
        # Example usage:

        node = Node("foo")

        config = Config().foo(MockModule, a=1, b=2)

        built_node = node.build(config)

        self.assertEqual(built_node.a, 1)
        self.assertEqual(built_node.b, 2)

        y = built_node(3)
        self.assertEqual(y, 6)
        
        
    def test_sequential_node(self):
        
        template = Node("foo") >> Node("bar")

        config = Config() \
                    ._(a=1, b=2) \
                    .foo(MockModule, b=3) \
                    .bar(MockModule)

        built_node = template.build(config)

        self.assertEqual(built_node["foo"].a, 1)
        self.assertEqual(built_node["foo"].b, 3)
        self.assertEqual(built_node["bar"].a, 1)
        self.assertEqual(built_node["bar"].b, 2)

        y = built_node(5)
        self.assertEqual(y, 12) # 5 + 1 + 3 + 1 + 2