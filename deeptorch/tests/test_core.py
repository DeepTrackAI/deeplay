import unittest
from ..core import DeepTorchModule
from ..templates import Layer
from ..config import Config


class TestCore(unittest.TestCase):
    class MockDTModule(DeepTorchModule):
        defaults = {
            "bias": 0,
            "block": Layer("layer"),
            "block.layer": lambda scale: lambda x: x * scale,
            "block.layer.scale": 1,
        }

        def __init__(self, bias=0, block=None, **kwargs):
            super().__init__(bias=bias, block=block, **kwargs)
            self.bias = self.attr("bias")
            self.layer = self.create("block")

        def forward(self, x):
            return self.layer(x) + self.bias

    def test_no_input(self):
        module = self.MockDTModule()
        self.assertEqual(module.bias, 0)
        self.assertEqual(module(1), 1)
        self.assertEqual(module(2), 2)
        self.assertEqual(module(3), 3)

    def test_set_bias_direct(self):
        module = self.MockDTModule(bias=1)
        self.assertEqual(module.bias, 1)
        self.assertEqual(module(1), 2)
        self.assertEqual(module(2), 3)
        self.assertEqual(module(3), 4)

    def test_set_scale(self):
        module = self.MockDTModule(block=Config().layer.scale(2))
        self.assertEqual(module.bias, 0)
        self.assertEqual(module(1), 2)
        self.assertEqual(module(2), 4)
        self.assertEqual(module(3), 6)

    def test_set_scale_inhereted(self):
        module = self.MockDTModule(block=Config()._(scale=2))
        self.assertEqual(module.bias, 0)
        self.assertEqual(module(1), 2)
        self.assertEqual(module(2), 4)
        self.assertEqual(module(3), 6)

    def test_set_scale_inhereted_2(self):
        module = self.MockDTModule(block=Config()._.scale(2))
        self.assertEqual(module.bias, 0)
        self.assertEqual(module(1), 2)
        self.assertEqual(module(2), 4)
        self.assertEqual(module(3), 6)

    def test_set_scale_and_bias(self):
        module = self.MockDTModule(bias=1, block=Config().__(scale=2))
        self.assertEqual(module.bias, 1)
        self.assertEqual(module(1), 3)
        self.assertEqual(module(2), 5)
        self.assertEqual(module(3), 7)

    def test_from_config(self):
        config = Config()(bias=1).__(scale=2)
        module = self.MockDTModule.from_config(config)
        self.assertEqual(module.bias, 1)
        self.assertEqual(module(1), 3)
        self.assertEqual(module(2), 5)
        self.assertEqual(module(3), 7)

    def test_nested_structure(self):
        module = self.MockDTModule(
            bias=1, block=Config().layer(self.MockDTModule).layer.block.layer(scale=3)
        )

        self.assertEqual(module.bias, 1)
        self.assertEqual(module(1), 4)
        self.assertEqual(module(2), 7)
        self.assertEqual(module(3), 10)

    def test_create_all(self):
        class MockDTModule2(DeepTorchModule):
            defaults = Config()

            def __init__(self, foo):
                self.foo = self.create_all("foo")

        module = MockDTModule2.from_config(Config().foo[0](1))
        self.assertEqual(len(module.foo), 1)
        self.assertEqual(module.foo[0], 1)

        module = MockDTModule2.from_config(Config().foo[0](1).foo[1](2))
        self.assertEqual(len(module.foo), 2)
        self.assertEqual(module.foo[0], 1)
        self.assertEqual(module.foo[1], 2)

        module = MockDTModule2.from_config(Config().foo(None).foo[2](3))
        self.assertEqual(len(module.foo), 3)
        self.assertEqual(module.foo[0], None)
        self.assertEqual(module.foo[1], None)
        self.assertEqual(module.foo[2], 3)

        module = MockDTModule2.from_config(Config().foo[0:4](1).foo[0:4:2](2))
        self.assertEqual(len(module.foo), 4)
        self.assertEqual(module.foo[0], 2)
        self.assertEqual(module.foo[1], 1)
        self.assertEqual(module.foo[2], 2)
        self.assertEqual(module.foo[3], 1)
