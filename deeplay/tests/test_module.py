import unittest
import torch.nn as nn
import deeplay as dl


class DummyClass:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class Module(dl.DeeplayModule):
    def __init__(self, a=0, b=0, c="0", **kwargs):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.x = dl.External(DummyClass, a, b, c)
        self.y = dl.Layer(nn.Linear, a, b)


class Module2(dl.DeeplayModule):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo
        self.bar = Module(1, 2, "C")


class TestDeeplayModule(unittest.TestCase):
    def test_configure_1(self):
        module = Module()
        module.configure(a=1)
        module.build()
        self.assertEqual(module.a, 1)
        self.assertEqual(module.x.a, 1)

    def test_configure_2(self):
        module = Module()
        module.configure("a", 1)
        module.configure("b", 2)
        module.build()
        self.assertEqual(module.a, 1)
        self.assertEqual(module.b, 2)
        self.assertEqual(module.x.a, 1)
        self.assertEqual(module.x.b, 2)

    def test_configure_3(self):
        module = Module()
        module.configure("a", 1)
        module.configure("b", 2)
        module.configure("c", "C")
        module.build()
        self.assertEqual(module.a, 1)
        self.assertEqual(module.b, 2)
        self.assertEqual(module.c, "C")
        self.assertEqual(module.x.a, 1)
        self.assertEqual(module.x.b, 2)
        self.assertEqual(module.x.c, "C")

    def test_configure_4(self):
        module = Module(b=2, c="C")
        module.configure("a", 1)
        module.configure("a", 3)
        module.build()
        self.assertEqual(module.a, 3)
        self.assertEqual(module.b, 2)
        self.assertEqual(module.c, "C")
        self.assertEqual(module.x.a, 3)
        self.assertEqual(module.x.b, 2)
        self.assertEqual(module.x.c, "C")

    def test_configure_5(self):
        module = Module()

        with self.assertRaises(ValueError):
            module.configure("d", 1)

    def test_configure_7(self):
        module = Module()

        with self.assertRaises(ValueError):
            module.configure(a=1, b=2, d="C")

    def test_configure_8(self):
        module = Module2(foo=Module())
        module.configure("foo", a=1, b=2, c="C")
        module.build()

        self.assertEqual(module.foo.a, 1)
        self.assertEqual(module.foo.b, 2)
        self.assertEqual(module.foo.c, "C")

    def test_configure_9(self):
        module = Module2(foo=Module())
        module.bar.configure(a=1, b=2, c="C")
        module.build()

        self.assertEqual(module.bar.a, 1)
        self.assertEqual(module.bar.b, 2)
        self.assertEqual(module.bar.c, "C")

    def test_configure_10(self):
        module = Module2(Module())
        module.build()

        module.configure("bar", a=3, b=4, c="D")
        self.assertEqual(module.bar.a, 3)
        self.assertEqual(module.bar.b, 4)
        self.assertEqual(module.bar.c, "D")

    def test_init_2(self):
        module = Module(a=1, b=2, c="C")
        module.build()

        self.assertEqual(module.a, 1)
        self.assertEqual(module.b, 2)
        self.assertEqual(module.c, "C")

    def test_init_3(self):
        module = Module(1, 2, "C")
        module.build()

        self.assertEqual(module.a, 1)
        self.assertEqual(module.b, 2)
        self.assertEqual(module.c, "C")

    def test_init_6(self):
        module = Module2(foo=Module(a=1, b=2, c="C"))
        module.build()
        self.assertEqual(module.foo.a, 1)
        self.assertEqual(module.foo.b, 2)
        self.assertEqual(module.foo.c, "C")


import torch
import torch.nn as nn

nn.Linear


class ModelWithLayer(dl.DeeplayModule):
    def __init__(self, in_features=10, out_features=20):
        super().__init__()
        self.layer_1 = dl.Layer(nn.Linear, in_features, out_features)
        self.layer_2 = dl.Layer(nn.Sigmoid)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


class TestLayer(unittest.TestCase):
    def test_create(self):
        layer = dl.Layer(nn.Identity).build()
        self.assertIsInstance(layer, nn.Identity)

    def test_create_with_args(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10).build()

        self.assertIsInstance(layer, nn.BatchNorm1d)
        self.assertEqual(layer.num_features, 10)

    def test_configure(self):
        layer = dl.Layer(nn.Identity)
        layer.configure(nn.BatchNorm1d, num_features=10)
        layer = layer.create()

        self.assertIsInstance(layer, nn.BatchNorm1d)
        self.assertEqual(layer.num_features, 10)

    def test_configure_2(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10)
        layer.configure(num_features=20)
        layer = layer.create()

        self.assertIsInstance(layer, nn.BatchNorm1d)
        self.assertEqual(layer.num_features, 20)

    def test_configure_3(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10)
        with self.assertRaises(ValueError):
            layer.configure(missdefined=10)

    def test_configure_4(self):
        layer = dl.Layer(nn.Identity)
        with self.assertRaises(ValueError):
            layer.configure(nn.Identity, num_features=20)

    def test_forward(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10)
        layer = layer.create()
        x = torch.randn(10, 10)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_in_module(self):
        model = ModelWithLayer()
        model.configure("layer_1", in_features=10, out_features=20)
        model = model.build()
        self.assertEqual(model.layer_1.in_features, 10)
        self.assertEqual(model.layer_1.out_features, 20)
        x = torch.randn(10, 10)
        y = model(x)
        self.assertEqual(y.shape, (10, 20))

    def test_if_crosstalk(self):
        model_1 = ModelWithLayer()
        model_2 = ModelWithLayer()
        model_1.configure("layer_1", in_features=10, out_features=20)
        model_2.configure("layer_1", in_features=40, out_features=70)

        model_1 = model_1.build()
        model_2 = model_2.build()

        self.assertEqual(model_1.layer_1.in_features, 10)
        self.assertEqual(model_1.layer_1.out_features, 20)
        self.assertEqual(model_2.layer_1.in_features, 40)
        self.assertEqual(model_2.layer_1.out_features, 70)

        # print(model_2)

    def test_config_is_sticky(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10)
        layer.configure(num_features=20)

        model = Module2(foo=layer)
        model.build()

        self.assertEqual(model.foo.num_features, 20)


import unittest
from unittest.mock import Mock, patch
from deeplay import (
    DeeplayModule,
)  # Import your actual module here

import unittest
from deeplay import DeeplayModule, ExtendedConstructorMeta
import torch.nn as nn


# A simple subclass for testing
class TestModule(DeeplayModule):
    def __init__(self, param1=None, param2=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2


# Unit tests for DeeplayModule
class TestDeeplayModule(unittest.TestCase):
    def test_initialization(self):
        # Testing basic initialization and attribute setting
        module = TestModule(param1=10, param2="test")
        self.assertEqual(module.param1, 10)
        self.assertEqual(module.param2, "test")

    def test_configure(self):
        # Testing the configure method
        module = TestModule()
        module.configure("param1", 20)
        module.configure(param2="configured")
        self.assertEqual(module.param1, 20)
        self.assertEqual(module.param2, "configured")

    def test_build(self):
        # Testing the build method
        module = TestModule()
        module.configure(param1=30)
        built_module = module.build()
        self.assertEqual(built_module.param1, 30)
        self.assertTrue(built_module._has_built)

    def test_create(self):
        # Testing the create method
        module = TestModule(param1=40)
        new_module = module.create()
        self.assertIsInstance(new_module, TestModule)
        self.assertEqual(new_module.param1, 40)
        self.assertNotEqual(new_module, module)

    def test_get_user_configuration(self):
        # Testing retrieval of user configuration
        module = TestModule(param1=50)
        module.configure(param1=60)
        config = module.get_user_configuration()
        self.assertEqual(config[("param1",)], 60)

    def test_invalid_configure(self):
        # Testing configure method with invalid attribute
        module = TestModule()
        with self.assertRaises(ValueError):
            module.configure("invalid_param", 100)

    # Additional tests for other methods and edge cases can be added here


# class TestModule(DeeplayModule):
#     def __init__(self, param1=None, param2=None, child_module=None):
#         super().__init__()
#         self.param1 = param1
#         self.param2 = param2
#         self.child_module = child_module if child_module else TestModule()


# class TestDeeplayModuleConfigure(unittest.TestCase):
#     def test_configure_with_positional_args(self):
#         # Test configuring with positional arguments
#         module = TestModule()
#         module.configure("param1", 10)
#         self.assertEqual(module.param1, 10)

#     def test_configure_with_keyword_args(self):
#         # Test configuring with keyword arguments
#         module = TestModule()
#         module.configure(param2="value")
#         self.assertEqual(module.param2, "value")

#     def test_configure_with_multiple_kwargs(self):
#         # Test configuring multiple attributes using keyword arguments
#         module = TestModule()
#         module.configure(param1=20, param2="another_value")
#         self.assertEqual(module.param1, 20)
#         self.assertEqual(module.param2, "another_value")

#     def test_configure_child_module(self):
#         # Test configuring an attribute which is itself a DeeplayModule
#         parent_module = TestModule()
#         child_module = TestModule()
#         parent_module.configure("child_module", param1=30, param2="child_value")
#         self.assertEqual(parent_module.child_module.param1, 30)
#         self.assertEqual(parent_module.child_module.param2, "child_value")

#     def test_configure_with_invalid_attribute(self):
#         # Test configuring with an invalid attribute
#         module = TestModule()
#         with self.assertRaises(ValueError):
#             module.configure("non_existent_param", 40)

#     def test_configure_with_invalid_pattern(self):
#         # Test configuring with a valid attribute but invalid pattern
#         module = TestModule()
#         with self.assertRaises(ValueError):
#             module.configure("param1", 50, extra_arg="unexpected")


if __name__ == "__main__":
    unittest.main()
