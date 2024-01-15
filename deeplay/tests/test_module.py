import unittest

import torch
import torch.nn as nn
import deeplay as dl

from deeplay import (
    DeeplayModule,
    Sequential,
)  # Import your actual module here


# A simple subclass for testing
class TestModule(DeeplayModule):
    def __init__(self, param1=None, param2=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2


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

        with self.assertRaises(RuntimeError):
            module.configure("foo", a=1, b=2, c="C")

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

    def test_create_independency_args(self):
        child = Module(a=1, b=2, c="C")
        parent = Module2(foo=child)
        created = parent.create()

        self.assertEqual(created.foo.a, 1)
        self.assertEqual(created.foo.b, 2)
        self.assertEqual(created.foo.c, "C")

        parent.foo.configure(a=3)

        created_2 = parent.create()

        self.assertEqual(created_2.foo.a, 3)
        self.assertEqual(created_2.foo.b, 2)
        self.assertEqual(created_2.foo.c, "C")
        self.assertEqual(created.foo.a, 1)

        self.assertIsNot(created.foo, created_2.foo)

    def test_replace_1(self):
        parent = Module2(foo=Module(a=1, b=2, c="C"))
        parent.replace("foo", Module(a=3, b=4, c="D"))
        parent.replace("bar", Module(a=5, b=6, c="E"))
        parent.build()

        self.assertEqual(parent.foo.a, 3)
        self.assertEqual(parent.foo.b, 4)
        self.assertEqual(parent.foo.c, "D")

        self.assertEqual(parent.bar.a, 5)
        self.assertEqual(parent.bar.b, 6)
        self.assertEqual(parent.bar.c, "E")

    def test_replace_2(self):
        parent = Module2(foo=Module(a=1, b=2, c="C"))

        new_child = Module2(foo=Module(a=3, b=4, c="D"))
        new_child.foo.configure(a=5, b=6, c="E")
        new_child.replace("bar", Module(a=7, b=8, c="F"))

        parent.replace("foo", new_child)
        parent.build()

        self.assertEqual(parent.foo.foo.a, 5)
        self.assertEqual(parent.foo.foo.b, 6)
        self.assertEqual(parent.foo.foo.c, "E")

        self.assertEqual(parent.foo.bar.a, 7)
        self.assertEqual(parent.foo.bar.b, 8)
        self.assertEqual(parent.foo.bar.c, "F")


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
        layer = dl.Layer(nn.Conv2d)
        with self.assertRaises(ValueError):
            layer.configure(num_features=20)

    def test_forward(self):
        layer = dl.Layer(nn.BatchNorm1d, num_features=10)
        layer = layer.create()
        x = torch.randn(10, 10)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_with_input_dict(self):
        layer = dl.Layer(nn.Linear, 1, 20)
        layer.set_input_map("x")
        layer.set_output_map("x")

        layer = layer.build()
        inp = {"x": torch.randn(10, 1)}
        out = layer(inp)
        self.assertEqual(out["x"].shape, (10, 20))

    def test_forward_with_input_dict_and_numeric_output(self):
        layer = dl.Layer(nn.Linear, 1, 20)
        layer.set_input_map("x")
        layer.set_output_map()

        layer = layer.build()
        inp = {"x": torch.randn(10, 1)}
        out = layer(inp)
        self.assertEqual(out.shape, (10, 20))

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

    def test_configure_in_init(self):
        class TestClass(dl.DeeplayModule):
            def __init__(self, model=None):
                super().__init__()

                model = dl.MultiLayerPerceptron(None, [64], 10)
                model.output.normalization.configure(nn.BatchNorm1d)
                self.model = model

        testclass = TestClass()

        self.assertEqual(testclass.model.output.normalization.classtype, nn.BatchNorm1d)

        testclass.build()

        self.assertIsInstance(testclass.model.output.normalization, nn.BatchNorm1d)


class TestSequential(unittest.TestCase):
    def test_forward_with_input_dict(self):
        class AggregationRelu(nn.Module):
            def forward(self, x, A):
                return nn.functional.relu(A @ x)

        model = Sequential(
            dl.Layer(nn.Linear, 1, 20),
            dl.Layer(AggregationRelu),
            dl.Layer(nn.Linear, 20, 1),
        )

        model[0].set_input_map("x")
        model[0].set_output_map("x")

        model[1].set_input_map("x", "A")
        model[1].set_output_map("x")

        model[2].set_input_map("x")
        model[2].set_output_map("x")

        model.build()

        inp = {"x": torch.randn(10, 1), "A": torch.randn(10, 10)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))

    def test_set_mapping_1(self):
        model = Sequential(
            dl.Layer(nn.Linear, 1, 20),
            dl.Layer(nn.ReLU),
            dl.Layer(nn.Linear, 20, 1),
        )
        model.set_input_map("x")
        model.set_output_map("x")

        model.build()
        inp = {"x": torch.randn(10, 1)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))

    def test_set_mapping_2(self):
        model = Sequential(
            dl.Layer(nn.Linear, 1, 20),
            dl.Layer(nn.ReLU),
            dl.Layer(nn.Linear, 20, 1),
        )
        model.set_input_map("x")
        model.set_output_map("x")

        model[1].set_output_map("x", x1=0, x2=0)

        model.build()
        inp = {"x": torch.randn(10, 1)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))
        self.assertEqual(torch.all(out["x1"] == out["x2"]), True)


if __name__ == "__main__":
    unittest.main()
