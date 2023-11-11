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
