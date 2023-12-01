import unittest

import deeplay as dl
import torch
import torch.nn as nn

from unittest.mock import Mock


class Wrapper(dl.DeeplayModule):
    def __init__(self, module):
        super().__init__()
        self.module = module


class Container(dl.DeeplayModule):
    def __init__(self):
        super().__init__()
        self.module = dl.Layer(nn.Identity)


class TestLayer(unittest.TestCase):
    def test_external(self):
        external = dl.Layer(nn.Identity)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, nn.Identity)
        self.assertIsInstance(built, nn.Identity)
        self.assertIsNot(built, created)

    def test_external_arg(self):
        external = dl.Layer(nn.Linear, 10, 20)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, nn.Linear)
        self.assertIsInstance(built, nn.Linear)
        self.assertIsNot(built, created)

        self.assertEqual(built.in_features, 10)
        self.assertEqual(built.out_features, 20)

        self.assertEqual(created.in_features, 10)
        self.assertEqual(created.out_features, 20)

    def test_wrapped(self):
        external = dl.Layer(nn.Sigmoid)
        wrapped = Wrapper(external)
        created = wrapped.create()
        built = wrapped.build()

        self.assertIsInstance(created, Wrapper)
        self.assertIsInstance(built, Wrapper)
        self.assertIsNot(built, created)

        self.assertIsInstance(created.module, nn.Sigmoid)
        self.assertIsInstance(built.module, nn.Sigmoid)
        self.assertIsNot(built.module, created.module)

    def test_wrapped_2(self):
        external = dl.Layer(nn.Tanh)
        external.configure(nn.Sigmoid)
        wrapped = Wrapper(external)

        created = wrapped.create()
        built = wrapped.build()

        self.assertIsInstance(created, Wrapper)
        self.assertIsInstance(built, Wrapper)
        self.assertIsNot(built, created)

        self.assertIsInstance(created.module, nn.Sigmoid)
        self.assertIsInstance(built.module, nn.Sigmoid)
        self.assertIsNot(built.module, created.module)

    def test_wrapped_3(self):
        container = Container()
        container.module.configure(nn.Sigmoid)
        wrapped = Wrapper(container)

        created = wrapped.create()
        built = wrapped.build()

        self.assertIsInstance(created, Wrapper)
        self.assertIsInstance(built, Wrapper)
        self.assertIsNot(built, created)

        self.assertIsInstance(created.module.module, nn.Sigmoid)
        self.assertIsInstance(built.module.module, nn.Sigmoid)
        self.assertIsNot(built.module, created.module)

    def test_wrapped_4(self):
        container = Container()
        container.build()
        wrapped = Wrapper(container)

        self.assertIsInstance(wrapped.module.module, nn.Identity)

    def test_set_extras(self):
        class SimpleModule(nn.Module):
            def forward(self, x, extra1, extra2):
                extra1()
                extra2()
                return x

        external = dl.Layer(SimpleModule)
        external.set_extras("extra1", extra2="other_name")
        built = external.build()

        built

        all_extras = {
            "extra1": Mock(),
            "other_name": Mock(),
            "extra2": Mock(),
            "unrelated": Mock(),
        }

        built(torch.randn(10, 10), extras=all_extras)

        all_extras["extra1"].assert_called_once()
        all_extras["other_name"].assert_called_once()
        all_extras["extra2"].assert_not_called()
        all_extras["unrelated"].assert_not_called()
