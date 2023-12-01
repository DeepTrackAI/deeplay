import unittest

import deeplay as dl
import torch.nn as nn


class Wrapper(dl.DeeplayModule):
    def __init__(self, module):
        super().__init__()
        self.module = module


class VariadicClass:
    def __init__(self, *args, **kwargs):
        self._args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


class KWVariadicClass:
    def __init__(self, arg1, kwarg=2, **kwargs):
        self.arg1 = arg1
        self.kwarg = kwarg
        for key, value in kwargs.items():
            setattr(self, key, value)


class GeneralVariadicClass:
    def __init__(
        self, pos_only, /, standard, *args, kw_only, kwonly_with_default=60, **kwargs
    ):
        self.pos_only = pos_only
        self.standard = standard
        self.kw_only = kw_only
        self.kwonly_with_default = kwonly_with_default
        self._args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestExternal(unittest.TestCase):
    def test_external(self):
        external = dl.External(nn.Identity)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, nn.Identity)
        self.assertIsInstance(built, nn.Identity)
        self.assertIsNot(built, created)

    def test_external_arg(self):
        external = dl.External(nn.Linear, 10, 20)
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
        external = dl.External(nn.Sigmoid)
        wrapped = Wrapper(external)
        built = wrapped.build()
        created = wrapped.create()
        self.assertIsInstance(created, Wrapper)
        self.assertIsInstance(built, Wrapper)
        self.assertIsNot(built, created)

        self.assertIsInstance(created.module, nn.Sigmoid)
        self.assertIsInstance(built.module, nn.Sigmoid)
        self.assertIsNot(built.module, created.module)

    def test_variadic(self):
        external = dl.External(VariadicClass, 10, 20, arg=30)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, VariadicClass)
        self.assertIsInstance(built, VariadicClass)
        self.assertIsNot(built, created)

        self.assertEqual(built._args, (10, 20))
        self.assertEqual(built.arg, 30)

        self.assertEqual(created._args, (10, 20))
        self.assertEqual(created.arg, 30)

    def test_kwvariadic_1(self):
        external = dl.External(KWVariadicClass, 5, kwarg=30, arg2=40)
        external.configure(arg1=10)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, KWVariadicClass)
        self.assertIsInstance(built, KWVariadicClass)
        self.assertIsNot(built, created)

        self.assertEqual(built.arg1, 10)
        self.assertEqual(built.kwarg, 30)
        self.assertEqual(built.arg2, 40)

        self.assertEqual(created.arg1, 10)
        self.assertEqual(created.kwarg, 30)
        self.assertEqual(created.arg2, 40)

    def test_kwvariadic_2(self):
        external = dl.External(KWVariadicClass, arg1=10, kwarg=30, arg2=40)
        built = external.build()
        created = external.create()
        self.assertIsInstance(created, KWVariadicClass)
        self.assertIsInstance(built, KWVariadicClass)
        self.assertIsNot(built, created)

        self.assertEqual(built.arg1, 10)
        self.assertEqual(built.kwarg, 30)
        self.assertEqual(built.arg2, 40)

        self.assertEqual(created.arg1, 10)
        self.assertEqual(created.kwarg, 30)
        self.assertEqual(created.arg2, 40)

    def test_general_variadic(self):
        with self.assertRaises(TypeError):
            external = dl.External(
                GeneralVariadicClass, 10, 20, 25, kw_only=30, kwonly_with_default=50
            )
