import unittest

from deeplay import DeeplayModule, External, Layer
from deeplay.decorators import before_build, after_build, after_init
from unittest.mock import Mock

import torch.nn as nn


class DecoratedModule(DeeplayModule):
    @before_build
    def run_function_before_build(self, func):
        func(self)

    @after_build
    def run_function_after_build(self, func):
        func(self)


class DecoratedExternal(External):
    @before_build
    def run_function_before_build(self, func):
        func(self)

    @after_build
    def run_function_after_build(self, func):
        func(self)


class LayerExpanded(Layer):
    # @before_build # breaks the code
    @after_init
    def set_p(self, v):
        self.p = v


class TestModule1(nn.Module):
    def __init__(self):
        super().__init__()


class TestModule(DeeplayModule):
    def __init__(self):
        self.encoder = LayerExpanded(TestModule1)
        self.decoder = LayerExpanded(TestModule1)


# module = TestModule()
# module.encoder.set_p(2)
# print("before:", module.encoder.p)

# module["encoder"]

# print("after:", module.encoder.p)


class DummyClass: ...


class TestDecorators(unittest.TestCase):
    def test_hooks_do_run(self):
        before_build_mocks = [Mock() for _ in range(3)]
        after_build_mocks = [Mock() for _ in range(3)]

        module = DecoratedModule()

        for mock in before_build_mocks:
            module.run_function_before_build(mock)

        for mock in after_build_mocks:
            module.run_function_after_build(mock)

        module.build()
        for mock in before_build_mocks:
            mock.assert_called_once_with(module)

        for mock in after_build_mocks:
            mock.assert_called_once_with(module)

    def test_hooks_survive_new(self):
        before_build_mocks = [Mock() for _ in range(3)]
        after_build_mocks = [Mock() for _ in range(3)]

        module = DecoratedModule()

        def wrapped(mock):
            return lambda x: mock(x)

        for mock in before_build_mocks:
            module.run_function_before_build(wrapped(mock))

        for mock in after_build_mocks:
            module.run_function_after_build(wrapped(mock))

        new_module = module.new()

        new_module.build()

        for mock in before_build_mocks:
            mock.assert_called_with(new_module)

        for mock in after_build_mocks:
            mock.assert_called_with(new_module)

    def test_hooks_module(self):
        module = DecoratedModule()

        @module.run_function_before_build
        def _before_build(mod):
            self.assertFalse(mod._has_built)

        @module.run_function_after_build
        def _after_build(mod):
            self.assertTrue(mod._has_built)

        module.build()

    def test_hooks_external_do_run(self):
        before_build_mocks = [Mock() for _ in range(3)]
        after_build_mocks = [Mock() for _ in range(3)]

        external = DecoratedExternal(DummyClass)

        for mock in before_build_mocks:
            external.run_function_before_build(mock)

        for mock in after_build_mocks:
            external.run_function_after_build(mock)

        built = external.build()

        for mock in before_build_mocks:
            mock.assert_called_once_with(external)

        for mock in after_build_mocks:
            mock.assert_called_once_with(built)

    def test_hooks_external_survive_new(self):
        before_build_mocks = [Mock() for _ in range(3)]
        after_build_mocks = [Mock() for _ in range(3)]

        external = DecoratedExternal(DummyClass)

        def wrapped(mock):
            return lambda x: mock(x)

        for mock in before_build_mocks:
            external.run_function_before_build(wrapped(mock))

        for mock in after_build_mocks:
            external.run_function_after_build(wrapped(mock))

        new_external = external.new()

        built = new_external.build()

        for mock in before_build_mocks:
            mock.assert_called_with(new_external)

        for mock in after_build_mocks:
            mock.assert_called_with(built)

    def test_hooks_survive_select(self):
        module = TestModule()
        module.encoder.set_p(2)

        self.assertTrue(hasattr(module.encoder, "p"))
        self.assertEqual(module.encoder.p, 2)

        module["encoder"]

        self.assertTrue(hasattr(module.encoder, "p"))
        self.assertEqual(module.encoder.p, 2)

        module = module.new()

        self.assertTrue(hasattr(module.encoder, "p"))
        self.assertEqual(module.encoder.p, 2)
