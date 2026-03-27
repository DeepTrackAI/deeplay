import unittest

from deeplay import DeeplayModule, External, Layer
from deeplay.decorators import before_build, after_build, after_init
from unittest.mock import Mock

import torch.nn as nn


class DecoratedModule(DeeplayModule):

    def __init__(self):
        self.before_build_counter = 0
        self.after_build_counter = 0
        super().__init__()

    @before_build
    def run_function_before_build(self, func):
        func(self)

    @after_build
    def run_function_after_build(self, func):
        func(self)


class DecoratedExternal(External):

    def __pre_init__(self, classtype: type, *args, **kwargs):

        self.before_build_counter = 0
        return super().__pre_init__(classtype, *args, **kwargs)

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


class DummyClass:
    def __init__(self):
        self.after_build_counter = 0


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

        module = DecoratedModule()

        def f_before(obj):
            obj.before_build_counter += 1

        def f_after(obj):
            obj.after_build_counter += 1

        for _ in range(3):
            module.run_function_before_build(f_before)

        for _ in range(3):
            module.run_function_after_build(f_after)

        new_module = module.new()
        new_module.build()

        self.assertEqual(new_module.before_build_counter, 3)
        self.assertEqual(new_module.after_build_counter, 3)

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

        external = DecoratedExternal(DummyClass)

        def f_before(obj):
            obj.before_build_counter += 1

        def f_after(obj):
            obj.after_build_counter += 1

        for _ in range(3):
            external.run_function_before_build(f_before)

        for _ in range(3):
            external.run_function_after_build(f_after)

        new_external = external.new()
        built = new_external.build()

        self.assertEqual(new_external.before_build_counter, 3)
        self.assertEqual(built.after_build_counter, 3)

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
