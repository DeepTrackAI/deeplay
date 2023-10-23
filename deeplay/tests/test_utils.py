import unittest
from ..core.utils import safe_call, as_kwargs


class TestUtils(unittest.TestCase):
    def test_safe_call_1(self):
        module = lambda: 1
        parameters = {}
        self.assertEqual(safe_call(module, parameters), 1)

    def test_safe_call_2(self):
        module = lambda: 1
        parameters = {"a": 2}
        self.assertEqual(safe_call(module, parameters), 1)

    def test_safe_call_3(self):
        module = lambda a: a
        parameters = {"a": 2}
        self.assertEqual(safe_call(module, parameters), 2)

    def test_safe_call_4(self):
        module = lambda a, b: a + b
        parameters = {"a": 2, "b": 3}
        self.assertEqual(safe_call(module, parameters), 5)

    def test_safe_call_5(self):
        module = lambda a, b, **kwargs: a + b + kwargs["c"]
        parameters = {"a": 2, "b": 3, "c": 4}
        self.assertEqual(safe_call(module, parameters), 9)


class TestAsKwargs(unittest.TestCase):
    def test_no_parameters(self):
        def func():
            pass

        args = []
        kwargs = {}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {})

    def test_positional_parameters(self):
        def func(a, b, c):
            return a + b + c

        args = [1, 2, 3]
        kwargs = {}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_keyword_parameters(self):
        def func(a=1, b=2, c=3):
            return a + b + c

        args = []
        kwargs = {"a": 10, "b": 20, "c": 30}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 10, "b": 20, "c": 30})

    def test_no_arguments(self):
        def func(a, b, c):
            return a + b + c

        args = []
        kwargs = {}
        with self.assertRaises(ValueError):
            as_kwargs(func, args, kwargs)

    def test_more_positional_arguments(self):
        def func(a, b, c):
            return a + b + c

        args = [1, 2, 3, 4]
        kwargs = {}
        with self.assertRaises(ValueError):
            as_kwargs(func, args, kwargs)

    def test_missing_required_arguments(self):
        def func(a, b, c):
            return a + b + c

        args = []
        kwargs = {"a": 1}
        with self.assertRaises(ValueError):
            as_kwargs(func, args, kwargs)

    def test_combined_positional_and_keyword_parameters(self):
        def func(a, b, c):
            return a + b + c

        args = [1]
        kwargs = {"b": 2, "c": 3}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_combined_positional_and_keyword_parameters_with_default_values_and_kwargs(
        self,
    ):
        def func(a, b, c, **kwargs):
            return a + b + c + sum(kwargs.values())

        args = [1]
        kwargs = {"b": 2, "c": 3, "d": 4, "e": 5}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    def test_positional_only_parameters(self):
        def func(a, /, b, c):
            return a + b + c

        args = [1, 2, 3]
        kwargs = {}
        with self.assertRaises(ValueError):
            as_kwargs(func, args, kwargs)

    def test_no_parameters_with_kwargs(
        self,
    ):
        def func():
            pass

        args = []
        kwargs = {"a": 1, "b": 2, "c": 3}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {})

    def test_positional_parameters_with_default_values(
        self,
    ):
        def func(a=1, b=2, c=3):
            return a + b + c

        args = [10, 20, 30]
        kwargs = {}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 10, "b": 20, "c": 30})

    def test_keyword_parameters_with_default_values(
        self,
    ):
        def func(a=1, b=2, c=3):
            return a + b + c

        args = []
        kwargs = {"a": 10, "b": 20, "c": 30}
        result = as_kwargs(func, args, kwargs)
        self.assertEqual(result, {"a": 10, "b": 20, "c": 30})
