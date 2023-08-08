import unittest
from ..utils import safe_call

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
