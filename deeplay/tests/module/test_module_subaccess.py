import unittest

from deeplay import DeeplayModule


class TestClass(DeeplayModule):
    def __init__(self):
        submodule = ChildClass()
        value = submodule.attr

        submodule.configure("attr", value + 1)

        self.submodule = submodule


class ChildClass(DeeplayModule):
    def __init__(self, attr=1):
        super().__init__()

        self.attr = attr


class TestModuleSubaccess(unittest.TestCase):
    def test_subaccess(self):
        test = TestClass()
        self.assertEqual(test.submodule.attr, 2)
