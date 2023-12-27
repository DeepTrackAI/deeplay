import unittest
from deeplay import ToDict
import torch


class TestToDict(unittest.TestCase):
    def test_assignment(self):
        d = ToDict()
        d["a"] = 1
        d["b"] = 2

        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

        d = ToDict(a=1, b=2)
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

        d2 = d.copy()
        d2["a"] = 3
        self.assertEqual(d["a"], 1)
        self.assertEqual(d2["a"], 3)

    # def test_device(self):
    #     d = ToDict()
    #     d["a"] = torch.zeros(1)
    #     d["b"] = torch.zeros(1)

    #     d = d.to("cuda")
    #     self.assertEqual(d["a"].device.type, "cuda")
    #     self.assertEqual(d["b"].device.type, "cuda")
