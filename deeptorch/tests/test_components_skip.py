import unittest
import torch
import torch.nn as nn
from .. import Config, Layer, Skip, Concatenate, OutputOf


class TestComponentsSkip(unittest.TestCase):

    def test_skip(self):

        template = Layer("layer1") >> Layer("layer2") >> Layer("socket")

        config = (
            Config()
            .layer1(nn.Linear, in_features=10, out_features=10, uid="layer1")
            .layer2(nn.Linear, in_features=10, out_features=10)
            .socket(Skip, func=lambda a, b: torch.cat((a, b), dim=1))
            .socket.inputs[1](OutputOf("layer1"))
        )

        module = template.build(config)
        x = torch.ones(1, 10)
        y = module(x)
        self.assertEqual(y.shape, (1, 20))

    def test_skip_2(self):

        template = Layer("layer1", uid="skip_start") >> Layer("layer2") >> Layer("socket")

        config = (
            Config()
            .layer1(nn.Linear, in_features=10, out_features=10)
            .layer2(nn.Linear, in_features=10, out_features=10)
            .socket(Concatenate, dim=1)
            .socket.inputs[1](OutputOf("skip_start"))
        )

        module = template.build(config)
        x = torch.ones(1, 10)
        y = module(x)
        self.assertEqual(y.shape, (1, 20))