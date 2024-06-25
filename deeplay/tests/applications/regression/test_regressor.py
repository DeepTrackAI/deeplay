from deeplay.components.mlp import MultiLayerPerceptron
from ..base import BaseApplicationTest
from deeplay.applications.regression.regressor import Regressor
import torch
import torch.nn


class TestRegressor(BaseApplicationTest.BaseTest):

    def get_class(self):
        return Regressor

    def get_networks(self):
        return [
            Regressor(MultiLayerPerceptron(1, [1], 1)),
            Regressor(MultiLayerPerceptron(2, [1], 1)),
            Regressor(MultiLayerPerceptron(1, [1], 2)),
        ]

    def get_training_data(self):
        return [
            (torch.randn(10, 1), torch.randn(10, 1)),
            (torch.randn(10, 2), torch.randn(10, 1)),
            (torch.randn(10, 1), torch.randn(10, 2)),
        ]

    def test_forward(self):
        for network, (x, y) in zip(self.get_networks(), self.get_training_data()):
            y_pred = network.create()(x)
            self.assertEqual(y_pred.shape, y.shape)
            self.assertIsInstance(y_pred, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
