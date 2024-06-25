from deeplay.components.mlp import MultiLayerPerceptron
from ..base import BaseApplicationTest
from deeplay.applications.classification.binary import BinaryClassifier
import torch
import torch.nn


class TestBinaryClassifier(BaseApplicationTest.BaseTest):

    def get_class(self):
        return BinaryClassifier

    def get_networks(self):
        return [
            BinaryClassifier(
                MultiLayerPerceptron(1, [1], 1, out_activation=torch.nn.Sigmoid)
            ),
            BinaryClassifier(
                MultiLayerPerceptron(2, [1], 1), loss=torch.nn.BCEWithLogitsLoss()
            ),
        ]

    def get_training_data(self):
        return [
            (torch.randn(10, 1), torch.randint(0, 2, (10, 1))),
            (torch.randn(10, 2), torch.rand(10, 1)),
        ]

    def test_forward(self):
        for network, (x, y) in zip(self.get_networks(), self.get_training_data()):
            y_pred = network.create()(x)
            self.assertEqual(y_pred.shape, y.shape)
            self.assertIsInstance(y_pred, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
