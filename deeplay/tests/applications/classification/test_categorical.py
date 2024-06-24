from deeplay.components.mlp import MultiLayerPerceptron
from ..base import BaseApplicationTest
from deeplay.applications.classification.categorical import CategoricalClassifier
import torch
import torch.nn


class TestBinaryClassifier(BaseApplicationTest.BaseTest):

    def get_class(self):
        return CategoricalClassifier

    def get_networks(self):
        return [
            CategoricalClassifier(MultiLayerPerceptron(1, [1], 2)),
            CategoricalClassifier(MultiLayerPerceptron(2, [1], 3)),
        ]

    def get_training_data(self):
        return [
            (torch.randn(10, 1), torch.randint(0, 2, (10,)).long()),
            (torch.randn(10, 2), torch.randint(0, 3, (10,)).long()),
        ]
