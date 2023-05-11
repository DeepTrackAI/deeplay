import torch.nn as nn

from ..layers import Default

class CategoricalClassificationHead:

    def __init__(self, classifier=None, activation=None):
        """Classification head.

        Parameters
        ----------
        classifier : None, Dict, nn.Module, optional
            Classifier config. If None, a default nn.Linear layer is used.
            If Dict, it is used as kwargs for nn.Linear. Note that `in_features` and
            `out_features` should not be specified.
            If nn.Module, it is used as the classifier.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.Softmax layer is used.
        """
        super().__init__()

        self.classifier = Default(classifier, nn.Linear)
        self.activation = Default(activation, nn.Softmax)

    def build(self, in_features, out_features):
        """Build the head.

        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.
        """
        return nn.Sequential(
            self.classifier.build(in_features, out_features),
            self.activation.build(out_features, out_features),
        )
    
class BinaryClassificationHead:

    def __init__(self, classifier=None, activation=None):
        """Classification head.

        Parameters
        ----------
        classifier : None, Dict, nn.Module, optional
            Classifier config. If None, a default nn.Linear layer is used.
            If Dict, it is used as kwargs for nn.Linear. Note that `in_features` and
            `out_features` should not be specified.
            If nn.Module, it is used as the classifier.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.Softmax layer is used.
        """
        super().__init__()

        self.classifier = Default(classifier, nn.Linear)
        self.activation = Default(activation, nn.Sigmoid)

    def build(self, in_features, out_features):
        """Build the head.

        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.
        """
        return nn.Sequential(
            self.classifier.build(in_features, out_features),
            self.activation.build(out_features, out_features),
        )
    
