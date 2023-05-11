import torch.nn as nn

from .. import Default, LazyModule, default


class CategoricalClassificationHead(LazyModule):
    def __init__(self, num_classes, classifier=default, activation=default):
        """Classification head.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        classifier : None, Dict, nn.Module, optional
            Classifier config. If None, a default nn.Linear layer is used.
            If Dict, it is used as kwargs for nn.Linear. Note that `in_features` and
            `out_features` should not be specified.
            If nn.Module, it is used as the classifier.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.Softmax layer is used.
        """
        super().__init__()
        self.num_classes = num_classes
        self.classifier = Default(classifier, nn.LazyLinear, num_classes)
        self.activation = Default(activation, nn.Softmax, dim=-1)

    def build(self):
        """Build the head.

        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.
        """
        return nn.Sequential(
            self.classifier.build(),
            self.activation.build(),
        )


class BinaryClassificationHead(LazyModule):
    def __init__(self, classifier=default, activation=default):
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

        self.classifier = Default(classifier, nn.LazyLinear, 1)
        self.activation = Default(activation, nn.Sigmoid)

    def build(self):
        """Build the head."""
        return nn.Sequential(
            self.classifier.build(),
            self.activation.build(),
        )
