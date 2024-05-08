from deeplay.components.mlp import MultiLayerPerceptron
import torch.nn as nn

from deeplay.external.layer import Layer


@MultiLayerPerceptron.register_style
def normed_leaky(mlp: MultiLayerPerceptron):

    mlp["blocks", :-1].all.normalized(nn.BatchNorm1d).activated(
        Layer(nn.LeakyReLU, negative_slope=0.05)
    )


class SmallMLP(MultiLayerPerceptron):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, [32, 32], out_features)
        self.style("normed_leaky")


class MediumMLP(MultiLayerPerceptron):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, [64, 128], out_features)
        self.style("normed_leaky")


class LargeMLP(MultiLayerPerceptron):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, [128, 128, 128], out_features)
        self.style("normed_leaky")


class XLargeMLP(MultiLayerPerceptron):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, [128, 256, 512, 512], out_features)
        self.style("normed_leaky")
