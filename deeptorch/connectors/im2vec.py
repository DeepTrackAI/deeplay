"""Image to vector connectors"""

import torch.nn as nn

from .. import Default, LazyModule, default


class FlattenDense(LazyModule):
    def __init__(
        self, out_features, flatten=default, dense=default, activation=default
    ):

        super().__init__()

        self.flatten = Default(flatten, nn.Flatten)
        self.dense = Default(dense, nn.LazyLinear, out_features=out_features)
        self.activation = Default(activation, nn.Sigmoid)

    def build(self):

        return nn.Sequential(
            self.flatten.build(), self.dense.build(), self.activation.build()
        )
