# %%
from deeplay.core import v2
import torch.nn as nn

from dataclasses import dataclass


class LinearHead(v2.DeeplayModule):
    """A simple linear head."""

    in_features: int
    out_features: int

    layer_1 = v2.Layer(nn.Linear)
    layer_2 = v2.Layer(nn.Sigmoid)

    def before_build(self):
        self.layer_1.default(
            in_features=self.in_features, out_features=self.out_features
        )

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


class MultiLayerPerceptron(v2.DeeplayModule):
    in_features: int
    hidden_features: list[int]
    out_features: int

    @v2.layerlist
    def blocks(self, idx):
        is_first = idx == 0
        is_last = idx == len(self.blocks) - 1
        in_features = self.in_features if is_first else self.hidden_features[idx - 1]
        out_features = self.out_features if is_last else self.hidden_features[idx]

        return v2.Sequential(
            v2.Layer(nn.Linear, in_features=in_features, out_features=out_features),
            v2.Layer(nn.ReLU if not is_last else nn.Identity),
        )

    @blocks.length
    def blocks(self):
        return len(self.hidden_features) + 1

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


model = MultiLayerPerceptron(10, [20], 2)
model.configure(hidden_features=[20, 30])
model.blocks.configure(nn.Linear, in_features=10, out_features=20)
# model.build()


# %%
