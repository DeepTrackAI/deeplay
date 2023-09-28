from ... import Layer, DeeplayModule, Config, Ref

import torch.nn as nn

__all__ = ["VectorRegressionHead", "ImageRegressionHead"]


class VectorRegressionHead(DeeplayModule):
    defaults = (
        Config()
        .num_outputs(1)
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(nn.LazyLinear, out_features=Ref("num_outputs"))
        .output.activation(nn.Identity)
    )

    def __init__(self, num_outputs=1, output=None):
        super().__init__(num_outputs=num_outputs, output=output)

        self.num_outputs = self.attr("num_outputs")
        self.output = self.new("output")

    def forward(self, x):
        x = self.output(x)
        return x


class ImageRegressionHead(DeeplayModule):
    defaults = (
        Config()
        .num_outputs(1)
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(nn.LazyConv2d, out_channels=Ref("num_outputs"), kernel_size=1)
        .output.activation(nn.Identity)
    )

    def __init__(self, num_outputs=1, output=None):
        super().__init__(num_outputs=num_outputs, output=output)

        self.num_outputs = self.attr("num_outputs")
        self.output = self.new("output")

    def forward(self, x):
        x = self.output(x)
        return x
