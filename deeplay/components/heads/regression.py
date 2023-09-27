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

    def forward(self, x):
        x = self.output(x)
        return x


class ImageRegressionHead(DeeplayModule):
    defaults = (
        Config()
        .num_outputs(1)
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(nn.Conv2d, out_channels=Ref("num_outputs"), kernel_size=1)
        .output.activation(nn.Identity)
    )

    def forward(self, x):
        x = self.output(x)
        return x
