from ... import Layer, DeeplayModule, Config, Ref

import torch.nn as nn

__all__ = ["ImageSegmentationHead"]


class ImageSegmentationHead(DeeplayModule):
    defaults = (
        Config()
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(nn.LazyConv2d, out_channels=1, kernel_size=1)
        .output.activation(nn.Sigmoid)
    )

    def __init__(self, output=None):
        """Classification head.

        Parameters
        ----------

        output : Config
            Output layer configuration. Default is a convolutional layer with 1 output channel and sigmoid activation.
        """
        super().__init__(output=output)
        self.output = self.new("output")

    def forward(self, x):
        x = self.output(x)
        return x
