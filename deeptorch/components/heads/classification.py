from ... import Layer, DeepTorchModule, Config, Ref

import torch.nn as nn


class CategoricalClassificationHead(DeepTorchModule):
    defaults = (
        Config()
        .num_classes(2)
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(nn.LazyLinear, out_features=Ref("num_classes"))
        .output.activation(nn.Softmax, dim=-1)
    )

    def __init__(self, num_classes, num_blocks=0, block=None, output=None):
        """Classification head.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        num_blocks : int
            Number of blocks before the output layer.
            Default is 0.
        block : Config
            Block configuration.
            Default is a linear layer with 64 output features and ReLU activation.
        output : Config
            Output layer configuration.
            Default is a linear layer with `num_classes` output features and softmax activation.
        """
        super().__init__(
            num_classes=num_classes,
            num_blocks=num_blocks,
            block=block,
            output=output,
        )

        self.num_classes = self.attr("num_classes")
        self.output = self.create("output")

    def forward(self, x):
        x = self.output(x)
        return x
