from ... import Layer, DeeplayModule, Config, Ref

import torch.nn as nn

__all__ = ["ImageGeneratorHead"]


class ImageGeneratorHead(DeeplayModule):
    defaults = (
        Config()
        .output(Layer("layer") >> Layer("activation"))
        .output.layer(
            nn.LazyConv2d,
            out_channels=Ref("output_size", lambda x: x[0]),
            kernel_size=1,
        )
        .output.activation(nn.Sigmoid)
    )

    def __init__(self, output_size, output=None):
        """Image generator head.

        Parameters
        ----------
        output_size : tuple of int, required
            Expected output size (channels, height, width). If the spatial dimensions of
            the input are different from output_size, the input is interpolated to match
            output_size.
        output : Config
            Output layer configuration. Default is a linear layer with `
        """
        super().__init__(
            output_size=output_size,
            output=output,
        )

        self.output_size = self.attr("output_size")
        self.output = self.new("output")

    def forward(self, x):
        if self.output_size is not None:
            x = nn.functional.interpolate(x, size=self.output_size[1:])
        x = self.output(x)
        return x
