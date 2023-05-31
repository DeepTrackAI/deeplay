import torch
import torch.nn as nn

from .. import blocks as bl
from .. import Default, LazyModule, default


class Encoder2d(LazyModule):
    def __init__(self, channels_out=None, blocks=default):
        """Convolutional encoder for 2D images.

        Parameters
        ----------
        channels_out : list of int
            Number of output channels for each block. Unused
            if blocks is specified.
        blocks : list of LazyModule blocks, optional
            List of blocks to use. If not specified, a default list of four
            ConvPoolBlock blocks is used.
        """
        super().__init__()

        if blocks is default:
            self.blocks = [bl.ConvPoolBlock(ch) for ch in channels_out]
        else:
            self.blocks = blocks

    def build(self):
        """Build the encoder.

        Parameters
        ----------
        channels_in : int
            Number of input channels for the first block.

        channels_out : list of int
            Number of output channels for each block.
        """

        return nn.Sequential(*[block.build() for block in self.blocks])

