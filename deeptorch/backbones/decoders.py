import torch.nn as nn


from .. import blocks as bl
from .. import Default, LazyModule, default

class Decoder2d:
    def __init__(self, blocks=_default, depth=4):
        """Convolutional decoder for 2D images.

        Parameters
        ----------
        blocks : list, optional
            List of blocks to use. If not specified, a default list of four
            ConvPoolBlock blocks is used.
        depth : int, optional
            Number of blocks to use.
            If blocks is specified, this parameter is ignored.
        """
        super().__init__()

        if blocks is _default:
            self.blocks = [bl.
        else:
            self.blocks = [
                layers.Default(block, blocks.ConvPoolBlock) for block in blocks
            ]

    def build(self):
        """Build the decoder.

        Parameters
        ----------
        channels_in : int
            Number of input channels for the first block.

        channels_out : list of int
            Number of output channels for each block.
        """

        return nn.ModuleList(block.build() for block in self.blocks)
