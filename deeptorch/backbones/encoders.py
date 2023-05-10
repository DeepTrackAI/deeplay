import torch
import torch.nn as nn

from ... import blocks
from ... import layers

_default = object()


class Encoder2d:
    def __init__(self, blocks=_default, depth=4):
        """Convolutional encoder for 2D images.

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
            self.blocks = [blocks.ConvPoolBlock() for _ in range(depth)]
        else:
            self.blocks = [
                layers.Default(block, blocks.ConvPoolBlock) for block in blocks
            ]

    def build(self, channels_in, channels_out):
        """Build the encoder.

        Parameters
        ----------
        channels_in : int
            Number of input channels for the first block.

        channels_out : list of int
            Number of output channels for each block.
        """

        encoder = nn.ModuleList()

        for i, channels_out in enumerate(channels_out):
            encoder.append(self.blocks[i].build(channels_in, channels_out))
            channels_in = channels_out

        return encoder

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
            self.blocks = [blocks.ConvPoolBlock() for _ in range(depth)]
        else:
            self.blocks = [
                layers.Default(block, blocks.ConvPoolBlock) for block in blocks
            ]

    def build(self, channels_in, channels_out):
        """Build the decoder.

        Parameters
        ----------
        channels_in : int
            Number of input channels for the first block.

        channels_out : list of int
            Number of output channels for each block.
        """

        decoder = nn.ModuleList()

        for i, channels_out in enumerate(channels_out):
            decoder.append(self.blocks[i].build(channels_in, channels_out))
            channels_in = channels_out

        return decoder