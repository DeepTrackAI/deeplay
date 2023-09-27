from ...templates import Layer
from ...core import DeeplayModule
from ...config import Config, Ref

import torch.nn as nn


def encoderDecoderConfig():
    return Config().depth(4)


class EncoderDecoder(DeeplayModule):
    @staticmethod
    def defaults():
        return encoderDecoderConfig()

    def __init__(self, depth=4, encoder_blocks=None, decoder_blocks=None):
        super().__init__(
            depth=depth, encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks
        )

        self.depth = self.attr("depth")

        self.encoder_blocks = nn.ModuleList(
            self.new("encoder_blocks", i) for i in range(self.depth)
        )
        self.decoder_blocks = nn.ModuleList(
            self.new("decoder_blocks", i) for i in range(self.depth)
        )

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        for block in self.decoder_blocks:
            x = block(x)
        return x
