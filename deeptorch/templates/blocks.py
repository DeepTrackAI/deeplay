from ..core.template import Element
import torch.nn as nn


class Default:
    ...


class Block(Element):
    ...


class ConvolutionalBlock(Block):
    def __init__(self):
        super().__init__()

        self.layer = Default(ConvActBlock)

    def build(self, in_channels, out_channels):
        return self.layer.build(in_channels, out_channels)


class ConvActBlock(Block):
    def __init__(self):
        super().__init__()

        self.layer = Default(nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(nn.ReLU)

    def build(self, in_channels, out_channels):
        return nn.Sequential(
            self.layer.build(in_channels, out_channels),
            self.activation.build(in_channels, out_channels),
        )


class ConvActNormBlock(Block):
    def __init__(self):
        super().__init__()

        self.layer = Default(nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(nn.ReLU)
        self.normalization = Default(nn.BatchNorm2d)

    def build(self, in_channels, out_channels):
        return nn.Sequential(
            self.layer.build(in_channels, out_channels),
            self.activation.build(out_channels, out_channels),
            self.normalization.build(out_channels, out_channels),
        )


class ConvNormActBlock(Block):
    def __init__(self):
        super().__init__()

        self.layer = Default(nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(nn.ReLU)
        self.normalization = Default(nn.BatchNorm2d)

    def build(self, in_channels, out_channels):
        return nn.Sequential(
            self.layer.build(in_channels, out_channels),
            self.normalization.build(out_channels, out_channels),
            self.activation.build(out_channels, out_channels),
        )


class PoolBlock(Block):
    def __init__(self):
        super().__init__()
        self.layer = Default(nn.MaxPool2d)

    def build(self, in_channels, out_channels):
        return self.layer.build(in_channels, out_channels)


class DimensionalityReductionBlock(Block):
    def __init__(self):
        super().__init__()
        self.layer = Default(nn.Flatten, start_dim=1)

    def build(self, in_channels, out_channels):
        return self.layer.build(in_channels, out_channels)
