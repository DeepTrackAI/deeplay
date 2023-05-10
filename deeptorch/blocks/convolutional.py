import torch
import torch.nn as nn
from . import Block
from ..layers import Default

_default = object()


class ConvActBlock(Block):
    def __init__(self, conv=_default, activation=_default):
        """Convolutional block with activation function.

        Parameters
        ----------
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default nn.Conv2d layer is used.
            If Dict, it is used as kwargs for nn.Conv2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.ReLU layer is used.
        """
        super().__init__()

        self.assert_valid(conv)
        self.assert_valid(activation)

        self.conv = Default(conv, nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)

    def build(self, channels_in, channels_out):

        return nn.Sequential(
            self.conv.build(channels_in, channels_out),
            self.activation.build(channels_out, channels_out),
        )


class ConvActNormBlock(Block):
    def __init__(self, conv=_default, activation=_default, normalization=_default):
        """Convolutional block with activation function and normalization.

        Parameters
        ----------
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default nn.Conv2d layer is used.
            If Dict, it is used as kwargs for nn.Conv2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.ReLU layer is used.
        normalization : None, Dict, nn.Module, optional
            Normalization layer config. If None, a default nn.BatchNorm2d layer is used.
        """
        super().__init__()

        self.assert_valid(conv)
        self.assert_valid(activation)
        self.assert_valid(normalization)

        self.conv = Default(conv, nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)
        self.normalization = Default(normalization, nn.BatchNorm2d)

    def build(self, channels_in, channels_out):

        return nn.Sequential(
            self.conv.build(channels_in, channels_out),
            self.activation.build(channels_out, channels_out),
            self.normalization.build(channels_out, channels_out),
        )


class ConvNormActBlock(Block):
    def __init__(self, conv=_default, activation=_default, normalization=_default):
        """Convolutional block with activation function and normalization.

        Parameters
        ----------
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default nn.Conv2d layer is used.
            If Dict, it is used as kwargs for nn.Conv2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.ReLU layer is used.
        normalization : None, Dict, nn.Module, optional
            Normalization layer config. If None, a default nn.BatchNorm2d layer is used.
        """
        super().__init__()

        self.assert_valid(conv)
        self.assert_valid(activation)
        self.assert_valid(normalization)

        self.conv = Default(conv, nn.Conv2d, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)
        self.normalization = Default(normalization, nn.BatchNorm2d)

    def build(self, channels_in, channels_out):

        return nn.Sequential(
            self.conv.build(channels_in, channels_out),
            self.normalization.build(channels_out, channels_out),
            self.activation.build(channels_out, channels_out),
        )


# ==================================================================================================== #
# POOLING BLOCKS
# ==================================================================================================== #


class ConvPoolBlock(Block):
    def __init__(self, conv=_default, pool=_default):
        """Convolutional block with pooling.

        Parameters
        ----------
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default ConvActBlock is used.
            If Dict, it is used as kwargs for nn.Conv2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        pool : None, Dict, nn.Module, optional
            Pooling layer config. If None, a default nn.MaxPool2d layer is used.
        """
        super().__init__()

        self.assert_valid(conv)
        self.assert_valid(pool)

        self.conv = Default(conv, ConvActBlock)
        self.pool = Default(pool, nn.MaxPool2d, kernel_size=2, stride=2)

    def build(self, channels_in, channels_out):

        return nn.Sequential(
            self.conv.build(channels_in, channels_out),
            self.pool.build(channels_out, channels_out),
        )


class StridedConvPoolBlock(Block):
    def __init__(self, conv=_default, stride=2):
        """Convolutional block that uses strided convolution instead of pooling.

        Parameters
        ----------
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default ConvActBlock is used.
            If Dict, it is used as kwargs for nn.Conv2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        stride : int, optional
            Stride of the convolutional layer.
        """
        super().__init__()

        self.assert_valid(conv)

        self.conv = Default(conv, ConvActBlock, conv=dict(stride=stride))

    def build(self, channels_in, channels_out):

        return nn.Sequential(
            self.conv.build(channels_in, channels_out),
        )


# ==================================================================================================== #
# TRANSPOSE CONVOLUTIONAL BLOCKS
# ==================================================================================================== #
