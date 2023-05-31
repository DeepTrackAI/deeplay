import torch
import torch.nn as nn
from . import Block
from .. import Default, default


class ConvActBlock(Block):
    def __init__(self, channels_out, conv=default, activation=default):
        """Convolutional block with activation function.

        Parameters
        ----------
        output_channels : int
            Number of output channels.
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

        self.conv = Default(conv, nn.LazyConv2d, channels_out, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)

    def build(self):

        return nn.Sequential(
            self.conv.build(),
            self.activation.build(),
        )


class ConvActNormBlock(Block):
    def __init__(
        self, channels_out, conv=default, activation=default, normalization=default
    ):
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

        self.conv = Default(conv, nn.LazyConv2d, channels_out, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)
        self.normalization = Default(normalization, nn.LazyBatchNorm2d)

    def build(self):

        return nn.Sequential(
            self.conv.build(),
            self.activation.build(),
            self.normalization.build(),
        )


class ConvNormActBlock(Block):
    def __init__(
        self, channels_out, conv=default, activation=default, normalization=default
    ):
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

        self.conv = Default(conv, nn.LazyConv2d, channels_out, kernel_size=3, padding=1)
        self.activation = Default(activation, nn.ReLU)
        self.normalization = Default(normalization, nn.LazyBatchNorm2d)

    def build(self):

        return nn.Sequential(
            self.conv.build(),
            self.normalization.build(),
            self.activation.build(),
        )


# ==================================================================================================== #
# POOLING BLOCKS
# ==================================================================================================== #


class ConvPoolBlock(Block):
    def __init__(self, channels_out, conv=default, pool=default):
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

        self.conv = Default(conv, ConvActBlock, channels_out)
        self.pool = Default(pool, nn.MaxPool2d, kernel_size=2, stride=2)

    def build(self):

        return nn.Sequential(
            self.conv.build(),
            self.pool.build(),
        )


class StridedConvPoolBlock(Block):
    def __init__(self, channels_out, conv=default, stride=2):
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

        self.conv = Default(conv, ConvActBlock, channels_out, conv=dict(stride=stride))

    def build(self):

        return self.conv.build()


# ==================================================================================================== #
# TRANSPOSE CONVOLUTIONAL BLOCKS
# ==================================================================================================== #


class ConvTransposeActBlock(Block):
    def __init__(self, channels_out, conv=default, activation=default, stride=2):
        """Transpose convolutional block with activation function.

        Parameters
        ----------
        output_channels : int
            Number of output channels.
        conv : None, Dict, nn.Module, optional
            Convolutional layer config. If None, a default nn.ConvTranspose2d layer is used.
            If Dict, it is used as kwargs for nn.ConvTranspose2d. Note that `in_channels` and
            `out_channels` should not be specified.
            If nn.Module, it is used as the convolutional layer.
        activation : None, Dict, nn.Module, optional
            Activation function config. If None, a default nn.ReLU layer is used.
        """
        super().__init__()

        self.assert_valid(conv)
        self.assert_valid(activation)

        self.conv = Default(
            conv,
            nn.LazyConvTranspose2d,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.activation = Default(activation, nn.ReLU)

    def build(self):

        return nn.Sequential(
            self.conv.build(),
            self.activation.build(),
        )
