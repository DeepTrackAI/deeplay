import deeplay as dl
import torch.nn as nn


class BackboneResnet18(dl.DeeplayModule):

    def __init__(self, in_channels: int = 3, pool_output: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.pool_output = pool_output

        input_block = dl.blocks.sequential.SequentialBlock(
            layer=dl.Layer(
                nn.Conv2d,
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            normalization=dl.Layer(nn.BatchNorm2d, num_features=64),
            activation=dl.Layer(nn.ReLU, inplace=True),
            pool=dl.Layer(nn.MaxPool2d, kernel_size=3, stride=2, padding=1),
        )

        shortcuts = [
            dl.blocks.sequential.SequentialBlock(
                layer=dl.Layer(nn.Conv2d, 64, 128, kernel_size=1, stride=2, bias=False),
                normalization=dl.Layer(nn.BatchNorm2d, num_features=128),
                order=["layer", "normalization"],
            ),
            dl.blocks.sequential.SequentialBlock(
                layer=dl.Layer(
                    nn.Conv2d, 128, 256, kernel_size=1, stride=2, bias=False
                ),
                normalization=dl.Layer(nn.BatchNorm2d, num_features=256),
                order=["layer", "normalization"],
            ),
            dl.blocks.sequential.SequentialBlock(
                layer=dl.Layer(
                    nn.Conv2d, 256, 512, kernel_size=1, stride=2, bias=False
                ),
                normalization=dl.Layer(nn.BatchNorm2d, num_features=512),
                order=["layer", "normalization"],
            ),
        ]

        blocks = [
            input_block,
            dl.blocks.Conv2dResidual(64, 64),
            dl.blocks.Conv2dResidual(64, 64),
            dl.blocks.Conv2dResidual(64, 128, shortcut=shortcuts[0]).configure(
                "in_layer", stride=2
            ),
            dl.blocks.Conv2dResidual(128, 128),
            dl.blocks.Conv2dResidual(128, 256, shortcut=shortcuts[1]).configure(
                "in_layer", stride=2
            ),
            dl.blocks.Conv2dResidual(256, 256),
            dl.blocks.Conv2dResidual(256, 512, shortcut=shortcuts[2]).configure(
                "in_layer", stride=2
            ),
            dl.blocks.Conv2dResidual(512, 512),
        ]

        if pool_output:
            blocks.append(
                dl.Layer(nn.AdaptiveAvgPool2d, (1, 1)),
            )

        self.blocks = dl.Sequential(*blocks)

        self.initialize(dl.initializers.Kaiming(targets=(nn.Conv2d,)))
        self.initialize(dl.initializers.Constant(targets=(nn.BatchNorm2d,)))

    def forward(self, x):
        x = self.blocks(x)
        if self.pool_output:
            x = x.view(x.size(0), -1)
        return x
