# %%

import deeplay as dl


class BackboneResnet18(dl.ConvolutionalEncoder2d):

    pool: dl.Layer

    def __init__(self, in_channels: int = 3, pool_output: bool = False):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=[3, 3],
            out_channels=512,
        )
        self.pool_output = pool_output
        # self.blocks[0].style("resnet", stride=1)
        self.blocks[1].style("resnet", stride=1)

    def forward(self, x):
        x = super().forward(x)
        if self.pool_output:
            x = self.pool(x).squeeze()
        return x


resnet = BackboneResnet18(in_channels=3)
resnet.__construct__()
# %%
[].clear()
