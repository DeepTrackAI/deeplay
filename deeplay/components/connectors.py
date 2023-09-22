from .. import (
    Config,
    DeeplayModule,
)

from torch import nn

__all__ = ["VectorToSpatialConnector"]


class VectorToSpatialConnector(DeeplayModule):
    """Converts a vector to a spatial tensor by reshaping it."""

    defaults = Config()

    def __init__(self, spatial_size, **kwargs):
        super().__init__(spatial_size=spatial_size, **kwargs)

        self.spatial_size = self.attr("spatial_size")

    def forward(self, x):
        return x.view(x.size(0), -1, *self.spatial_size)
