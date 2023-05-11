import torch
import torch.nn as nn

from ..layers import Default
from ..backbones.encoders import Encoder2d
from ..heads.classification import CategoricalClassificationHead

_default = object()

class ImageClassifier:

    def __init__(self, backbone=_default, connector=_default, head=_default):
        """Image classifier.

        Parameters
        ----------
        backbone : None, Dict, nn.Module, optional
            Backbone config. If None, a default Encoder2d backbone is used.
            If Dict, it is used as kwargs for the backbone class.
            If nn.Module, it is used as the backbone.
        connector : None, Dict, nn.Module, optional
            Connector config. Connects the backbone to the head by reducing the
            dimensionality of the output of the backbone from 3D (width, height, channels)
            to 1D (channels). 
        head : None, Dict, nn.Module, optional
            Head config. If None, a default CategoricalClassificationHead head is used.
            If Dict, it is used as kwargs for the head class.
            If nn.Module, it is used as the head.
        """
        super().__init__()

        self.assert_valid(backbone)
        self.assert_valid(head)

        self.backbone = Default(backbone, Encoder2d)
        self.connector = Default(connector, nn.Flatten)
        self.head = Default(head, CategoricalClassificationHead)

    def build(self, in_channels, out_channels, image_size=None):
        """Build the image classifier.

        Parameters
        ----------
        in_channels : int
            Number of input channels. E.g. 3 for RGB images, 1 for grayscale images.

        out_channels : int
            Number of output channels. I.e. number of classes.

        image_size : int, optional
            Size of the input image. Can be omitted if a shape invariant connector is used, 
            such as global pooling. 
        """
        backbone = self.backbone.build(in_channels, out_channels)
        connector = self.connector.build()

        # Find the output shape of the backbone.
        if image_size is not None:
            x = torch.zeros(1, in_channels, *image_size)
        else:
            x = backbone(x)
            x = connector(x)


        head = self.head.build(backbone.out_channels, out_channels)

        return nn.Sequential(backbone, connector, head)
