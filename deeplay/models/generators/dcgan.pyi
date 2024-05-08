from typing import Literal, Type, Union, Optional, overload
from _typeshed import Incomplete
from deeplay.blocks.conv.conv2d import Conv2dBlock as Conv2dBlock
from deeplay.components import ConvolutionalDecoder2d as ConvolutionalDecoder2d
from deeplay.external.layer import Layer as Layer
from deeplay.initializers.normal import Normal as Normal

def dcgan_generator(generator: ConvolutionalDecoder2d): ...

class DCGANGenerator(ConvolutionalDecoder2d):
    latent_dim: int
    output_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int
    label_embedding: Incomplete
    def __init__(self, latent_dim: int = 100, features_dim: int = 128, out_channels: int = 1, class_conditioned_model: bool = False, embedding_dim: int = 100, num_classes: int = 10, output_channels: Incomplete | None = None) -> None: ...
    @overload
    def style(self, style: Literal["cyclegan_resnet_decoder"], ) -> Self: ...
    @overload
    def style(self, style: Literal["dcgan_generator"], ) -> Self: ...
    def style(self, style: str, **kwargs) -> Self: ...
    def forward(self, x, y: Incomplete | None = None): ...
