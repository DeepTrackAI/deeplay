from typing import Literal, Type, Union, Optional, overload
from _typeshed import Incomplete
from deeplay.blocks.sequential import SequentialBlock as SequentialBlock
from deeplay.components import ConvolutionalEncoder2d as ConvolutionalEncoder2d
from deeplay.external.layer import Layer as Layer
from deeplay.initializers.normal import Normal as Normal

def dcgan_discriminator(encoder: ConvolutionalEncoder2d): ...

class DCGANDiscriminator(ConvolutionalEncoder2d):
    input_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int
    in_channels: Incomplete
    features_dim: Incomplete
    label_embedding: Incomplete
    def __init__(self, in_channels: int = 1, features_dim: int = 64, class_conditioned_model: bool = False, embedding_dim: int = 100, num_classes: int = 10, input_channels: Incomplete | None = None) -> None: ...
    @overload
    def style(self, style: Literal["resnet18"], ) -> Self: ...
    @overload
    def style(self, style: Literal["cyclegan_resnet_encoder"], ) -> Self: ...
    @overload
    def style(self, style: Literal["cyclegan_discriminator"], ) -> Self: ...
    @overload
    def style(self, style: Literal["dcgan_discriminator"], ) -> Self: ...
    def style(self, style: str, **kwargs) -> Self: ...
    def forward(self, x, y: Incomplete | None = None): ...
