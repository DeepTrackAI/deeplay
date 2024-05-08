from typing import Literal, Type, Union, Optional, overload
import torch.nn as nn
from _typeshed import Incomplete
from deeplay.blocks.base import BaseBlock as BaseBlock, DeferredConfigurableLayer as DeferredConfigurableLayer
from deeplay.external import Layer as Layer
from deeplay.module import DeeplayModule as DeeplayModule
from deeplay.ops.logs import FromLogs as FromLogs
from deeplay.ops.merge import Add as Add, MergeOp as MergeOp
from deeplay.ops.shape import Permute as Permute
from typing import Literal, Type
from typing_extensions import Self

class Conv2dBlock(BaseBlock):
    pool: DeferredConfigurableLayer | nn.Module
    in_channels: Incomplete
    out_channels: Incomplete
    kernel_size: Incomplete
    stride: Incomplete
    padding: Incomplete
    def __init__(self, in_channels: int | None, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, **kwargs) -> None: ...
    def normalized(self, normalization: Type[nn.Module] | DeeplayModule = ..., mode: str = 'append', after: Incomplete | None = None) -> Self: ...
    def pooled(self, pool: Layer = ..., mode: str = 'prepend', after: Incomplete | None = None) -> Self: ...
    def upsampled(self, upsample: Layer = ..., mode: str = 'append', after: Incomplete | None = None) -> Self: ...
    def transposed(self, transpose: Layer = ..., mode: str = 'prepend', after: Incomplete | None = None, remove_upsample: bool = True, remove_layer: bool = True) -> Self: ...
    def strided(self, stride: int | tuple[int, ...], remove_pool: bool = True) -> Self: ...
    def multi(self, n: int = 1) -> Self: ...
    def shortcut(self, merge: MergeOp = ..., shortcut: Literal['auto'] | Type[nn.Module] | DeeplayModule | None = 'auto') -> Self: ...
    @overload
    def style(self, style: Literal["residual"], order: str="lanlan|", activation: Union[Type[nn.Module], Layer]=..., normalization: Union[Type[nn.Module], Layer]=..., dropout: float=0.1) -> Self:
        """Make a residual block with the given order of layers.
    
        Parameters
        ----------
        order : str
            The order of layers in the residual block. The shorthand is a string of 'l', 'a', 'n', 'd' and '|'.
            'l' stands for layer, 'a' stands for activation, 'n' stands for normalization, 'd' stands for dropout,
            and '|' stands for the skip connection. The order of the characters in the string determines the order
            of the layers in the residual block. The characters after the '|' determine the order of the layers after
            the skip connection.
        activation : Union[Type[nn.Module], Layer]
            The activation function to use in the residual block.
        normalization : Union[Type[nn.Module], Layer]
            The normalization layer to use in the residual block.
        dropout : float
            The dropout rate to use in the residual block.
        
    """
    @overload
    def style(self, style: Literal["spatial_self_attention"], to_channel_last: bool=False, normalization: Union[Layer, Type[nn.Module]]=...) -> Self: ...
    @overload
    def style(self, style: Literal["spatial_cross_attention"], to_channel_last: bool=False, normalization: Union[Layer, Type[nn.Module]]=..., condition_name: str="condition") -> Self: ...
    @overload
    def style(self, style: Literal["spatial_transformer"], to_channel_last: bool=False, normalization: Union[Layer, Type[nn.Module]]=..., condition_name: Optional[str]="condition") -> Self: ...
    @overload
    def style(self, style: Literal["resnet"], stride: int=1) -> Self: ...
    @overload
    def style(self, style: Literal["resnet18_input"], ) -> Self: ...
    def style(self, style: str, **kwargs) -> Self: ...

def residual(block: Conv2dBlock, order: str = 'lanlan|', activation: Type[nn.Module] | Layer = ..., normalization: Type[nn.Module] | Layer = ..., dropout: float = 0.1): ...
def spatial_self_attention(block: Conv2dBlock, to_channel_last: bool = False, normalization: Layer | Type[nn.Module] = ...): ...
def spatial_cross_attention(block: Conv2dBlock, to_channel_last: bool = False, normalization: Layer | Type[nn.Module] = ..., condition_name: str = 'condition'): ...
def spatial_transformer(block: Conv2dBlock, to_channel_last: bool = False, normalization: Layer | Type[nn.Module] = ..., condition_name: str | None = 'condition'): ...
