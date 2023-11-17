from typing import Sequence, Protocol, Callable, Tuple

import torch



class InvertableTransform(Protocol):

    def forward(self, x) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        ...

class InvertableTransforms(object):
    def __init__(self, transforms: Sequence[InvertableTransform]):
        self._transforms = transforms

    def forward(self, x):
        for transform in self._transforms:
            x = transform.forward(x)
        return x

class AffineTransform( )