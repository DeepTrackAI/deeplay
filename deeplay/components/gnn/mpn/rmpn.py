from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from deeplay import DeeplayModule, Layer, LayerList, LayerSkip
from deeplay.ops import Cat

from deeplay.components.dict import AddDict

from ..tpu import TransformPropagateUpdate

from .transformation import Transform
from .propagation import Sum
from .update import Update

import torch.nn as nn


class ResidualMessagePassingNeuralNetwork(DeeplayModule):
    hidden_features: Sequence[Optional[int]]
    out_features: int
    blocks: LayerList[TransformPropagateUpdate]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0].layer

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1].layer

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1].layer

    @property
    def transform(self) -> LayerList[Layer]:
        """Return the transform layers of the network. Equivalent to `.blocks.transform`."""
        return self.blocks.layer.transform

    @property
    def propagate(self) -> LayerList[Layer]:
        """Return the propagate layers of the network. Equivalent to `.blocks.propagate`."""
        return self.blocks.layer.propagate

    @property
    def update(self) -> LayerList[Layer]:
        """Return the update layers of the network. Equivalent to `.blocks.update`."""
        return self.blocks.layer.update

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.out_features = out_features

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )

        if out_features is None:
            raise ValueError("out_features must be specified")

        if out_features <= 0:
            raise ValueError(
                f"Number of output features must be positive, got {out_features}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.blocks = LayerList()
        for i, c_out in enumerate([*hidden_features, out_features]):
            activation = (
                Layer(nn.ReLU) if i < len(hidden_features) - 1 else out_activation
            )

            transform = Transform(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            transform.set_input_map("x", "edge_index", "edge_attr")
            transform.set_output_map("edge_attr")

            propagate = Sum()
            propagate.set_input_map("x", "edge_index", "edge_attr")
            propagate.set_output_map("aggregate")

            update = Update(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            update.set_input_map("x", "aggregate")
            update.set_output_map("x")

            block = TransformPropagateUpdate(
                transform=transform,
                propagate=propagate,
                update=update,
            )
            residual_block = LayerSkip(layer=block, skip=AddDict("x", "edge_attr"))
            self.blocks.append(residual_block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[List[int]] = None,
        out_features: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None: ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        transform: Optional[Type[nn.Module]] = None,
        propagate: Optional[Type[nn.Module]] = None,
        update: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]]],
        order: Optional[Sequence[str]] = None,
        transform: Optional[Type[nn.Module]] = None,
        propagate: Optional[Type[nn.Module]] = None,
        update: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None: ...

    configure = DeeplayModule.configure
