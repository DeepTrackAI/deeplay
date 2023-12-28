from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import DeeplayModule, Layer, LayerList, LayerActivationNormalization
from .TAU import TransformAggregateUpdate

from .normalization import sparse_normalization

import torch
import torch.nn as nn


class GraphConvolutionalNeuralNetwork(DeeplayModule):
    in_channels: Optional[int]
    hidden_channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[TransformAggregateUpdate]

    @property
    def input_block(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden_blocks(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output_block(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    def __init__(
        self,
        in_channels: Optional[int],
        hidden_channels: Sequence[int],
        out_channels: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if any(h <= 0 for h in hidden_channels):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_channels}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.normalize = Layer(sparse_normalization)
        self.normalize.set_input_map("x", "A")
        self.normalize.set_output_map("A")

        class aggregation(nn.Module):
            def forward(self, x, A):
                if A.is_sparse:
                    return torch.spmm(A, x)
                elif torch.is_tensor(A) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (x.size(0),) * 2,
                        device=A.device,
                    )
                    return torch.spmm(A, x)
                elif torch.is_tensor(A) & (A.size(0) == A.size(1)):
                    return A @ x
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, an edge index tensor, or a square dense tensor.",
                        "Consider updating the aggregate layer to handle alternative formats.",
                    )

        self.blocks = LayerList()

        for i, (c_in, c_out) in enumerate(
            zip([in_channels, *hidden_channels], [*hidden_channels, out_channels])
        ):
            transform = (
                Layer(nn.Linear, c_in, c_out) if c_in else Layer(nn.LazyLinear, c_out)
            )
            transform.set_input_map("x")
            transform.set_output_map("x")

            aggregate = Layer(aggregation)
            aggregate.set_input_map("x", "A")
            aggregate.set_output_map("x")

            update = Layer(nn.ReLU) if i < len(self.hidden_channels) else out_activation
            update.set_input_map("x")
            update.set_output_map("x")

            block = TransformAggregateUpdate(
                transform=transform,
                aggregate=aggregate,
                update=update,
            )
            self.blocks.append(block)

    def forward(self, x):
        x = self.normalize(x)
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        transform: Optional[Type[nn.Module]] = None,
        aggregate: Optional[Type[nn.Module]] = None,
        update: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        index: Union[int, slice, List[Union[int, slice]]],
        order: Optional[Sequence[str]] = None,
        transform: Optional[Type[nn.Module]] = None,
        aggregate: Optional[Type[nn.Module]] = None,
        update: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    configure = DeeplayModule.configure
