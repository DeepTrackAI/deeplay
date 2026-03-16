from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from deeplay import DeeplayModule, Layer, LayerList

from ..tpu import TransformPropagateUpdate
from deeplay.ops import Cat

import torch
import torch.nn as nn


class GraphConvolutionalNeuralNetworkConcat(DeeplayModule):
    in_features: Optional[int]
    hidden_features: Sequence[Optional[int]]
    out_features: int
    blocks: LayerList[TransformPropagateUpdate]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.blocks[:-1]`"""
        return self.blocks[:-1]

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.blocks[-1]`."""
        return self.blocks[-1]

    @property
    def transform(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.blocks.transform

    @property
    def propagate(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.propagate

    @property
    def update(self) -> LayerList[Layer]:
        """Return the normalizations of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.update

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        if in_features is None:
            raise ValueError("in_features must be specified")

        if out_features is None:
            raise ValueError("out_features must be specified")

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")

        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_features must be positive, got {hidden_features}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        class Propagate(DeeplayModule):
            def forward(self, x, A):
                if A.is_sparse:
                    return torch.spmm(A, x)
                elif (not A.is_sparse) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (x.size(0),) * 2,
                        device=A.device,
                    )
                    return torch.spmm(A, x)
                elif (not A.is_sparse) & len({A.size(0), A.size(1), x.size(0)}) == 1:
                    return A.type(x.dtype) @ x
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, an edge index tensor, or a square dense tensor.",
                        "Consider updating the propagate layer to handle alternative formats.",
                    )

        self.blocks = LayerList()

        for i, (c_in, c_out) in enumerate(
            zip([in_features, *hidden_features], [*hidden_features, out_features])
        ):
            transform = Layer(nn.Linear, c_in, c_out)
            transform.set_input_map("x")
            transform.set_output_map("x_prime")

            propagate = Layer(Propagate)
            propagate.set_input_map("x_prime", "edge_index")
            propagate.set_output_map("x_prime")

            update = Layer(nn.ReLU) if i < len(self.hidden_features) else out_activation
            update.set_input_map("x_prime")
            update.set_output_map("x_prime")

            block = TransformPropagateUpdate(
                transform=transform,
                propagate=propagate,
                update=update,
                order=["transform", "update", "propagate"]
            )
            self.blocks.append(block)

        self.concat = Cat()
        self.concat.set_input_map('x_prime', 'x')
        self.concat.set_output_map('x')

        self.dense = Layer(nn.Linear, out_features*2, out_features)
        self.dense.set_input_map('x')
        self.dense.set_output_map('x')

        self.activate = Layer(nn.ReLU)
        self.activate.set_input_map('x')
        self.activate.set_output_map('x')

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.concat(x)
        x = self.dense(x)
        x = self.activate(x)

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
