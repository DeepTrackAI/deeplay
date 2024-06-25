from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from deeplay.blocks.sequence.sequence1d import Sequence1dBlock

from deeplay import DeeplayModule, Layer, LayerList, RecurrentBlock

import torch.nn as nn


class RecurrentNeuralNetwork(DeeplayModule):
    in_features: int
    hidden_features: Sequence[int]
    out_features: int
    blocks: LayerList[Sequence1dBlock]

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
    def layer(self) -> LayerList[Layer]:
        """Return the layers of the network. Equivalent to `.blocks.layer`."""
        return self.blocks.layer

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    @property
    def normalization(self) -> LayerList[Layer]:
        """Return the normalizations of the network. Equivalent to `.blocks.normalization`."""
        return self.blocks.normalization

    @property
    def dropout(self) -> LayerList[Layer]:
        """Return the dropout of the network. Equivalent to `.blocks.dropout`."""
        return self.blocks.dropout

    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        out_features: int,
        batch_first: bool = False,
        return_cell_state: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.batch_first = batch_first
        self.return_cell_state = return_cell_state

        if in_features is None:
            raise ValueError("in_features must be specified, got None")
        elif in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        if out_features <= 0:
            raise ValueError(
                f"Number of output features must be positive, got {out_features}"
            )

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )

        self.blocks = LayerList()
        for c_in, c_out in zip(
            [in_features, *hidden_features], [*hidden_features, out_features]
        ):

            self.blocks.append(Sequence1dBlock(c_in, c_out, batch_first=batch_first))

        if return_cell_state:
            self.blocks[-1].configure(return_cell_state=True)

    def bidirectional(self) -> "RecurrentNeuralNetwork":
        """Make the network bidirectional."""
        for block in self.blocks:
            block.bidirectional()
        return self

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
        index: Union[int, slice, List[Union[int, slice]], None] = None,
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None: ...

    configure = DeeplayModule.configure
