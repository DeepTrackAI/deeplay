from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from .. import DeeplayModule, Layer, LayerList, RecurrentBlock

import torch.nn as nn

class RecurrentDropout(Layer):
    def __init__(self, p=0.0):
        super(RecurrentDropout, self).__init__(classtype=RecurrentDropout)
        self.p = p
        self.dropout = nn.Dropout(p=self.p)

    def forward(self, x):
        if isinstance(x[0],nn.utils.rnn.PackedSequence):
            return nn.utils.rnn.PackedSequence(self.dropout(x[0].data),x[0].batch_sizes),x[1]
        return nn.Dropout(p=self.p)(x[0]),x[1]#self.dropout(x[0]),x[1]

class RecurrentNeuralNetwork(DeeplayModule):
    in_features: Optional[int]
    hidden_features: Sequence[Optional[int]]
    out_features: int
    blocks: LayerList[RecurrentBlock]

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
        in_features: Optional[int],
        hidden_features: Sequence[Optional[int]],
        out_features: int,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

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

        f_out = in_features

        self.blocks = LayerList()
        for c_in, c_out in zip(
            [in_features, *hidden_features], [*hidden_features, out_features]
        ):
            """Torch automatically overwrites dropout==0 for RNN with num_layers=1. To allow for hidden layers of different size, we include dropout layers separately. """
            self.blocks.append(
                RecurrentBlock(
                    Layer(nn.RNN, c_in, c_out), 
                    Layer(nn.Identity, num_features=f_out),
                    Layer(nn.Identity, num_features=f_out),
                    Layer(RecurrentDropout,0.0)
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[List[int]] = None,
        out_features: Optional[int] = None,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ) -> None:
        ...

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
    ) -> None:
        ...

    configure = DeeplayModule.configure
