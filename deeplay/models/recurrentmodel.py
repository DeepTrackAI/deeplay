from typing import Optional, overload, Sequence, List, Dict, Any, Literal, Union, Type
from deeplay import (
    DeeplayModule,
    Layer,
    LayerList,
    MultiLayerPerceptron,
)

import torch

from deeplay.blocks.sequence.sequence1d import Sequence1dBlock
from deeplay.components.rnn import RecurrentNeuralNetwork


class RecurrentModel(RecurrentNeuralNetwork):
    """
    Recurrent Neural Network (RNN) model.

    This RNN can be configured to be a simple RNN, LSTM, or GRU, with options for bidirectionality,
    number of layers, and other typical RNN configurations. It supports embedding layers and can be
    customized with different activation functions for the output layer.

    Configurables
    -------------
    - in_features (int): The number of expected features in the input. Must be specified.
    - hidden_features (Sequence[int]): The number of features in each hidden layer.
    - out_features (Optional[int]): Number of features in the output layer. If None, the final RNN layer's output is returned directly.
    - rnn_type (Literal['RNN', 'LSTM', 'GRU']): Type of RNN. Defaults to 'RNN'.
    - out_activation (Union[Literal['softmax', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu', 'none'], torch.nn.Module]):
      Activation function for the output layer. Can be a string specifying the activation type or an instance of a PyTorch Module. Defaults to 'none'.
    - bidirectional (bool): If True, makes the RNN bidirectional. Defaults to False.
    - batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature). Defaults to True.
    - dropout (float): Dropout value for the outputs of each RNN layer except the last layer. Defaults to 0.
    - embedding (Optional[torch.nn.Embedding]): An embedding layer to be applied to the input data. If None, no embedding is applied.

    Properties
    ----------
    - input: Returns the input layer of the network.
    - hidden: Returns the hidden layers of the network.
    - output: Returns the output layer of the network.
    - layer: Returns all layers of the network.
    - activation: Returns the activation functions used in the network.
    - normalization: Returns the normalization layers used in the network, if any.

    Methods
    -------
    - forward(x, lengths): Defines the forward pass of the RNN.

    Notes
    -----
    The RNN module is designed to be flexible and configurable, allowing for various RNN types and structures.
    """

    in_features: Optional[int]
    hidden_features: Sequence[int]
    out_features: Optional[int]
    rnn_type: Optional[str]
    out_activation: Union[Type[torch.nn.Module], torch.nn.Module, None]
    batch_first: bool
    embedding: Optional[torch.nn.Embedding]
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
        """Return the dropout layers of the network. Equivalent to `.blocks.dropout`."""
        return self.blocks.dropout

    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        out_features: Optional[int] = None,
        return_sequence: bool = False,
        return_cell_state: bool = False,
        rnn_type: Union[Literal["RNN", "LSTM", "GRU"], Type[torch.nn.Module]] = "LSTM",
        out_activation: Union[Type[torch.nn.Module], torch.nn.Module, None] = None,
        bidirectional: bool = False,
        batch_first: bool = True,
        dropout: float = 0,
        embedding: Optional[torch.nn.Embedding] = None,
    ):

        self.return_sequence = return_sequence
        self.return_cell_state = return_cell_state
        self.in_features = in_features

        self.embedding = embedding
        if embedding:
            self.embedding_dropout = torch.nn.Dropout(dropout)

        super().__init__(
            in_features,
            hidden_features=hidden_features[:-1],
            out_features=hidden_features[-1],
            batch_first=batch_first,
            return_cell_state=return_cell_state,
        )
        for block in self.blocks:
            if rnn_type == "LSTM" or rnn_type == torch.nn.LSTM:
                block.LSTM()
            elif rnn_type == "GRU" or rnn_type == torch.nn.GRU:
                block.GRU()
            elif rnn_type == "RNN" or rnn_type == torch.nn.RNN:
                block.RNN()
            else:
                block.layer.configure(rnn_type)

        for block in self.blocks:
            if bidirectional:
                block.bidirectional()
            if dropout > 0:
                block.set_dropout(dropout)

        self.bidirectional = bidirectional

        if out_activation is None:
            out_activation = Layer(torch.nn.Identity)
        elif isinstance(out_activation, type) and issubclass(
            out_activation, torch.nn.Module
        ):
            out_activation = Layer(out_activation)

        # Define the output layer
        if out_features is not None:
            self.head = MultiLayerPerceptron(
                hidden_features[-1],
                [],
                out_features,
                flatten_input=False,
                out_activation=out_activation.new(),
            )

    def forward(self, x):

        if self.embedding:
            x = self.embedding(x)
            x = self.embedding_dropout(x)

        outputs = x
        if self.return_cell_state:
            outputs, hidden = super().forward(outputs)
        else:
            outputs = super().forward(outputs)

        if self.bidirectional:
            outputs = (
                outputs[:, :, : self.hidden_features[-1]]
                + outputs[:, :, self.hidden_features[-1] :]
            )

        if not self.return_sequence:
            if self.batch_first:
                outputs = outputs[:, -1, :]
            else:
                outputs = outputs[-1, :, :]

        if self.out_features is not None:
            outputs = self.head(outputs)
            if self.return_cell_state:
                return outputs, hidden
            return outputs

        if self.return_cell_state:
            return outputs, hidden
        return outputs

    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[Sequence[int]] = None,
        out_features: Optional[int] = None,
        rnn_type: Optional[str] = None,
        bidirectional: Optional[bool] = None,
        out_activation: Optional[str] = None,
        batch_first: Optional[bool] = None,
        dropout: Optional[float] = None,
        embedding: Optional[torch.nn.Embedding] = None,
    ) -> None: ...

    configure = DeeplayModule.configure
