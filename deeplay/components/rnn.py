from typing import Optional,overload,Sequence,List,Dict,Any

from .. import DeeplayModule, Layer, LayerList
import torch

class RecurrentNeuralNetwork(DeeplayModule):
    """
    Recurrent Neural Network (RNN) module.

    This RNN can be configured to be a simple RNN, LSTM, or GRU, with options for bidirectionality, 
    number of layers, and other typical RNN configurations.

    Configurables
    -------------
    - in_features (int): The number of expected features in the input.
    - hidden_features (int): The number of features in the hidden state.
    - num_layers (int): Number of recurrent layers. (Default: 1)
    - nonlinearity (str): The non-linearity to use ('tanh' or 'relu'). Only used when `rnn_type` is 'RNN'. (Default: 'tanh')
    - rnn_type (str): Type of RNN ('RNN', 'LSTM', or 'GRU'). (Default: 'RNN')
    - bidirectional (bool): If True, becomes a bidirectional RNN. (Default: False)
    - batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). (Default: True)
    - dropout (float): If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer. (Default: 0.0)
    """
    
    in_features: Optional[int] 
    hidden_features: Sequence[Optional[int]] 
    rnn_type: Optional[str]
    bidirectional: bool
    batch_first: bool
    dropout: float
    blocks: LayerList[Layer]
    
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

    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = 256, 
        num_layers: Optional[int] = 1,
        out_features: Optional[int] = 0,
        rnn_type: str = 'LSTM', 
        bidirectional: bool = False, 
        batch_first: bool = True, 
        dropout: float = 0.1
    ):
        super(RecurrentNeuralNetwork, self).__init__()

        self._in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

            
        if in_features is not None and in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        if hidden_features <= 0:
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )

        self.blocks = LayerList()
        rnn_class = torch.nn.LSTM if self.rnn_type == 'LSTM' else torch.nn.GRU
        self.rnn = rnn_class(
            in_features,
            hidden_features,
            num_layers,
            dropout=dropout,
        )
        #self.blocks.append(Layer(self.rnn))
        self.blocks.append(self.rnn)
        
        if bidirectional:
                layer_output_size = hidden_features * 2
        else:
            layer_output_size = hidden_features

        # Define the output layer
        
        if self.out_features>0:
            self.output_layer = Layer(torch.nn.Linear, layer_output_size, out_features)


    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        outputs, hidden = self.rnn(x)
        if lengths is not None:
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.output_layer is not None:
            outputs = self.output_layer(outputs)
        return outputs, hidden


    @overload
    def configure(
        self,
        /,
        in_features: Optional[int] = None,
        hidden_features: Optional[Sequence[int]] = None,
        out_features: Optional[int] = None,
        rnn_type: Optional[str] = None,
        bidirectional: Optional[bool] = None,
        batch_first: Optional[bool] = None,
        dropout: Optional[float] = None
    ) -> None:
        ...

    @overload
    def configure(
        self,
        network_config: Dict[str, Any]  # A dictionary containing all the configuration parameters
    ) -> None:
        ...
        
    configure = DeeplayModule.configure

class EncoderRNN(RecurrentNeuralNetwork):
    """
    Encoder RNN module.

    This module extends the RecurrentNeuralNetwork module to create an RNN encoder with GRU units.

    Configurables
    -------------
    Inherits all configurables from RecurrentNeuralNetwork, with the addition of:
    - Embedding (nn.Embedding): Embedding layer for the input (Default: None)
    
    Constraints
    -----------
    Inherits constraints from RecurrentNeuralNetwork, with the output shape adjusted for bidirectionality if enabled.

    Evaluation
    ----------
    Inherits the evaluation process from RecurrentNeuralNetwork, with the addition of summing the outputs from both directions if bidirectional.

    Examples
    --------
    >>> # Example usage
    >>> encoder = EncoderRNN(in_features=my_input_size, hidden_features=my_hidden_features, out_features=my_out_features, embedding=my_embedding_layer, rnn_type='GRU', bidirectional=True)
    """

    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = 256, 
        num_layers: Optional[int] = 1,
        out_features: Optional[int] = 0,
        rnn_type: str = 'LSTM', 
        bidirectional: bool = False, 
        batch_first: bool = True, 
        dropout: float = 0.1,
        embedding: Optional[torch.nn.Embedding] = None
        ):
        # Initialize the parent class
        super().__init__(in_features, hidden_features, num_layers, out_features, rnn_type, bidirectional, batch_first, dropout)

        self.embedding = embedding
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        if self.embedding is None:
            raise ValueError("An embedding layer must be provided to EncoderRNN.")
        
        # Adjust the RNN construction for bidirectionality
        self.rnn = torch.nn.GRU(
            in_features,
            hidden_features,
            num_layers,
            dropout=(0 if self.num_layers == 1 else self.dropout),
            bidirectional=bidirectional
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        if self.embedding:
            input_seq = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths, enforce_sorted=False)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            # Sum bidirectional RNN outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden
    
import torch.nn as nn
import torch.nn.functional as F
from .. import DeeplayModule

class DecoderRNN(RecurrentNeuralNetwork):
    """
    Decoder RNN module.

    This module is used for decoding sequences in a sequence-to-sequence model, extending the BaseRNN.

    Configurables
    -------------
    Inherits all configurables from BaseRNN, with the addition of:
    - output_size (int): The size of the output
    """

    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = 256, 
        num_layers: Optional[int] = 1,
        out_features: Optional[int] =0,
        rnn_type: str = 'LSTM', 
        bidirectional: bool = False, 
        batch_first: bool = True, 
        dropout: float = 0.1,
        embedding: Optional[torch.nn.Embedding] = None
        ):
        # Initialize the parent class
        super().__init__(in_features, hidden_features, num_layers, 0, rnn_type, bidirectional, batch_first, dropout)
        self.embedding = embedding
        self.embedding_dropout=torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(hidden_features, out_features)
        self.n_layers=num_layers

        if self.embedding is None:
            raise ValueError("An embedding layer must be provided to DecoderRNN.")

    def forward(self, input_step, last_hidden):
        """
        Forward pass for the decoder for a single step.

        Parameters
        ----------
        - input_step: The input at the current timestep
        - last_hidden: The hidden state from the previous timestep
        """
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        #embedded = embedded.unsqueeze(0)  # Add batch dimension

        # Forward through unidirectional RNN
        rnn_output, hidden = self.rnn(embedded, last_hidden)

        # Squeeze the output to remove extra dimension
        rnn_output = rnn_output.squeeze(0)

        # Predict next word using the RNN output
        output = self.out(rnn_output)
        output = F.softmax(output, dim=1)

        return output, hidden