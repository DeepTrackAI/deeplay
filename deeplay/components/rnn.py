import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..core.templates import Layer
from ..core.core import DeeplayModule
from ..core.config import Config, Ref

class BaseRNN(DeeplayModule):
    """Base RNN module.

    This module serves as a base for various RNN architectures.

    Configurables
    -------------
    - input_size (int): The number of expected features in the input `x`
    - hidden_size (int): The number of features in the hidden state `h`
    - num_layers (int): Number of recurrent layers (Default: 1)
    - dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last layer (Default: 0)
    - rnn_type (str): Type of RNN to use ('LSTM', 'GRU') (Default: 'LSTM')

    Constraints
    -----------
    - input shape: (batch_size, seq_len, input_size)
    - output shape: (batch_size, seq_len, hidden_size * num_directions)

    Evaluation
    ----------
    >>> output, hidden = self.rnn(x)
    >>> return output, hidden

    Examples
    --------
    >>> # With default configuration
    >>> rnn = BaseRNN(Config().update(input_size=my_input_size, hidden_size=my_hidden_size))
    >>> # With custom number of layers and dropout
    >>> rnn = BaseRNN(Config().update(input_size=my_input_size, hidden_size=my_hidden_size, num_layers=2, dropout=0.1, rnn_type='GRU'))

    Return Values
    -------------
    The forward method returns a tuple of two elements:
    - outputs: the output features `h` from the last layer of the RNN, for each timestep
    - hidden: the hidden state for the last timestep for each layer, `h_n`

    Additional Notes
    ----------------
    The `Config` class is used for configuring the RNN. For more details refer to [Config Documentation](#).
    """

    @staticmethod
    def defaults():
        return (
            Config()
            .input_size(None)  # Expected to be set by user
            .hidden_size(256)
            .num_layers(1)
            .dropout(0.1)
            .rnn_type('LSTM')  # Can be 'LSTM' or 'GRU'
        )

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.get('input_size')
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.rnn_type = config.get('rnn_type', 'LSTM')

        rnn_class = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout,
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)
        outputs, hidden = self.rnn(x)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs)
        return outputs, hidden

class EncoderRNN(BaseRNN):
    """Encoder RNN module.

    This module extends the BaseRNN module to create an RNN encoder with GRU units.

    Configurables
    -------------
    Inherits all configurables from BaseRNN, with the addition of:
    - bidirectional (bool): If `True`, becomes a bidirectional RNN (Default: True)
    - Embedding (nn.Embedding): Embedding layer for the input (Default: None)
    
    Constraints
    -----------
    Inherits constraints from BaseRNN, with the output shape adjusted for bidirectionality if enabled.

    Evaluation
    ----------
    Inherits the evaluation process from BaseRNN, with the addition of summing the outputs from both directions if bidirectional.

    Examples
    --------
    >>> # With default configuration
    >>> encoder = EncoderRNN(Config().update(input_size=my_input_size, hidden_size=my_hidden_size, embedding=my_embedding))
    >>> # With custom number of layers, dropout, and bidirectionality
    >>> encoder = EncoderRNN(Config().update(input_size=my_input_size, hidden_size=my_hidden_size, num_layers=2, dropout=0.1, rnn_type='GRU', bidirectional=True))

    Additional Notes
    ----------------
    Inherits additional notes from BaseRNN, with the note that the `embedding` layer should be provided and initialized by the user.
    """

    @staticmethod
    def defaults():
        return (
            BaseRNN.defaults()
            .bidirectional(True)
            .embedding(None)  # Expected to be set by user
        )

    def __init__(self, config):
        super().__init__(config)
        self.bidirectional = config.get('bidirectional', True)
        self.embedding = config.get('embedding')
        if self.embedding is None:
            raise ValueError("An embedding layer must be provided to EncoderRNN.")
        # Adjust the RNN construction for bidirectionality
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            dropout=(0 if self.num_layers == 1 else self.dropout),
            bidirectional=self.bidirectional
            )

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs)
        if self.bidirectional:
            # Sum bidirectional RNN outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden
