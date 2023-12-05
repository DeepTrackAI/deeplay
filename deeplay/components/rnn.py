from typing import Optional,overload,Sequence,List,Dict,Any,Literal,Union
from .. import DeeplayModule, Layer, LayerList, LayerActivation
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
    hidden_features: Sequence[int]
    rnn_type: Optional[str]
    bidirectional: bool
    batch_first: bool
    dropout: float
    blocks: LayerList[LayerActivation]
    
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
        in_features: Optional[int], 
        hidden_features: Sequence[int], 
        out_features: Optional[int] = None,
        rnn_type: Literal['RNN', 'LSTM', 'GRU'] = 'GRU',
        out_activation: Union[Literal['softmax', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu', 'none'], torch.nn.Module] = 'none',
        bidirectional: bool = False, 
        batch_first: bool = True, 
        dropout: float = 0.1,
        embedding: Optional[torch.nn.Embedding] = None
    ):
        super(RecurrentNeuralNetwork, self).__init__()

        self.embedding=embedding
        self._in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if in_features is None:
            raise ValueError("in_features must be specified")
        if hidden_features is None or hidden_features == []:
            raise ValueError("hidden_features must be specified")
        if in_features is not None and in_features <= 0:
            raise ValueError(f"in_channels must be positive, got {in_features}")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_channels must be positive, got {hidden_features}"
            )
        if out_features is not None:
            if out_features <= 0:
                raise ValueError(
                    f"Number of output features must be positive, got {out_features}"
                )
        if rnn_type not in ["RNN", "LSTM", "GRU"]:
            raise ValueError(
                f"rnn_type must be one of 'RNN', 'LSTM', or 'GRU', got {rnn_type}"
            )
        
        if embedding:
            self.embedding_dropout=torch.nn.Dropout(dropout)

        if self.rnn_type == 'LSTM':
             self.rnn_class = torch.nn.LSTM
        elif self.rnn_type == 'GRU':
            self.rnn_class = torch.nn.GRU
        else:
            self.rnn_class = torch.nn.RNN
             
        if isinstance(out_activation, torch.nn.Module):
            self.out_activation=Layer(out_activation)
            print(self.out_activation)
        else:
            if out_activation == 'softmax':
                self.out_activation = Layer(torch.nn.Softmax,dim=1)
            elif out_activation == 'sigmoid':
                self.out_activation = Layer(torch.nn.Sigmoid)
            elif out_activation == 'tanh':
                self.out_activation = Layer(torch.nn.Tanh)
            elif out_activation == 'relu':
                self.out_activation = Layer(torch.nn.ReLU)
            elif out_activation == 'leaky_relu':
                self.out_activation = Layer(torch.nn.LeakyReLU)
            elif out_activation == 'gelu':
                self.out_activation = Layer(torch.nn.GELU)
            else:
                self.out_activation = Layer(torch.nn.Identity)
            
        self.blocks = LayerList()

        rnn = Layer(self.rnn_class,
        input_size=in_features,
        hidden_size=hidden_features[0],
        num_layers=1,
        dropout=dropout,
        bidirectional=bidirectional,
        )
        self.blocks.append(LayerActivation(rnn,Layer(torch.nn.Identity,num_features=hidden_features[0])))

        if len(hidden_features) > 2:
            for i in range(len(hidden_features) - 2):
                self.blocks.append(LayerActivation(
                        Layer(self.rnn_class,
                        input_size=hidden_features[i] * 2 if bidirectional else hidden_features[i],
                        hidden_size=hidden_features[i + 1],
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=bidirectional),
                        Layer(torch.nn.Identity,num_features=hidden_features[i+1])))
                
        if len(hidden_features) > 1:
                self.blocks.append(LayerActivation(
                        Layer(self.rnn_class,
                        input_size=hidden_features[-2] * 2 if bidirectional else hidden_features[-2],
                        hidden_size=hidden_features[-1],
                        num_layers=1,
                        dropout=0,
                        bidirectional=bidirectional),
                        self.out_activation if not out_features else Layer(torch.nn.Identity,num_features=hidden_features[-1])))
                
        # Define the output layer
        if self.out_features is not None:
            output_layer = LayerActivation(Layer(torch.nn.Linear, in_features=hidden_features[-1],out_features=out_features),self.out_activation)
            self.blocks.append(output_layer)

    def forward(self, x, lengths: Optional[torch.Tensor] = None):

        if self.embedding:
            x = self.embedding(x)
            x = self.embedding_dropout(x)
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        outputs, hidden = self.blocks[0](x)
        for layer in self.blocks[1:-1]:
          outputs, hidden = layer(outputs)
        if not self.out_features:
           outputs,hidden = self.blocks[-1](outputs)
        if lengths is not None:
           outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
           outputs = outputs[:, :, :self.hidden_features[-1]] + outputs[:, :, self.hidden_features[-1]:]

        if self.out_features is not None:
           if self.batch_first:
               outputs = outputs[:, -1, :]
           else:
               outputs = outputs[-1, :, :]
           outputs=torch.nn.Flatten()(outputs)
           outputs = self.blocks[-1](outputs)
           return outputs


        return outputs, hidden #return last hidden state


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
        embedding: Optional[torch.nn.Embedding] = None
    ) -> None:
        ...
        
    configure = DeeplayModule.configure


