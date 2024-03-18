from typing import Optional,overload,Sequence,List,Dict,Any,Literal,Union,Type
from deeplay import DeeplayModule, Layer, LayerList, RecurrentBlock,MultiLayerPerceptron,LayerActivationNormalizationDropout

import torch

class RecurrentDropout(torch.nn.Module):
    def __init__(self, p=0.0):
        super(RecurrentDropout, self).__init__()
        self.p = p
        self.dropout = torch.nn.Dropout(p=self.p)

    def forward(self, x):
        if isinstance(x[0],torch.nn.utils.rnn.PackedSequence):
            return torch.nn.utils.rnn.PackedSequence(self.dropout(x[0].data),x[0].batch_sizes),x[1]
        return self.dropout(x[0]),x[1]

class RecurrentModel(DeeplayModule):
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
    bidirectional: bool
    batch_first: bool
    dropout: float
    embedding: Optional[torch.nn.Embedding] 
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
        """Return the dropout layers of the network. Equivalent to `.blocks.dropout`."""
        return self.blocks.dropout


    def __init__(
        self, 
        in_features: Optional[int], 
        hidden_features: Sequence[int], 
        out_features: Optional[int] = None,
        rnn_type: Union[Literal['RNN', 'LSTM', 'GRU'],Type[torch.nn.Module]] = 'RNN',
        out_activation: Union[Type[torch.nn.Module], torch.nn.Module, None] = None,
        bidirectional: bool = False, 
        batch_first: bool = True, 
        dropout: float = 0,
        embedding: Optional[torch.nn.Embedding] = None,
    ):
        super().__init__()
        self.embedding=embedding
        self._in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if in_features is None:
            raise NotImplementedError("in_features must be specified, Lazy RNNs are not supported")
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
        if isinstance(rnn_type, str) and rnn_type not in ['RNN', 'LSTM', 'GRU']:    
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'LSTM', 'GRU'], got {rnn_type}"
            )

        if embedding:
            self.embedding_dropout=torch.nn.Dropout(dropout)

        if isinstance(rnn_type, type) and issubclass(rnn_type, torch.nn.Module):
            self.rnn_class = rnn_type
        elif self.rnn_type == 'LSTM':
            self.rnn_class = torch.nn.LSTM
        elif self.rnn_type == 'GRU':
            self.rnn_class = torch.nn.GRU
        else:
            self.rnn_class = torch.nn.RNN
             
        if out_activation is None:
            out_activation = Layer(torch.nn.Identity)
        elif isinstance(out_activation, type) and issubclass(out_activation, torch.nn.Module):
            out_activation = Layer(out_activation)

        self.blocks = LayerList()
        self.blocks.append(
                RecurrentBlock(
                    Layer(self.rnn_class, in_features, hidden_features[0],bidirectional=bidirectional,batch_first=batch_first), 
                    Layer(torch.nn.Identity),
                    Layer(torch.nn.Identity),
                    Layer(RecurrentDropout,dropout)
                )
            )
        
        if len(hidden_features) > 1:
            for i in range(len(hidden_features) - 1):
    
                self.blocks.append(
                    RecurrentBlock(
                        Layer(self.rnn_class, 
                              hidden_features[i] * 2 if bidirectional else hidden_features[i],
                              hidden_features[i + 1],bidirectional=bidirectional,batch_first=batch_first), 
                        Layer(torch.nn.Identity),
                        Layer(torch.nn.Identity),
                        Layer(RecurrentDropout,dropout) if i < len(hidden_features) - 2 else Layer(RecurrentDropout,0.0)
                    )
                )

        # Define the output layer
        if self.out_features is not None:
            self.blocks.append(LayerActivationNormalizationDropout(
                Layer(torch.nn.Linear, hidden_features[-1], self.out_features),
                out_activation,
                Layer(torch.nn.Identity, num_features=self.out_features),
                Layer(torch.nn.Dropout, p=0),
            ))

    def forward(self, x):

        if self.embedding:
            x = self.embedding(x)
            x = self.embedding_dropout(x)

        outputs=x
        for block in self.blocks[:-1]:
            outputs,hidden=block(outputs)

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


