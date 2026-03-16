from __future__ import annotations
from typing import Optional, Sequence, Type, Union
import warnings

from deeplay import (
    DeeplayModule,
    Layer,
    LayerList,
)
from deeplay.components.gnn import MessagePassingNeuralNetworkGAUDI, GraphConvolutionalNeuralNetworkConcat
from deeplay.components.gnn.pooling import MinCutPooling, MinCutUpsampling
from deeplay.ops import Cat
from deeplay.components.gnn.pooling import GlobalGraphPooling, GlobalGraphUpsampling
# from deeplay.components.gnn.mpn import TransformOnlySenderNodes
from deeplay.components.mlp import MultiLayerPerceptron
from deeplay.ops import GetEdgeFeaturesNew
# from deeplay.components.gnn.mpn.propagation import Mean

import torch.nn as nn
    

class GraphEncoder(DeeplayModule):
    """ A Graph Encoder module that leverages multiple graph processing blocks to learn representations
    from graph-structured data. This module supports graph convolution and pooling operations, enabling
    effective encoding of graph information for downstream tasks.

    Parameters
    ----------
    hidden_features: int 
        The number of hidden features in the hidden layers, both in the gcn and pooling, of the encoder.
    num_blocks: int
        The number of processing blocks in the encoder.
    num_clusters: list[int]
        The number of clusters the graph is pooled to in each processing block.
    thresholds: list[float]
        The threshold values for the connectivity in the clustering process.
    
    Configurables
    -------------
    - hidden features (int): Number of features of the hidden layers.
    - num_blocks (int): Number of processing blocks in the encoder.
    - num_clusters list[int]: Number of clusters the graph is pooled to in each processing block.
    - thresholds list[int]: The threshold values for the connectivity in the clustering process.
    - poolings (template-like):A list of pooling layers to use. Default is using MinCut pooling for all layers,
        except for the last, which is global pooling.
    - save_intermediates (bool): Flag indicating whether to save intermediate adjacency matrices and other information, useful
        when using it together with the GraphDecoder. Default is True.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        - edge_index: torch.Tensor of shape (2, num_edges)
        - batch: torch.Tensor of shape (num_nodes)
        - edge_attr: torch.Tensor of shape (num_edges, edge_features)

    Example
    ----------
    >>> encoder = dl.GraphEncoder(hidden_features=96, num_blocks=3, num_clusters=[5, 3, 1], thresholds=[0.1, 0.2, None], save_intermediates=False).build()
    >>> inp = {}
    >>> inp["x"] = torch.randn(10, 16)
    >>> inp['batch'] = torch.zeros(10, dtype=int)
    >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
    >>> inp["edge_attr"] = torch.randn(20, 8)
    >>> output = encoder(inp)

    
    Return Values
    -------------
    The forward method returns a mapping object with the updated node features, edge_index, edge_attributes,
    and the cut and orthogonality losses from the MinCut pooling.

    """
    hidden_features: int
    num_blocks: int
    num_clusters: Sequence[int]
    thresholds: Optional[Sequence[float]]
    poolings: Optional[Sequence[nn.Module]]
    save_intermediates: Optional[bool]
     
    def __init__(
            self,
            hidden_features: int,
            num_blocks: int,
            num_clusters: Sequence[int],
            thresholds: Optional[Sequence[float]] = None,
            poolings: Optional[Sequence[Union[Type[nn.Module], nn.Module]]] = None,
            save_intermediates: Optional[bool] = True,
        ):
        super().__init__(
            hidden_features = hidden_features,
            num_blocks = num_blocks,
            num_clusters = num_clusters,
            thresholds = thresholds,
            poolings = poolings,
            save_intermediates = save_intermediates,
            )
        
        if not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError(f"hidden_features must be a positive integer, got {hidden_features}")
        
        if poolings is None:
            poolings = [MinCutPooling] * (num_blocks - 1) + [GlobalGraphPooling]

        assert len(poolings) == num_blocks, "Number of poolings should match num_blocks."
        assert len(num_clusters) == num_blocks, "Lenght of number of clusters should match num_blocks."


        self.message_passing = MessagePassingNeuralNetworkGAUDI(
            hidden_features=[hidden_features],
            out_features=hidden_features,
            out_activation=nn.ReLU
        )

        # self.message_passing.transform = TransformOnlySenderNodes(
        #     combine=Cat(),
        #     layer=Layer(nn.LazyLinear, hidden_features),
        #     activation=nn.ReLU,
        # )
        
        # self.message_passing.transform.set_input_map("x", "edge_index", "input_edge_attr")
        # self.message_passing.propagate = Mean()
        # self.message_passing.propagate.set_input_map("x", "edge_index", "edge_attr")

        self.dense = Layer(nn.Linear, hidden_features, hidden_features)
        self.dense.set_input_map('x')
        self.dense.set_output_map('x')

        self.activate = Layer(nn.ReLU)
        self.activate.set_input_map('x')
        self.activate.set_output_map('x')
        

        self.blocks = LayerList()

        for i in range(num_blocks):
            pool = poolings[i]

            if save_intermediates == True:
                edge_index_map = "edge_index" if i == 0 else f"edge_index_{i}"
                select_output_map = f"s_{i}"
                connect_output_map = f"edge_index_{i+1}"
                batch_input_map = "batch" if i == 0 else f"batch_{i}"
                batch_output_map = f"batch_{i+1}"
                mincut_cut_loss_map = f"L_cut_{i}"
                mincut_ortho_loss_map = f"L_ortho_{i}"

                block = GraphEncoderBlock(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    num_clusters=num_clusters[i],
                    threshold=thresholds[i] if thresholds is not None else None,
                    pool=pool,
                    edge_index_map=edge_index_map,
                    select_output_map=select_output_map,
                    connect_output_map=connect_output_map,
                    batch_input_map=batch_input_map,
                    batch_output_map=batch_output_map,
                    mincut_cut_loss_map=mincut_cut_loss_map,
                    mincut_ortho_loss_map=mincut_ortho_loss_map,
                )

            else:
                mincut_cut_loss_map = f"L_cut_{i}"
                mincut_ortho_loss_map = f"L_ortho_{i}"

                block = GraphEncoderBlock(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    num_clusters=num_clusters[i],
                    threshold=thresholds[i] if thresholds is not None else None,
                    pool=pool,
                    mincut_cut_loss_map=mincut_cut_loss_map,
                    mincut_ortho_loss_map=mincut_ortho_loss_map,
            )

            self.blocks.append(block)          
    
    def forward(self, x):
        x['input_edge_index'] = x['edge_index']     # Do this in a nicer way
        x['input_edge_attr'] = x['edge_attr']
        x = self.message_passing(x)
        x = self.dense(x)
        x = self.activate(x)
        for block in self.blocks:
            x = block(x)
        return x
    

class GraphDecoder(DeeplayModule):
    """
    A Graph Decoder module that reconstructs graph structures from learned representations generated 
    by the GraphEncoder. This module aims to decode the latent graph features back into graph node
    and edge attributes.

    Parameters
    ----------
    hidden_features: int 
        The dimensionality of the hidden layers of the decoder. This should match the hidden 
        features from the corresponding GraphEncoder.
    num_blocks: int
        The number of processing blocks in the decoder. This should match the number of blocks
        of the GraphEncoder.
    output_node_dim: int 
        The dimensionality of the output node features. This should match the original dimensionallity
        of the input node features of the GraphEncoder.
    output_edge_dim: int 
        The dimensionality of the output edge features. This should match the original dimensionallity 
        of the input edge attributes of the GraphEncoder.
    
    Configurables
    -------------
    - hidden features (int): Number of features of the hidden layers.
    - num_blocks (int): Number of processing blocks in the decoder.
    - output_node_dim (int): Number of dimensions of the output node features.
    - output_edge_dim (int): Number of dimensions of the output edge attributes.
    - upsamplings (template-like): A list of upsampling layers to use. Default is using MinCut upsampling
        for all layers, except for the first, which is global upsampling. This should reflect the pooling
        layers of the GraphEncoder.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        - edge_index: torch.Tensor of shape (2, num_edges)
        - batch: torch.Tensor of shape (num_nodes)
    
    Example
    ----------
    >>> encoder = dl.GraphEncoder(hidden_features=96, num_blocks=3, num_clusters=[20, 5, 1], thresholds=[0.1, 0.5, None], save_intermediates=False).build()
    >>> inp = {}
    >>> inp["x"] = torch.randn(10, 16)
    >>> inp['batch'] = torch.zeros(10, dtype=int)
    >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
    >>> inp["edge_attr"] = torch.randn(20, 8)
    >>> encoder_output = encoder(inp)
    >>> decoder = dl.GraphDecoder(hidden_features=96, num_blocks=3, output_node_dim=16, output_edge_dim=8).build()
    >>> decoder_output = decoder(encoder_output)

    Return Values
    -------------
    The forward method returns a mapping object with the reconstructed node features and edge attributes.

    """

    hidden_features: int
    num_blocks: int
    output_node_dim: int
    output_edge_dim: int
    upsamplings: Optional[Sequence[nn.Module]]

    def __init__(
            self,
            hidden_features: int,
            num_blocks: int,
            output_node_dim: int,
            output_edge_dim: int,
            upsamplings: Optional[Sequence[Union[Type[nn.Module], nn.Module]]] = None,
        ):
        super().__init__(
            hidden_features = hidden_features,
            output_node_dim = output_node_dim, 
            output_edge_dim = output_edge_dim,
            num_blocks = num_blocks,
            upsamplings = upsamplings,
            )
        
        if not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError(f"hidden_features must be a positive integer, got {hidden_features}")

        if upsamplings is None:
            upsamplings = [GlobalGraphUpsampling] + [MinCutUpsampling] * (num_blocks - 1)

        assert len(upsamplings) == num_blocks, "Number of upsamplings should match num_blocks."

        self.blocks = LayerList()

        for i in range(num_blocks):
            upsample = upsamplings[i]
            edge_index_map = "edge_index" if i == num_blocks-1 else f"edge_index_{num_blocks-1-i}"
            select_input_map = f"s_{num_blocks-1-i}"
            connect_input_map = f"edge_index_{num_blocks-i}"

            block = GraphDecoderBlock(
                in_features=hidden_features,
                out_features=hidden_features,
                upsample=upsample,
                edge_index_map=edge_index_map,
                select_input_map=select_input_map,
                connect_input_map=connect_input_map,
            )

            self.blocks.append(block)    
      
        self.dense = Layer(nn.Linear, hidden_features, hidden_features)     
        self.dense.set_input_map('x')
        self.dense.set_output_map('x')    

        self.activate = Layer(nn.ReLU)
        self.activate.set_input_map('x')
        self.activate.set_output_map('x') 
            
        self.get_edge_attr = GetEdgeFeaturesNew()
        self.get_edge_attr.set_input_map("x", "input_edge_index")
        self.get_edge_attr.set_output_map("edge_attr_sender", "edge_attr_receiver") 

        self.dense_sender = Layer(nn.Linear, hidden_features, hidden_features)     
        self.dense_sender.set_input_map('edge_attr_sender')
        self.dense_sender.set_output_map('edge_attr_sender')    

        self.activate_sender = Layer(nn.ReLU)
        self.activate_sender.set_input_map('edge_attr_sender')
        self.activate_sender.set_output_map('edge_attr_sender')  

        self.dense_receiver = Layer(nn.Linear, hidden_features, hidden_features)     
        self.dense_receiver.set_input_map('edge_attr_receiver')
        self.dense_receiver.set_output_map('edge_attr_receiver')    

        self.activate_receiver = Layer(nn.ReLU)
        self.activate_receiver.set_input_map('edge_attr_receiver')
        self.activate_receiver.set_output_map('edge_attr_receiver') 

        self.concat_edge_attr = Cat()
        self.concat_edge_attr.set_input_map('edge_attr_sender', 'edge_attr_receiver')
        self.concat_edge_attr.set_output_map('edge_attr')

        self.dense_edge_mlp_1 = Layer(nn.Linear, hidden_features * 2, hidden_features)     
        self.dense_edge_mlp_1.set_input_map('edge_attr')
        self.dense_edge_mlp_1.set_output_map('edge_attr')   

        self.activate_edge_mlp_1 = Layer(nn.ReLU)
        self.activate_edge_mlp_1.set_input_map('edge_attr')
        self.activate_edge_mlp_1.set_output_map('edge_attr') 

        self.dense_edge_mlp_2 = Layer(nn.Linear, hidden_features, output_edge_dim)     
        self.dense_edge_mlp_2.set_input_map('edge_attr')
        self.dense_edge_mlp_2.set_output_map('edge_attr')   

        # # get the edge features:
        # self.edge_mlp = MultiLayerPerceptron(
        #     in_features = hidden_features * 2,
        #     hidden_features = [hidden_features],
        #     out_features = output_edge_dim,
        #     out_activation = None,
        # )        
        # self.edge_mlp.set_input_map('edge_attr')
        # self.edge_mlp.set_output_map('edge_attr') 

        # get the node features:
        self.node_mlp = MultiLayerPerceptron(
            in_features = hidden_features,
            hidden_features = [hidden_features, hidden_features],
            out_features = output_node_dim,
            out_activation = None,
        )        
        self.node_mlp.set_input_map('x')
        self.node_mlp.set_output_map('x')  

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.dense(x)
        x = self.activate(x)

        x = self.get_edge_attr(x)
        x = self.dense_sender(x)
        x = self.activate_sender(x)
        x = self.dense_receiver(x)
        x = self.activate_receiver(x)
        x = self.concat_edge_attr(x)

        x = self.dense_edge_mlp_1(x)
        x = self.activate_edge_mlp_1(x)

        x = self.dense_edge_mlp_2(x)

        # x = self.edge_mlp(x)

        x = self.node_mlp(x)
        return x
    

class GraphEncoderBlock(DeeplayModule): 
    """
    A Graph Encoder Block that processes graph data through a Graph Convolutional Neural Network (GCN)
    and applies pooling operations to generate encoded representations of the graph structure. 
    This block is a fundamental component of the GraphEncoder, enabling hierarchical feature extraction.

    Parameters
    ----------
    in_features: int
        The number of input features for each node in the graph.
    out_features: int
        The number of output features for each node after processing.

    Configurables
    -------------
    - in_features (int): The number of input features for each node in the graph.
    - out_features (int): The number of output features for each node after processing.
    - pool (template-like): The pooling operation to be used. Defaults to MinCutPooling.
    - num_clusters (int): The number of clusters for MinCutPooling. Must be provided if using MinCutPooling.
    - threshold (float): Threshold value for pooling operations.
    - edge_index_map (str): The mapping for edge index inputs. Defaults to "edge_index".
    - select_output_map (str): The mapping for the selection outputs from the pooling layer. Defaults to "s".
    - connect_output_map (str): The mapping for connecting outputs to subsequent layers. Defaults to "edge_index_pool".
    - batch_input_map (str): The mapping for batch input. Defaults to "batch".
    - batch_output_map (str): The mapping for batch output. Defaults to "batch".
    - mincut_cut_loss_map (str): The mapping for saving the mincut cut loss. Defaults to "L_cut".
    - mincut_ortho_loss_map (str): The mapping for saving the mincut orthogonallity loss. Defaults to "L_ortho".

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        - edge_index: torch.Tensor of shape (2, num_edges)
        - batch: torch.Tensor of shape (num_nodes)
        - edge_attr: torch.Tensor of shape (num_edges, edge_features)

    Example
    ----------
        >>> block = dl.GraphEncoderBlock(in_features=16, out_features=16, num_clusters=5, threshold=0.1).build()
        >>> inp = {}
        >>> inp["x"] = torch.randn(10, 16)
        >>> inp['batch'] = torch.zeros(10, dtype=int)
        >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
        >>> inp["edge_attr"] = torch.randn(20, 8)
        >>> output = block(inp)
    """

    in_features: Optional[int]
    hidden_features: Sequence[Optional[int]]
    out_features: int
    pool: Optional[nn.Module]
    num_clusters: Optional[int]
    threshold: Optional[float]
    edge_index_map: Optional[str]  
    select_output_map: Optional[str] 
    connect_output_map: Optional[str]
    batch_input_map: Optional[str]
    batch_output_map: Optional[str]
    mincut_cut_loss_map: Optional[str]
    mincut_ortho_loss_map: Optional[str]

    def __init__(
            self,
            in_features: int,
            out_features: int,
            pool: Optional[Union[Type[nn.Module], nn.Module, None]] = MinCutPooling,
            num_clusters: Optional[int] = None,
            threshold: Optional[float] = None,
            edge_index_map: Optional[str] = "edge_index",    
            select_output_map: Optional[str] = "s",          
            connect_output_map: Optional[str] = "edge_index_pool",
            batch_input_map: Optional[str] = "batch",
            batch_output_map: Optional[str] = "batch",
            mincut_cut_loss_map: Optional[str] = 'L_cut',
            mincut_ortho_loss_map: Optional[str] = 'L_ortho',
        ):
        super().__init__(
            in_features = in_features,
            num_clusters = num_clusters,
            threshold = threshold,
            out_features = out_features,
            pool = pool,
        )
        self.edge_index_map = edge_index_map
        self.connect_output_map = connect_output_map
    
        self.gcn = GraphConvolutionalNeuralNetworkConcat(        
            in_features=in_features,
            hidden_features=[],  
            out_features=out_features,
            out_activation=nn.ReLU, 
            )

        self.gcn.propagate.set_input_map("x", edge_index_map)

        if pool == MinCutPooling:
            if num_clusters is None:
                raise ValueError("num_clusters must be provided for MinCutPooling")
            
            self.pool = pool(hidden_features=[out_features], num_clusters=num_clusters, threshold=threshold)
            self.pool.mincut_loss.set_input_map(edge_index_map, select_output_map)
            self.pool.mincut_loss.set_output_map(mincut_cut_loss_map, mincut_ortho_loss_map)
        else:
            self.pool = pool()
        
        self.pool.select.set_output_map(select_output_map)

        if hasattr(self.pool, 'reduce'):
            self.pool.reduce.set_input_map('x', select_output_map)
        if hasattr(self.pool, 'batch_compatible'):
            self.pool.batch_compatible.set_input_map(select_output_map, batch_input_map)
            self.pool.batch_compatible.set_output_map(select_output_map, batch_output_map)
        if hasattr(self.pool, 'connect'):
            self.pool.connect.set_input_map(edge_index_map, select_output_map)
            self.pool.connect.set_output_map(connect_output_map)
        if hasattr(self.pool, 'red_self_con') and self.pool.red_self_con is not None:
            self.pool.red_self_con.set_input_map(connect_output_map)   
            self.pool.red_self_con.set_output_map(connect_output_map)
        if hasattr(self.pool, 'apply_threshold') and self.pool.apply_threshold is not None:
            self.pool.apply_threshold.set_input_map(connect_output_map)
            self.pool.apply_threshold.set_output_map(connect_output_map)
        if hasattr(self.pool, 'sparse'):
            self.pool.sparse.set_input_map(connect_output_map)
            self.pool.sparse.set_output_map(connect_output_map)

    def forward(self, x):
        x = self.gcn(x)
        x = self.pool(x)
        return x
    

class GraphDecoderBlock(DeeplayModule): 
    """
    A Graph Decoder Block that upsamples a graph and applies a Graph Convolutional Neural Network (GCN).
    This block is a fundamental component of the GraphDecoder, enabling the reconstruction of graph features
    in a Graph Encoder Decoder model.
     
    Parameters
    ----------
    in_features: int
        The number of input features for each node in the graph.
    out_features: int
        The number of output features for each node after processing.
   
    Configurables
    -------------
    - in_features (int): The number of input features for each node in the graph.
    - out_features (int): The number of output features for each node after processing.
    - upsample (template-like): The upsampling operation to be used. Defaults to MinCutUpsampling.
    - edge_index_map (str): The mapping for edge index inputs. Defaults to "edge_index".
    - select_input_map (str): The mapping for selection inputs for the upsampling layer. Defaults to "s".
    - connect_input_map (str): The mapping for the connectivity for the upsampling layer. Defaults to "edge_index_pool".
    - connect_output_map (str): The mapping for the connectivity outputs of the upsampling layer. Defaults to "-".
    - batch_map (str): The mapping for batch inputs or outputs. Defaults to "batch".
        
    Example
    ----------
        >>> encoderblock = dl.GraphEncoderBlock(in_features=16, out_features=16, num_clusters=5, threshold=0.2).build()
        >>> inp = {}
        >>> inp["x"] = torch.randn(10, 16)
        >>> inp['batch'] = torch.zeros(10, dtype=int)
        >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
        >>> inp["edge_attr"] = torch.randn(20, 8)
        >>> encoderblock_output = encoderblock(inp)
        >>> decoderblock = dl.GraphDecoderBlock(in_features=16, out_features=16).build()
        >>> decoderblock_output = decoderblock(encoderblock_output)

    """
    in_features: int
    out_features: int
    upsample: Optional[nn.Module]
    edge_index_map: Optional[str]    
    select_input_map: Optional[str]  
    connect_input_map: Optional[str] 
    connect_output_map: Optional[str] 
    batch_map: Optional[str]

    def __init__(
            self,
            in_features: int,
            out_features: int,
            upsample: Optional[Union[Type[nn.Module], nn.Module, None]] = MinCutUpsampling,
            edge_index_map: Optional[str] = "edge_index",   
            select_input_map: Optional[str] = "s",          
            connect_input_map: Optional[str] = "edge_index_pool",
            connect_output_map: Optional[str] = "-",
        ):
        super().__init__(
            in_features = in_features,
            out_features = out_features,
            upsample = upsample,
            edge_index_map=edge_index_map,      
            select_input_map=select_input_map,  
            connect_input_map=connect_input_map,
        )

        if upsample == MinCutUpsampling:
            self.upsample = upsample()
            self.upsample.upsample.set_input_map('x', connect_input_map, select_input_map)
            self.upsample.upsample.set_output_map('x', connect_output_map)

        else:
            self.upsample = upsample()
            self.upsample.upsample.set_input_map('x', select_input_map)
      
        self.gcn = GraphConvolutionalNeuralNetworkConcat(   
            in_features=in_features,                                                                            
            hidden_features=[],  
            out_features=out_features,
            out_activation=nn.ReLU, 
            )
        
        self.gcn.propagate.set_input_map("x", edge_index_map)

    def forward(self, x):
        x = self.upsample(x)
        x = self.gcn(x)
        return x