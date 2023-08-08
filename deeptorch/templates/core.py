import torch.nn as nn
from ..config import Config
from ..utils import safe_call

__all__ = ["Node"]

class Node:
    def __init__(self, className="", **kwargs):
        self.className = className
    
    def __rshift__(self, other):
        return NodeSequence(self, other)
    
    def build(self, config):
        subconfig = config.with_selector(self.className)

        module = subconfig.get("module")

        if module is None:
            raise ValueError(f"Module not found for selector {self.className} in config {subconfig._rules} with selector {subconfig._selector}")

        if hasattr(module, "from_config"):
            return module.from_config(subconfig)
        
        parameters = subconfig.get_parameters()
        return safe_call(module, parameters)
    
    def from_config(self, config):
        return self.build(config)
    

class NodeSequence(Node):

    def __init__(self, *nodes, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nodes

    def __rshift__(self, other):
        return NodeSequence(*self.layers, other)

    def build(self, config):
        return nn.Sequential(*[node.build(config) for node in self.nodes])


# # Should be moved to components

# class Encoder(DeepTorchModule):
#     """
#     Parameters
#     ----------
#     depth : int, optional
#         Number of layers in the encoder.
#     layers : dict, LayerConfig, optional
#         Layer config for each layer in the encoder.
#             """

#     defaults = {
#         "block": Node("layer") >> Node("activation") >> Node("pool"),
#         "layer.module": nn.Conv2d,
#         "layer.channels_out": lambda i: 8 * 2 ** i,
#         "layer.kernel_size": 3,
#         "layer.padding": 1,
#         "activation.module": nn.ReLU,
#         "pool.module": nn.MaxPool2d,
#         "pool.kernel_size": 2,
#         "pool.stride": 2,
#     }

#     def __init__(self, depth=4, blocks=None):

#         super().__init__(depth=depth, blocks=blocks)
        
#         self.depth = self.attr("depth")
#         self.layers = [self.create("block", i) for i in range(self.depth)]

#     def forward(self, x): 
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class Decoder(DeepTorchModule):
#     """
#     Parameters
#     ----------
#     depth : int, optional
#         Number of layers in the decoder.
#     layers : dict, LayerConfig, optional
#         Layer config for each layer in the decoder.
#     """

#     defaults = {
#         "block": Node("layer") >> Node("activation") >> Node("upsample"),
#         "layer.module": nn.Conv2d,
#         "layer.channels_out": lambda i: 8 * 2 ** (self.depth - i - 1),
#         "layer.kernel_size": 3,
#         "layer.padding": 1,
#         "activation.module": nn.ReLU,
#         "upsample.module": nn.Upsample,
#         "upsample.scale_factor": 2,
#     }

#     def __init__(self, depth=4, blocks=None):

#         super().__init__(depth=depth, blocks=blocks)

#         self.depth = self.request_attr("depth")
#         self.layers = [self.create("block", i) for i in range(self.depth)]

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
# class EncoderDecoder(DeepTorchModule):
#     """
#     Parameters
#     ----------
#     depth : int, optional
#         Number of layers in the encoder and decoder.
#     layers : dict, LayerConfig, optional
#         Layer config for each layer in the encoder and decoder.
#     """

#     defaults = {
#         "encoder": Node("encoder"),
#         "encoder.module": Encoder,
#         "decoder": Node("decoder"),
#         "decoder.module": Decoder,
#         "depth": 4,
#     }

#     def __init__(self, encoder=None, decoder=None):

#         super().__init__(encoder=encoder, decoder=decoder)

#         self.encoder = self.create("encoder")
#         self.decoder = self.create("decoder")

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
# EncoderDecoder(
#     Config()
#         .encoder.layer.module(nn.Conv2d)
#         .encoder.layer.channels_out(lambda i: 8 * 2 ** i)
# )