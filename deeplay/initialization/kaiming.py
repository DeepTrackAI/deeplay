from deeplay.initialization.initializer import Initializer
import torch.nn as nn


class InitializerKaiming(Initializer):
    def __init__(self, mode: str = "fan_in", nonlinearity: str = "relu"):
        self.mode = mode
        self.nonlinearity = nonlinearity

    def initialize_bias(self, tensor):
        tensor.data.fill_(0)

    def initialize_weight(self, tensor):
        nn.init.kaiming_normal_(tensor, mode=self.mode, nonlinearity=self.nonlinearity)
