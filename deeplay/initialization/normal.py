from deeplay.initialization.initializer import Initializer


class InitializerNormal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def initialize_bias(self, tensor):
        tensor.data.fill_(self.mean)

    def initialize_weight(self, tensor):
        tensor.data.normal_(self.mean, self.std)
