class Initializer:

    def __init__(self, targets):
        self.targets = targets

    def initialize(self, module):
        if isinstance(module, self.targets):
            if hasattr(module, "weight") and module.weight is not None:
                self.initialize_weight(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                self.initialize_bias(module.bias)

    def initialize_weight(self, tensor):
        pass

    def initialize_bias(self, tensor):
        pass
