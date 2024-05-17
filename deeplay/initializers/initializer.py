class Initializer:

    def __init__(self, targets):
        self.targets = targets

    def initialize(self, module, tensors=("weight", "bias")):
        if isinstance(module, self.targets):
            for tensor in tensors:
                if hasattr(module, tensor) and getattr(module, tensor) is not None:
                    self.initialize_tensor(getattr(module, tensor), name=tensor)

    def initialize_tensor(self, tensor, name):
        raise NotImplementedError
