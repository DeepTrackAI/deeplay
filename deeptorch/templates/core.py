import torch.nn as nn
from ..config import Config
from ..utils import safe_call

__all__ = ["Layer", "InputLayer", "LayerSequence", "LayerAdd", "LayerSub", "LayerMul", "LayerDiv", "Template"]

class Layer:
    def __init__(self, className="", **kwargs):
        self.className = className
    
    def __rshift__(self, other):
        return LayerSequence(self, other)
    
    def __add__(self, other):
        return LayerAdd(self, other)
    
    def __sub__(self, other):
        return LayerSub(self, other)
    
    def __mul__(self, other):
        return LayerMul(self, other)
    
    def __div__(self, other):
        return LayerDiv(self, other)
    
    def build(self, config: Config):
        subconfig = config.with_selector(self.className)

        module = subconfig.get_module()

        if module is None:
            raise ValueError(f"Module not found for selector {self.className} in config {subconfig._rules} with selector {subconfig._context}")

        if hasattr(module, "from_config"):
            return module.from_config(subconfig)
        
        parameters = subconfig.get_parameters()
        return safe_call(module, parameters)
    
    def from_config(self, config):
        return self.build(config)

class InputLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, config):
        return nn.Identity()

class LayerSequence(Layer):

    def __init__(self, *Layers, **kwargs):
        super().__init__(**kwargs)
        self.Layers = Layers

    def __rshift__(self, other):
        return LayerSequence(*self.Layers, other)

    def build(self, config):
        
        modules = {}
        for Layer in self.Layers:
            name = Layer.className or Layer.__class__.__name__
            module = Layer.build(config)
            idx = 0
            while name in modules:
                name = f"{Layer.className}({idx})"
                idx += 1
            
            modules[name] = module

        return Template(modules)


class LayerAdd(Layer):

    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def build(self, config):
        return _TorchAdd(self.a.build(config), self.b.build(config))


class LayerSub(Layer):

    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def build(self, config):
        return _TorchSub(self.a.build(config), self.b.build(config))


class LayerMul(Layer):

    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def build(self, config):
        return _TorchMul(self.a.build(config), self.b.build(config))

class LayerDiv(Layer):

    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def build(self, config):
        return _TorchDiv(self.a.build(config), self.b.build(config))

class Template(nn.ModuleDict):

    
    def __init__(self, kwargs):
        # print("Template", kwargs)
        super().__init__(kwargs)
        
    def forward(self, x):
        for key, module in self.items():
            x = module(x)
        return x

# == Helper classes for arithmetic operations
class _TorchSub(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) - self.b(x)

class _TorchAdd(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) + self.b(x)

class _TorchMul(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) * self.b(x)


class _TorchDiv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) / self.b(x)

