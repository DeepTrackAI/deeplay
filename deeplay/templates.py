import torch.nn as nn
from .config import Config
from .core import UninitializedModule

__all__ = [
    "Layer",
    "LayerInput",
    "OutputOf",
    "LayerSequence",
    "LayerAdd",
    "LayerSub",
    "LayerMul",
    "LayerDiv",
    "Template",
]


class Layer:
    def __init__(self, classname="", uid=None, **_):
        self.classname = classname
        self.uid = uid

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
        subconfig = config.with_selector(self.classname)

        module = UninitializedModule(subconfig)

        if self.uid is not None:
            config.add_ref(self.uid, module)

        return module

    def from_config(self, config):
        return self.build(config)


class LayerInput(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, config):
        return nn.Identity()


class InputOf(Layer):
    """A layer that references a specific module in the config."""

    def __init__(self, selector, **kwargs):
        super().__init__(**kwargs)
        self.selector = selector

    def build(self, config: Config):
        module = config.get_ref(self.selector)
        if module is None:
            raise ValueError(
                f"Module not found for selector {self.selector}. Make sure the module is created before it is referenced."
            )
        return RemoteModule(module, take_output=False)


class OutputOf(Layer):
    """A layer that references a specific module in the config."""

    def __init__(self, selector, **kwargs):
        super().__init__(**kwargs)
        self.selector = selector

    def build(self, config: Config):
        module = config.get_ref(self.selector)
        if module is None:
            raise ValueError(
                f"Module not found for selector {self.selector}. Make sure the module is created before it is referenced."
            )
        return RemoteModule(module, take_output=True)


class RemoteModule(nn.Module):
    """A module that references a module in another module."""

    def __init__(self, module: nn.Module, take_output=True):
        super().__init__()
        self.remote = module
        self.take_output = take_output

        self._x = None

        self._register_hooks()

    def _replace_uninitialized_remote(self):
        self.remote = self.remote.module()
        self.handle.remove()
        self._register_hooks()

    def _register_hooks(self):
        if self.take_output:
            self.handle = self.remote.register_forward_hook(self._take_output)
        else:
            self.handle = self.remote.register_forward_pre_hook(self._take_input)

    def _take_output(self, _, __, y):
        self._x = y

    def _take_input(self, _, x):
        self._x = x

    def forward(self, _):
        if isinstance(self.remote, UninitializedModule):
            self._replace_uninitialized_remote()

        return self._x


class LayerSequence(Layer):
    def __init__(self, *layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def __rshift__(self, other):
        return LayerSequence(*self.layers, other)

    def build(self, config):
        modules = {}
        for layer in self.layers:
            name = layer.classname or layer.__class__.__name__
            module = layer.build(config)
            idx = 0
            while name in modules:
                name = f"{layer.classname}({idx})"
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
