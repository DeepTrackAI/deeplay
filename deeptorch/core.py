import torch.nn as nn
import inspect
from .config import Config, NoneSelector, IndexSelector

from .utils import safe_call


def _match_signature(func, args, kwargs):
    """Returns a dictionary of arguments that match the signature of func.
    This can be used to find the names of arguments passed positionally.
    """
    sig = inspect.signature(func)
    # remove 'self' from the signature
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])

    # remove arguments in kwargs that are not in the signature
    for name in list(kwargs.keys()):
        if name not in sig.parameters:
            del kwargs[name]

    bound = sig.bind(*args, **kwargs)
    return bound.arguments


class DeepTorchModule(nn.Module):
    defaults = {}

    def __init__(self, **kwargs):
        super().__init__()
        self._all_uninitialized_submodules = []
        for name, module in self.named_modules():
            if isinstance(module, UninitializedModule):
                self._all_uninitialized_submodules.append(module)

        self._any_uninitialized_submodules = bool(self._all_uninitialized_submodules)

    def __new__(cls, *args, **kwargs):
        __init__args = _match_signature(cls.__init__, args, kwargs)
        config = cls._build_config(__init__args)

        obj = object.__new__(cls)
        obj.set_config(config)

        return obj

    def attr(self, key):
        """Get an attribute from the config."""
        return self.config.get(key)

    def create(self, key, i=None, length=None):
        """Create a module from the config."""
        subconfig = self.config.with_selector(key)
        if i is not None:
            subconfig = subconfig[i]

        return UninitializedModule(subconfig)

    def create_many(self, key, n):
        """Create many modules from the config."""
        return [self.create(key, i, length=n) for i in range(n)]

    def create_all(self, key):
        """Create all modules from the config."""
        subconfig: Config = self.config.with_selector(key)
        rules = subconfig._get_all_matching_rules(
            NoneSelector(), match_key=True, allow_indexed=True
        )
        indexes = set()

        for rule in rules:
            if isinstance(rule.head, IndexSelector):
                rule_indexes = rule.head.get_list_of_indices()
                if not isinstance(rule_indexes, list):
                    rule_indexes = [rule_indexes]
                indexes.update(rule_indexes)

        max_index = max(indexes) if indexes else 0

        return self.create_many(key, max_index + 1)

    def set_config(self, config: Config):
        self.config = config

    def __call__(self, *args, **kwargs):
        y = super().__call__(*args, **kwargs)
        self._replace_uninitialized_modules()
        return y

    def _replace_uninitialized_modules(self):
        if not self._any_uninitialized_submodules:
            return
        remaining = []
        for name, module in self._all_uninitialized_submodules.copy():
            # check that the module is still uninitialized
            if self._modules[name] is module:
                if module.is_initialized():
                    self._modules[name] = module.module()
                else:
                    remaining.append((name, module))

        self._all_uninitialized_submodules = remaining
        self._any_uninitialized_submodules = bool(remaining)

    @classmethod
    def from_config(cls, config):
        config = cls._add_defaults(config)

        obj = object.__new__(cls)
        obj.set_config(config)

        # if obj.__init__ has any required positional arguments, we need to pass them.
        __init__args = _match_signature(cls.__init__, [], config.get_parameters())
        obj.__init__(**__init__args)
        return obj

    @classmethod
    def _add_defaults(cls, config: Config):
        if isinstance(cls.defaults, dict):
            for key, value in cls.defaults.items():
                config.default(key, value)
        elif isinstance(cls.defaults, Config):
            # We set prepend to true to allow the caller to override the defaults.
            config.merge(NoneSelector(), cls.defaults, as_default=True, prepend=True)

        return config

    @classmethod
    def _build_config(cls, kwargs):
        # Should only be called from __new__.

        config = Config()
        for key, value in kwargs.items():
            if isinstance(value, Config):
                config.merge(key, value)
            else:
                config.set(key, value)

        config = cls._add_defaults(config)

        return config


class UninitializedModule(nn.Module):
    config: Config

    def __new__(cls, config: Config):
        if not config.has_forward_hooks():
            # If there are no forward hooks, we can immediately initialize the module.
            try:
                return cls.create_module(config)
            except RuntimeError:
                # Can happen if there are no immediate hooks, but indirect references to hooks.
                # In this case, we need to wait until the hooks are resolved.
                # TODO: make specific error to not catch all runtime errors.
                return super().__new__(cls)
        else:
            return super().__new__(cls)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._initialized_module = None

    def forward(self, x):
        if self._initialized_module is not None:
            return self._initialized_module(x)
        self.config.run_all_forward_hooks(x)
        self._initialized_module = self.create_module(self.config)
        return self._initialized_module(x)

    def is_initialized(self):
        return self._initialized_module is not None

    def module(self):
        return self._initialized_module

    @classmethod
    def create_module(cls, config: Config):
        """Create a module from the config."""

        template = config.get(NoneSelector())
        parameters = config.get_parameters()
        uid = parameters.get("uid", None)

        # uid is set in Layer

        res = cls.build_template(template, config)

        if uid is not None:
            config.add_ref(uid, res)

        return res

    @classmethod
    def build_template(cls, template, config):
        from .templates import Layer

        if isinstance(template, (list, tuple)):
            return [
                cls.build_template(template[i], config[i]) for i in range(len(template))
            ]
        elif isinstance(template, Layer):
            return template.from_config(config)
        elif inspect.isclass(template) and issubclass(template, DeepTorchModule):
            return template.from_config(config)
        elif isinstance(template, nn.Module):
            return template
        elif callable(template):
            return safe_call(template, config.get_parameters())
        else:
            return template

    # def _make_into(self, module):
    #     # Best effort into making this module into the given module.
    #     # This way, all references to this module will be replaced with the given module.
    #     self.__class__ = module.__class__
    #     self.__dict__ = module.__dict__
    #     # self.__module__ = module.__module__
    #     # self.__doc__ = module.__doc__
    #     # self.__annotations__ = module.__annotations__
    #     # self.__weakref__ = module.__weakref__
