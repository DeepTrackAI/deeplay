from torch import Tensor
import torch.nn as nn
import inspect
from typing import Any, Union

from torch.nn.modules.module import Module
from .config import Config, NoneSelector, IndexSelector

from .utils import safe_call, match_signature as _match_signature

__all__ = ["DeeplayModule", "UninitializedModule"]


class UninitializedModule(nn.Module):
    config: Config

    def __new__(cls, config: Config, now=False):
        if now:
            return cls.create_module(config)

        try:
            return cls.create_module(config)
        except (ValueError, TypeError) as e:
            # Can happen if there are no immediate hooks, but indirect references to hooks.
            # In this case, we need to wait until the hooks are resolved.
            # TODO: make specific error to not catch all runtime errors.
            return super().__new__(cls)

    def __init__(self, config: Config, now=False):
        super().__init__()
        self.config = config
        self._initialized_module = None

    def forward(self, *x, **kwargs):
        if self._initialized_module is not None:
            return self._initialized_module(*x, **kwargs)
        self._initialized_module = self.create_module(self.config)
        return self._initialized_module(*x, **kwargs)

    def is_initialized(self):
        return self._initialized_module is not None

    def module(self):
        return self._initialized_module

    @classmethod
    def create_module(cls, config: Config):
        """Create a module from the config."""

        template = config.get(NoneSelector())
        uid = config.get("uid", None)
        res = cls.build_template(template, config)

        if uid is not None:
            config.add_ref(uid, res)

        return res

    @classmethod
    def build_template(cls, template, config: Config):
        return config.build_object(template)

    def __getattr__(self, name):
        # if not self.is_initialized():
        #     raise AttributeError(f"Uninitialized module has no attribute {name}")
        # if not hasattr(self._initialized_module, name):
        #     raise AttributeError(
        #         f"Neither {self} nor {self._initialized_module} has attribute {name}"
        #     )
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._initialized_module, name)

    def __repr__(self):
        if self._initialized_module is not None:
            return repr(self._initialized_module)
        return super().__repr__()


class DeeplayModule(nn.Module):
    defaults = {}

    config: Config
    _extra_attributes: dict

    def __init__(self, **kwargs):
        super().__init__()
        self._all_uninitialized_submodules = []
        self._any_uninitialized_submodules = False
        self._deeplay_forward_hooks = self.config.get_all_forward_hooks()

    def __new__(cls, *args, **kwargs):
        __init__args = _match_signature(cls.__init__, args, kwargs)
        config = cls._build_config(__init__args)

        obj = object.__new__(cls)
        obj.set_config(config)
        obj._extra_attributes = {}

        return obj

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, UninitializedModule):
            self._all_uninitialized_submodules.append((name, value))
            self._any_uninitialized_submodules = True
        return super().__setattr__(name, value)

    def attr(self, key) -> Any:
        """Get an attribute from the config."""

        value = self.config.get(key)
        if not isinstance(value, nn.Module):
            # If the module is not a nn.Module, we need to add it to the extra attributes
            # so that it is properly printed.
            self._extra_attributes[key] = value
        return value

    def new(
        self, key, i=None, length=None, now=False, extra_kwargs=None
    ) -> UninitializedModule or "DeeplayModule":
        """Create a module from the config."""
        subconfig = self.config.with_selector(key)
        if i is not None:
            subconfig = subconfig[i]

        for k, v in (extra_kwargs or {}).items():
            subconfig.set(k, v)

        lazy = UninitializedModule(subconfig, now=now)
        if now and isinstance(lazy, UninitializedModule):
            raise RuntimeError(
                f"Cannot create module {key} now, because it has forward hooks."
            )

        if not isinstance(lazy, nn.Module):
            # If the module is not a nn.Module, we need to add it to the extra attributes
            # so that it is properly printed.
            self._extra_attributes[key] = lazy
        return lazy

    def set_config(self, config: Config):
        self.config = config

    def __call__(self, *args, **kwargs):
        for hook in self._deeplay_forward_hooks:
            hook.value(self, *args, **kwargs)

        y = super().__call__(*args, **kwargs)
        # TODO: we could consider dynamically replacing the __call__ overload to avoid
        # the overhead of this check.
        # Should be benchmarked.
        self._replace_uninitialized_modules()
        return y

    def extra_repr(self) -> str:
        # also wrint extra attributes
        extra = "\n".join(
            f"({k}): {v}" for k, v in self._extra_attributes.items() if k != "config"
        )
        return extra

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
        obj._extra_attributes = {}

        # if obj.__init__ has any required positional arguments, we need to pass them.
        _factory_kwargs = _match_signature(
            cls.__init__, [], config.get_parameters(create=False)
        )
        obj.__init__(**_factory_kwargs)
        return obj

    @classmethod
    def _add_defaults(cls, config: Config):
        if isinstance(cls.defaults, dict):
            for key, value in cls.defaults.items():
                config.default(key, value)
        elif isinstance(cls.defaults, Config):
            # We set prepend to true to allow the caller to override the defaults.
            config.merge(NoneSelector(), cls.defaults, as_default=True, prepend=True)
        elif callable(cls.defaults):
            config.merge(NoneSelector(), cls.defaults(), as_default=True, prepend=True)

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

    # def _make_into(self, module):
    #     # Best effort into making this module into the given module.
    #     # This way, all references to this module will be replaced with the given module.
    #     self.__class__ = module.__class__
    #     self.__dict__ = module.__dict__
    #     # self.__module__ = module.__module__
    #     # self.__doc__ = module.__doc__
    #     # self.__annotations__ = module.__annotations__
    #     # self.__weakref__ = module.__weakref__
