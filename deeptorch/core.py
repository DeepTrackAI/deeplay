import torch.nn as nn
import inspect
from .config import Config

def _match_signature(func, args, kwargs):
    """Returns a dictionary of arguments that match the signature of func.
    This can be used to find the names of arguments passed positionally.
    """
    sig = inspect.signature(func)
    # remove 'self' from the signature
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])

    bound = sig.bind(*args, **kwargs)
    return bound.arguments

class DeepTorchModule(nn.Module):

    defaults = {}

    def __init__(self, **kwargs):
        super().__init__()
    
    def __new__(cls, *args, **kwargs):

        __init__args = _match_signature(cls.__init__, args, kwargs)
        config = cls._build_config(__init__args)
        
        obj = object.__new__(cls)
        obj.set_config(config)
        
        return obj

    def attr(self, key):
        """ Get an attribute from the config.
        """
        return self.config.get(key)
    
    def create(self, key):
        """ Create a module from the config.
        """
        subconfig = self.config.with_selector(key)
        template = self.config.get(key)
        return template.from_config(subconfig)
        
    def set_config(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config):
        config = cls._add_defaults(config)
        
        obj = object.__new__(cls)
        obj.set_config(config)
        obj.__init__()
        return obj
    
    @classmethod
    def _add_defaults(cls, config):
        for key, value in cls.defaults.items():
            config.default(key, value)
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
