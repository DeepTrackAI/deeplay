import torch.nn as nn
import inspect
from .config import Config, ClassSelector, NoneSelector
from .templates import Node
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
    
    def create(self, key, i=None, length=None):
        """ Create a module from the config.
        """
        subconfig = self.config.with_selector(key)
        if i is not None:
            subconfig = subconfig[i]

        template = subconfig.get(NoneSelector())

        if isinstance(template, Node) or inspect.isclass(template) and issubclass(template, DeepTorchModule):
            return template.from_config(subconfig)
        elif isinstance(template, nn.Module):
            return template
        elif callable(template):
            return safe_call(template, subconfig.get_parameters())
        else:
            return template
            
    
    def create_many(self, key, n):
        """ Create many modules from the config.
        """

        return nn.ModuleList([self.create(key, i, length=n) for i in range(n)])
        
    def set_config(self, config: Config):
        self.config = config

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
