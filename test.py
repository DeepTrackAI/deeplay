# %%

import torch

conv = torch.nn.Conv2d(3, 16, 3, padding=1)

# conv.in_channels = 2

# %%
def _create_module(module, **kwargs):
    import inspect

    # Get the arguments of the constructor.
    signature = inspect.signature(module.__init__)

    # Get the arguments that are in the signature.
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    
    # take rest of arguments from the original module
    for k, v in module.__dict__.items():
        if k in signature.parameters and k not in kwargs:
            kwargs[k] = v
    
    # Create a new module with the new arguments.
    new_module = module.__class__(**kwargs)
    return new_module

_create_module(conv, in_channels=2)

# %%
