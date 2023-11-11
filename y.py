# %%
# To create an __init__ method that accepts both positional and keyword arguments like a dataclass,
# we can use the signature from the class annotations. Here is an example implementation:

import deeplay as v3
import inspect
import torch.nn as nn
from typing import overload, Literal, Type


layerlist: v3.LayerList[v3.LayerActivationBlock] = v3.LayerList()

for i in range(5):
    layerlist.append(
        v3.LayerActivationBlock(
            layer=v3.Layer(nn.Linear, 10, 10),
            activation=v3.Layer(nn.ReLU),
        )
    )

layerlist[2:4].layer.configure(nn.Tanh)
out = layerlist.create()


# %%
# class X:
#     @overload
#     def func(self, a: int, b: int, c: int):
#         pass

#     def func(self, **kwargs):
#         pass


# # %%
class MLP(v3.DeeplayModule):
    blocks: v3.LayerList[v3.LayerActivationNormalizationBlock]

    def __init__(self, in_features: int, hidden_features: list[int], out_features: int):
        blocks = v3.LayerList()

        for i, f_out in enumerate(hidden_features):
            f_in = in_features if i == 0 else hidden_features[i - 1]
            blocks.append(
                v3.LayerActivationNormalizationBlock(
                    layer=v3.Layer(nn.Linear, f_in, f_out),
                    activation=v3.Layer(nn.ReLU),
                    normalization=v3.Layer(nn.Identity, num_features=f_out),
                )
            )

        blocks.append(
            v3.LayerActivationNormalizationBlock(
                layer=v3.Layer(nn.Linear, hidden_features[-1], out_features),
                activation=v3.Layer(nn.Identity),
                normalization=v3.Layer(nn.Identity, num_features=out_features),
            )
        )
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


model = MLP(10, [20, 30, 40], 50)
# model.blocks.configure(slice(-1), "normalization", nn.BatchNorm1d)
model.build()
# # %%


# # %%
# def f(a, b, c=2, **kdwargs):
#     pass


# args = ("a",)
# kwargs = {"b": 2, "c": 3, "d": 2}

# sig = inspect.signature(f)
# parameters = sig.bind(*args, **kwargs)
# parameters.apply_defaults()
# parameters.arguments
# # %%

# import inspect


# def construct_arguments_dict(f, args, kwargs):
#     # Get the parameter names of the function
#     params = inspect.signature(f).parameters

#     # Initialize the arguments dictionary
#     arguments = {}

#     # Assign positional arguments to their respective parameter names
#     for param_name, arg in zip(params, args):
#         arguments[param_name] = arg

#     # Add/Override with keyword arguments
#     arguments.update(kwargs)

#     return arguments


# def example_function(a, b, c=3, **kwargs):
#     return a * b + c + sum(kwargs.values())


# args = (1, 2)
# kwargs = {"c": 4, "extra": 5}

# arguments = construct_arguments_dict(example_function, args, kwargs)
# print(arguments)
# print(example_function(*args, **kwargs))  # Using args and kwargs
# print(example_function(**arguments))  # Using constructed arguments dictionary
#
