from ..templates import Layer
from ..core import DeepTorchModule
from ..config import Config, Ref

import torch
import torch.nn as nn


class Skip(DeepTorchModule):

    defaults = (
        Config()
        .inputs[0](None)
    )

    def __init__(self, inputs, func):
        """Skip module.
        
        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        func : callable
            Function to apply to the inputs.
        """

        super().__init__(inputs=inputs, func=func)

        self.func = self.attr("func")
        self.inputs = self.create_all("inputs")

    def forward(self, x):
        inputs = [inp(x) for inp in self.inputs]
        return self.func(*inputs)


class Concatenate(Skip):

    defaults = (
        Config()
        .merge(Skip.defaults)
        .dim(1)
    )

    def __init__(self, inputs, dim=1):
        """Concatenate module.
        
        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        dim : int
            Dimension to concatenate on.
            Default is 1.
        """

        super().__init__(inputs=inputs, dim=dim)

        self.dim = self.attr("dim")
        self.inputs = self.create_all("inputs")

    def forward(self, x):
        inputs = [inp(x) for inp in self.inputs]
        return torch.cat(inputs, dim=self.dim)
    

class Add(Skip):

    defaults = (
        Config()
        .merge(Skip.defaults)
    )

    def __init__(self, inputs):
        """Add module.
        
        Parameters
        ----------
        inputs : list of Layer
            List of inputs.
        """

        super().__init__(inputs=inputs)

        self.inputs = self.create_all("inputs")

    def forward(self, x):
        inputs = [inp(x) for inp in self.inputs]
        return sum(inputs)
    

