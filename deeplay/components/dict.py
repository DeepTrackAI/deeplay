"""Utility Modules for Dictionary and Graph Data Operations

This module contains utilities for operations involving dictionary-like
structures or PyTorch Geometric `Data` objects. These operations are useful in
geometric deep learning pipelines, where input data is often stored in a
structured format with various attributes.

Classes:
- FromDict: Extracts specified keys from a dictionary or `Data` object.
- AddDict: Performs element-wise addition for specified keys in two
  dictionaries or `Data` objects.
- CatDictElements: Concatenates specified elements within a dictionary or
  `Data` object along a given dimension.
"""

from typing import Dict, Any, Union, Tuple

from deeplay import DeeplayModule

import torch
from torch_geometric.data import Data


class FromDict(DeeplayModule):
    """Extract specified keys from a dictionary-like structure.

    Parameters
    ----------
    keys : str
        The keys to extract from the input dictionary.

    Returns
    -------
    Any or Tuple[Any, ...]
        The values corresponding to the specified keys.

    Example
    -------
    >>> extractor = FromDict("key1", "key2").create()
    >>> result = extractor({"key1": value1, "key2": value2})
    (value1, value2)
    """

    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(self, x: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
        return (
            x[self.keys[0]]
            if len(self.keys) == 1
            else tuple(x[key] for key in self.keys)
        )

    def extra_repr(self) -> str:
        return ", ".join(self.keys)


class AddDict(DeeplayModule):
    """Element-wise addition of two dictionaries.

    Parameters
    ----------
    keys : Tuple[str]
        Specifies the keys to be added element-wise.

    Constraints
    -----------
    - Both dictionaries `x` (base) and `y` (addition) must contain the same
      keys for the addition operation.

    - Input types:
        - `x`: Dict[str, Any] or `torch_geometric.data.Data`
        - `y`: Dict[str, Any] or `torch_geometric.data.Data`

    Example
    -------
    >>> adder = AddDict("key1", "key2").create()
    >>> result = adder({"key1": value1, "key2": value2},
                       {"key1": 1, "key2": 2})
    {"key1": value1 + 1, "key2": value2 + 2}
    """

    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(
        self, x: Union[Dict[str, Any], Data], y: Dict[str, Any]
    ) -> Union[Dict[str, Any], Data]:

        if isinstance(x, Data):
            x = x.clone()
        else:
            x = x.copy()

        x.update({key: torch.add(x[key], y[key]) for key in self.keys})
        return x


class CatDictElements(DeeplayModule):
    """Concatenates specified elements within a dictionary-like structure along
    a given dimension.

    Parameters
    ----------
    keys : Tuple[tuple]
        Specifies the keys to be concatenated as tuples. Each tuple contains
        two keys: source and target. The source key is the key to be
        concatenated with the target key.
    dim : int, optional
        Specifies the dimension along which the concatenation is performed.
        Default is -1.

    Example
    -------
    >>> concat = CatDictElements(("key1", "key2"), ("key3", "key4")).create()
    >>> result = concat({"key1": tensor1, "key2": tensor2,
                         "key3": tensor3, "key4": tensor4})
    {"key2": torch.cat([tensor2, tensor1], dim=-1),
     "key4": torch.cat([tensor4, tensor3], dim=-1)}
    """

    def __init__(self, *keys: Tuple[tuple], dim: int = -1):
        super().__init__()
        self.source, self.target = zip(*keys)
        self.dim = dim

    def forward(self, x: Union[Dict[str, Any], Data]) -> Union[Dict[str, Any], Data]:
        x = x.clone() if isinstance(x, Data) else x.copy()
        x.update(
            {
                t: torch.cat([x[t], x[s]], dim=self.dim)
                for t, s in zip(self.target, self.source)
            }
        )
        return x
