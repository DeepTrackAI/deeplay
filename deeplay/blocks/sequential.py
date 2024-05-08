import warnings

from torch import nn

from .block import Block

from typing import List, Optional, Union, overload, Any, Literal
from deeplay import DeeplayModule
from deeplay.external import Layer, External


class SequentialBlock(Block):
    def __init__(self, order: Optional[List[str]] = None, **kwargs: DeeplayModule):
        super().__init__()

        if order is None:
            order = list(kwargs.keys())

        self.order = []

        for name in order:
            if not name in kwargs:
                ...
                # warnings.warn(
                #     f"Block {self.__class__.__name__} does not have a module called `{name}`. "
                #     "You can provide it using `configure({name}=module)` or "
                #     "by passing it as a positional argument to the constructor."
                # )
            else:
                setattr(self, name, kwargs[name])
                self.order.append(name)

    def append(self, layer: DeeplayModule, name: Optional[str] = None):
        """Append a layer to the block, executing it after all the other layers.

        Parameters
        ----------
        layer : DeeplayLayer
            The layer to append.
        name : Optional[str], optional
            The name of the layer, by default None.
            If None, the name of the layer will be the lowercase of its class name.
        """
        name = self._create_name(layer, name)
        self.configure(**{name: layer}, order=self.order + [name])

    def prepend(self, layer: DeeplayModule, name: Optional[str] = None):
        """Prepend a layer to the block, executing it before all the other layers.

        Parameters
        ----------
        layer : DeeplayLayer
            The layer to prepend.
        name : Optional[str], optional
            The name of the layer, by default None.
            If None, the name of the layer will be the lowercase of its class name.
        """
        name = self._create_name(layer, name)
        self.configure(**{name: layer}, order=[name] + self.order)

    def insert(self, layer: DeeplayModule, after: str, name: Optional[str] = None):
        """Insert a layer to the block, executing it after a specific layer.

        Parameters
        ----------
        layer : DeeplayLayer
            The layer to insert.
        after : str
            The name of the layer after which the new layer will be executed.
        name : Optional[str], optional
            The name of the layer, by default None.

        Raises
        ------
        ValueError
            If the layer `after` is not found in the block.
        """

        name = self._create_name(layer, name)
        if after not in self.order:
            raise ValueError(f"Layer `{after}` not found in the block.")
        index = self.order.index(after) + 1
        self.configure(
            **{name: layer}, order=self.order[:index] + [name] + self.order[index:]
        )

    def remove(self, name: str, allow_missing: bool = False):
        """Remove a layer from the block.

        Parameters
        ----------
        name : str
            The name of the layer to remove.
        allow_missing : bool, optional
            Whether to raise an error if the layer is not found in the block, by default False.

        Raises
        ------
        ValueError
            If the layer `name` is not found in the block and `allow_missing` is False.
        """
        if name not in self.order:
            if not allow_missing:
                raise ValueError(f"Layer `{name}` not found in the block.")
            else:
                return

        self.configure(order=[n for n in self.order if n != name])

    def append_dropout(self, p: float, name: Optional[str] = "dropout"):
        """Append a dropout layer to the block.

        Parameters
        ----------
        p : float
            The dropout probability.
        name : Optional[str], optional
            The name of the dropout layer, by default "dropout".
        """
        self.append(Layer(nn.Dropout, p), name=name)

    def prepend_dropout(self, p: float, name: Optional[str] = "dropout"):
        """Prepend a dropout layer to the block.

        Parameters
        ----------
        p : float
            The dropout probability.
        name : Optional[str], optional
            The name of the dropout layer, by default "dropout".
        """
        self.prepend(Layer(nn.Dropout, p), name=name)

    def insert_dropout(self, p: float, after: str, name: Optional[str] = "dropout"):
        """Insert a dropout layer to the block.

        Parameters
        ----------
        p : float
            The dropout probability.
        after : str
            The name of the layer after which the dropout layer will be executed.
        name : Optional[str], optional
            The name of the dropout layer, by default "dropout".

        Raises
        ------
        ValueError
            If the layer `after` is not found in the block.
        """
        self.insert(Layer(nn.Dropout, p), after=after, name=name)

    def remove_dropout(self, name: str = "dropout", allow_missing: bool = False):
        """Remove a dropout layer from the block.

        Parameters
        ----------
        name : str, optional
            The name of the dropout layer to remove, by default "dropout".
        allow_missing : bool, optional
            Whether to raise an error if the dropout layer is not found in the block, by default False.

        Raises
        ------
        ValueError
            If the dropout layer `name` is not found in the block and `allow_missing` is False.
        """
        self.remove(name, allow_missing=allow_missing)

    def set_dropout(
        self,
        p: float,
        name: str = "dropout",
        on_missing: Literal["append", "prepend", "insert"] = "append",
        after: Optional[str] = None,
    ):
        """Set the dropout probability of a dropout layer.

        Parameters
        ----------
        p : float
            The dropout probability.
        name : str, optional
            The name of the dropout layer, by default "dropout".
        on_missing : str, optional
            The action to take if the dropout layer is not found in the block.
            If "append", a new dropout layer will be appended to the block.
            If "prepend", a new dropout layer will be prepended to the block.
            If "insert", a new dropout layer will be inserted after the layer specified in `after`.
            By default "append".
        after : str, optional
            The name of the layer after which the dropout layer will be executed if `on_missing` is "insert", by default None.

        """

        if on_missing != "insert" and after is not None:
            warnings.warn("`after` is only used when `on_missing` is 'insert'.")

        if name not in self.order:
            if on_missing == "append":
                self.append_dropout(p, name=name)
            elif on_missing == "prepend":
                self.prepend_dropout(p, name=name)
            elif on_missing == "insert":
                if after is None:
                    raise ValueError(
                        "You must specify the layer after which to insert the dropout layer."
                    )
                self.insert_dropout(p, after=after, name=name)
            else:
                raise ValueError(
                    f"Invalid value for `on_missing`. Expected 'append', 'prepend', or 'insert', got {on_missing}."
                )

        else:
            getattr(self, name).configure(p=p)

    def configure(self, *args, **kwargs):
        super().configure(*args, **kwargs)

    def forward(self, x):
        for name in self.order:
            x = getattr(self, name)(x)
        return x

    def _create_name(self, module: DeeplayModule, name: Optional[str] = None):

        name = self._create_name_from_module_if_name_is_none(module, name)

        if name in self.order:
            raise ValueError(
                f"Layer `{name}` already exists in the block. "
                "To change it, use .{name}.configure(...). "
                f"To execute the same layer multiple times, use .configure(order=[order with {name} multiple times])"
            )

        return name

    def _create_name_from_module_if_name_is_none(
        self, module: DeeplayModule, name: Optional[str]
    ):
        if name is not None:
            return name

        if isinstance(module, External):
            return module.classtype.__name__.lower()
        else:
            return module.__class__.__name__.lower()
