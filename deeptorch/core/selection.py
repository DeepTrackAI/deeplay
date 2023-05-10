from .groups import Group
from .queries import Query
from .template import Template
from typing import List, Union, Tuple, Dict, Any, Callable, Optional

# Query is a string or a function that takes an element and an index and
# returns a boolean.


class Selection(list):
    """A selection of elements.

    A selection is a list of groups. Each group is a list of elements that"""

    def __init__(self, module, groups: list = None):
        self.module = module
        if groups is None:
            groups = [("", module)]
        super().__init__(groups)

    def __getitem__(self, key):
        # Allow indexing into the groups.
        # Ensure that a sliced selection is still a selection.
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            return Selection(self.module, super().__getitem__(key))
        else:
            raise ValueError(f"Invalid key: {key}")

    def select(self, query: Query) -> "Selection":
        """Returns a new selection that matches the query.

        In case of multiple matches, only the first match is returned.

        Parameters
        ----------
            query: A query string.

        Returns
        -------
            A new selection.

        """
        query = Query(query)
        groups = []
        for name, module in self:
            new_group = []
            tree = module.named_modules()
            subtree = query.filter(tree)
            groups.extend(subtree)

        return Selection(self.module, groups)

    def select_first(self, query: Query) -> "Selection":
        """Returns a new selection that matches the query.

        In case of multiple matches, all matches are returned.

        Unlike `select`, the grouping is not preserved. For each element in
        the initial selection, a new group is created that contains all
        matches for that element.

        Parameters
        ----------
            query: A query string.

        Returns
        -------
            A new selection.

        """
        query = Query(query)

        groups = []
        for name, module in self:
            new_group = []
            tree = module.named_modules()
            subtree = query.filter(tree)
            groups.extend(subtree)
        return Selection(self.module, groups)

    def set(self, **kwargs):
        """Sets the attributes of the elements in the selection.

        Parameters
        ----------
            kwargs: The attributes to set.  

        """
        for name, module in self:     
            if isinstance(module, Template):
                module.set(**kwargs)
            else:
                new_module = self._create_module(module, **kwargs)
                self._set_module(name, new_module)

    @staticmethod
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
    
    def _set_module(self, name, module):
        """Sets the module at the given name."""
        # Get the parent module.
        parent_module = self.module
        parts = name.split(".")
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)

        # Set the module.
        setattr(parent_module, parts[-1], module)