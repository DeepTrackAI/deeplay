from .groups import Group
from typing import List, Union, Tuple, Dict, Any, Callable, Optional

# Query is a string or a function that takes an element and an index and
# returns a boolean.
Query = Union[str, Callable[[Any, int], bool]]


class Selection(list):
    """A selection of elements.

    A selection is a list of groups. Each group is a list of elements that"""

    def __init__(self, groups: list = None):
        if groups is None:
            groups = []
        super().__init__(groups)

    def __getitem__(self, key):
        # Allow indexing into the groups.
        # Ensure that a sliced selection is still a selection.
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            return Selection(super().__getitem__(key))
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
        groups = []
        for group in self:
            new_group = []
            for element in group:
                new_element = element.select(query)
                new_group.append(new_element)

            groups.append(new_group)

        return Selection(groups)

    def select_all(self, query: Query) -> "Selection":
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
        groups = []
        for group in self:
            for element in group:
                new_group = element.select_all(query)
                groups.append(new_group)
        return Selection(groups)

    def set(self, **kwargs):
        """Sets the attributes of the elements in the selection.

        Parameters
        ----------
            kwargs: The attributes to set.

        """
        for group in self:
            for element in group:
                element.set(**kwargs)

    def filter(self, query: Query) -> "Selection":
        """Returns a new selection that matches the query.

        In case of multiple matches, only the first match is returned.

        Parameters
        ----------
            query: A query string.

        Returns
        -------
            A new selection.

        """
        groups = []
        for group in self:
            new_group = []
            for element in group:
                if element.matches(query):
                    new_group.append(element)
                new_element = element.filter(query)
                new_group.append(new_element)

            groups.append(new_group)

        return Selection(groups)
