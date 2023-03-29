from .selection import Selection


def selection_filter(func):
    """Decorator to register a query handler."""

    def wrapper(*args, **kwargs):
        def inner(selection):
            return [element for element in selection if func(element, *args, **kwargs)]

        return inner

    return wrapper


def selection_handler(func):
    """Decorator to register a query handler."""

    def wrapper(*args, **kwargs):
        def inner(selection):
            return func(selection, *args, **kwargs)

        return inner

    return wrapper


@selection_filter
def filter_match_id(element, id):
    return element.has_id(id)


@selection_filter
def filter_match_tag(element, tag):
    return element.has_tag(tag)


@selection_filter
def filter_match_class(element, class_name):
    return element.has_class(class_name)


@selection_handler
def select_by_index(selection, index):
    return selection[index]


@selection_handler
def select_by_slice(selection, slice):
    return selection[slice]


class Query:
    def __init__(self, query):
        """Initializes the query."""
        self._query = self._parse_query(query)

    def select(self, selection: Selection) -> Selection:
        """Returns a list of elements that match the query."""

        groups = []
        for group in selection:
            for element in group:
                new_group = element.select(self)
                groups.append(new_group)
        return Selection(groups)

    def select_first(self, selection: Selection) -> Selection:
        """Returns a list of elements that match the query."""
        subset = self._query(selection)
        return Selection(subset[:1])

    def _parse_query(self, query):

        if isinstance(query, Query):
            return query._query
        elif isinstance(query, str):
            return self._parse_query_string(query)
        elif isinstance(query, int):
            return self._parse_query_int(query)
        elif isinstance(query, slice):
            return self._parse_query_slice(query)
        elif isinstance(query, list):
            return self._parse_query_list(query)
        elif callable(query):
            return self._parse_query_callable(query)
        else:
            raise ValueError(f"Invalid query: {query}")

    def _parse_query_string(self, query):
        """Parses a query string."""

        if query.startswith("#"):
            return filter_match_id(query[1:])
        elif query.startswith("."):
            return filter_match_class(query[1:])
        elif query.startswith("@"):
            return filter_match_tag(query[1:])
        else:
            return filter_match_tag(query)

    def _parse_query_int(self, query):
        """Parses an integer query."""
        return select_by_index(query)

    def _parse_query_slice(self, query):
        """Parses a slice query."""
        return select_by_slice(query)

    def _parse_query_list(self, query):
        """Parses a list query."""
        raise NotImplementedError()

    def _parse_query_callable(self, query):
        """Parses a callable query."""
        return selection_filter(query)
