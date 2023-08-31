__all__ = ["selector_matches"]


def selector_matches(selector, query):
    if len(query) == 0:
        return True

    if len(selector) == 0:
        return len(query) == 0

    # return true if all elements of query are in selector and are in the same order
    # return false otherwise

    for query_item in query:
        if query_item not in selector:
            return False

        first_index = selector.index(query_item)
        selector = selector[first_index + 1 :]

    return True
