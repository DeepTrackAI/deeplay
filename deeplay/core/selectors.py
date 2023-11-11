import warnings

__all__ = [
    "NoneSelector",
    "Selector",
    "ClassSelector",
    "WildcardSelector",
    "DoubleWildcardSelector",
    "IndexSelector",
    "ParentalRelation",
    "Ref",
    "parse_selectors",
    "parse_selectors_from_string",
    "parse_selectors_from_tuple",
]


class Selector:
    def __add__(self, other):
        if isinstance(other, NoneSelector):
            return self
        if isinstance(other, Selector):
            return ParentalRelation(self, other)
        return NotImplemented

    def regex(self):
        exp = self._regex()
        return f"^{exp}$"

    def __radd__(self, other):
        if isinstance(other, NoneSelector):
            return self
        return NotImplemented

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return IndexSelector(self, index[0], index[1])
        return IndexSelector(self, index)

    def key(self):
        return str(self)

    def pop(self):
        return NoneSelector(), self


class NoneSelector(Selector):
    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __str__(self) -> str:
        return ""

    def _regex(self):
        return "(\\[\\d+\\])??"

    def key(self):
        return ""

    def __iter__(self):
        yield from [""]


class ClassSelector(Selector):
    def __init__(self, classname: str):
        self.classes = classname.split(" ")

    def __str__(self) -> str:
        if len(self.classes) == 1:
            return self.classes[0]

        return f"({' '.join(self.classes)})"

    def _regex(self):
        classes = [c + "(\\[\\d+\\])??" for c in self.classes]
        return f"({'|'.join(classes)})"

    def __iter__(self):
        yield from self.classes

    def key(self):
        if len(self.classes) > 1:
            warnings.warn(
                f"ClassSelector {self} has multiple classes. Using the first one as the key."
            )
        return self.classes[0]


class WildcardSelector(Selector):
    def __str__(self) -> str:
        return "*"

    def _regex(self):
        return "[a-zA-Z0-9_\\[\\]]+"

    def __iter__(self):
        raise NotImplementedError("WildcardSelector is not iterable")

    def key(self):
        raise NotImplementedError("WildcardSelector does not have a key")


class DoubleWildcardSelector(Selector):
    def __str__(self) -> str:
        return "**"

    def _regex(self):
        return ".*"

    def __iter__(self):
        raise NotImplementedError("DoubleWildcardSelector is not iterable")

    def key(self):
        raise NotImplementedError("DoubleWildcardSelector does not have a key")


class IndexSelector(Selector):
    def __init__(self, selector: Selector, index: int or slice, length=None):
        self.selector = selector
        self.index = index
        self.length = length

    def get_list_of_indices(self):
        if isinstance(self.index, int):
            return [self.index]

        stop = self.length or self.index.stop

        if stop is None:
            raise TypeError(
                f"IndexSelector with slice {self.index} must have a stop value"
            )

        if stop < 0 and self.length is None:
            raise TypeError(
                f"IndexSelector with slice {self.index} must have a length value to support negative indexing"
            )

        if stop < 0:
            stop = stop % self.length

        integers = [i for i in range(stop)][self.index]

        return integers

    def _regex(self):
        # length is the number of elements in the list being indexed

        integers = self.get_list_of_indices()
        integers = "|".join([str(i) for i in integers])
        return self.selector._regex() + f"\\[({integers})\\]"

    def __str__(self) -> str:
        return f"{self.selector}[{self.index}]"

    def __iter__(self):
        for selector in self.selector:
            for i in self.get_list_of_indices():
                yield f"{selector}[{i}]"

    def key(self):
        return self.selector.key()

    def pop(self):
        body, head = self.selector.pop()
        return body, IndexSelector(head, self.index, self.length)


class ParentalRelation(Selector):
    def __init__(self, parent: Selector, child: Selector):
        self.parent = parent
        self.child = child

    def __str__(self) -> str:
        return f"{self.parent}.{self.child}"

    def _regex(self):
        if isinstance(self.parent, DoubleWildcardSelector) or isinstance(
            self.child, DoubleWildcardSelector
        ):
            # If either the parent or child is a double wildcard, then the dot separator is optional.
            # This is to allow for edge cases like **.foo to match foo where no dot is present.
            return f"{self.parent._regex()}\\.?{self.child._regex()}"

        return f"{self.parent._regex()}\\.{self.child._regex()}"

    def __iter__(self):
        for parent in self.parent:
            for child in self.child:
                yield f"{parent}.{child}"

    def key(self):
        return self.child.key()

    def pop(self):
        if isinstance(self.child, ParentalRelation):
            return self.child.pop()
        return self.parent, self.child


class Ref:
    def __init__(self, selectors, func=None):
        """A reference to a Layer in the config tree."""
        self.func = func or (lambda x: x)
        self.selectors = parse_selectors(selectors)

    def __call__(self, x):
        return self.func(x)


def parse_selectors(x) -> Selector:
    """Parses a string into a Selector object."""
    if isinstance(x, (Selector, NoneSelector)):
        return x
    if isinstance(x, Ref):
        return x.selectors
    if isinstance(x, tuple):
        return parse_selectors_from_tuple(x)
    if isinstance(x, str):
        return parse_selectors_from_string(x)
    if isinstance(x, int):
        return IndexSelector(NoneSelector(), x)
    if x is None:
        return NoneSelector()
    raise TypeError(f"Cannot parse {x} as a selector")


def parse_selectors_from_string(x) -> Selector:
    # Very simple parser for now
    selectors = x.split(".")
    if len(selectors) == 1:
        if selectors[0] == "*":
            return WildcardSelector()
        if selectors[0] == "**":
            return DoubleWildcardSelector()
        return ClassSelector(x)
    return parse_selectors_from_tuple(selectors)


def parse_selectors_from_tuple(x) -> Selector:
    x = [parse_selectors(s) for s in x]

    if len(x) == 0:
        return NoneSelector()

    s = x[0]
    for _x in x[1:]:
        s = s + _x
    return s
