from math import e
from typing import Any, Callable, Tuple, Union
from operator import add, sub, mul, truediv, pow, neg


def two_operation(op) -> Callable[["Variable", Union[int, "Variable"]], "Variable"]:
    def inner(self, y: Union[int, "Variable"]):

        if isinstance(y, int):
            return Variable(lambda z: op(self(z), y))
        else:
            return Variable(lambda z: op(self(z), y(z)))

    return inner


def reverse_two_operation(op) -> Callable[["Variable", int], "Variable"]:
    def inner(self, y: int) -> "Variable":
        return Variable(lambda z: op(y, self(z)))

    return inner


def unary_operation(op) -> Callable[["Variable"], "Variable"]:
    def inner(self) -> "Variable":
        return Variable(lambda z: op(self(z)))

    return inner


class Variable:
    """Represents a variable integer value that can be operated on.

    This class is used to represent a variable integer value that can be operated on.
    This is used inside shape expressions to represent the shape of the tensor that
    is not fully known.

    Parameters
    ----------
    func : Callable[[int], int], optional
        The function that operates on the variable, by default None.
        Should usually not be set.

    Returns
    -------
    int
        The result of the operation on the variable.

    Examples
    --------
    >>> x = Variable()
    >>> y = x + 1
    >>> y(1)
    2
    >>> y(2)
    3
    """

    def __init__(self, func=None) -> None:
        if func is not None:
            self.func = func
        else:
            self.func = lambda x: x

    def __call__(self, x: int) -> int:
        return x

    __add__ = two_operation(add)
    __radd__ = reverse_two_operation(add)
    __sub__ = two_operation(sub)
    __rsub__ = reverse_two_operation(sub)
    __mul__ = two_operation(mul)
    __rmul__ = reverse_two_operation(mul)
    __truediv__ = two_operation(truediv)
    __rtruediv__ = reverse_two_operation(truediv)
    __pow__ = two_operation(pow)
    __rpow__ = reverse_two_operation(pow)
    __neg__ = unary_operation(neg)


class Computed:

    def __init__(self, func: Callable[[Tuple[int, ...]], Any]) -> None:
        self.func = func

    def __call__(self, *args: Tuple[int, ...]) -> Any:
        return self.func(*args)
