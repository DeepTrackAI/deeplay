from functools import wraps
import time
import datetime


class Callback:
    """A deepcopy safe callback."""

    def __init__(self, func, *args, **kwargs):
        self.timestamp = time.perf_counter_ns()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, instance):
        return self.func(instance, *self.args, **self.kwargs)

    def __hash__(self) -> int:
        return id(self)


def before_build(func):
    """Decorator for methods that will be run before build instead of immediately."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.register_before_build_hook(Callback(func, *args, **kwargs))
        return self

    return wrapper


def after_build(func):
    """Decorator for methods that will be run after build instead of immediately.

    If the build method creates a new object, the hook will run on the new object.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.register_after_build_hook(Callback(func, *args, **kwargs))
        return self

    return wrapper


def after_init(func):
    """Decorator for methods that will be run after init _and_ immediately.

    If called during init, the hook will not be stored and will only run immediately.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if not self.is_constructing:
            self.register_after_init_hook(Callback(func, *args, **kwargs))

        return self

    return wrapper
