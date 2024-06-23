from functools import wraps
import time
import datetime

from deeplay import meta


class Callback:
    """A deepcopy safe callback."""

    instance_count = 0

    def __init__(self, func, *args, **kwargs):
        self.id = Callback.instance_count
        Callback.instance_count += 1
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, instance):
        return self.func(instance, *self.args, **self.kwargs)

    def __hash__(self) -> int:
        return id(self)


def stateful(func):
    """Decorator for methods modify the object state."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.root_module._append_to_tape(self, func.__name__, args, kwargs)

        with self.root_module.calling_stateful():
            func(self, *args, **kwargs)
        return self

    return wrapper


def before_build(func):
    """Decorator for methods that will be run before build instead of immediately."""

    @stateful
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.register_before_build_hook(Callback(func, *args, **kwargs))
        return self

    return wrapper


def after_build(func):
    """Decorator for methods that will be run after build instead of immediately.

    If the build method creates a new object, the hook will run on the new object.
    """

    @stateful
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.register_after_build_hook(Callback(func, *args, **kwargs))
        return self

    return wrapper


def after_init(func):
    """Decorator for methods that will be run after init _and_ immediately.

    If called during init, the hook will not be stored and will only run immediately.
    """

    @stateful
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if not self.is_constructing:
            self.register_after_init_hook(Callback(func, *args, **kwargs))

        return self

    return wrapper
