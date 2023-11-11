from typing import Type, TypeVar

T = TypeVar("T")


class ExtendedConstructorMeta(type):
    _is_top_level = {"value": True}

    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        """Construct an instance of a class whose metaclass is Meta."""

        __user_config = kwargs.pop("__user_config", None)
        obj = cls.__new__(cls, *args, **kwargs)
        # if isinstance(obj, cls):
        cls.__pre_init__(obj, *args, **kwargs)

        if __user_config is not None:
            obj._user_config = __user_config

        if cls._is_top_level["value"]:
            cls._is_top_level["value"] = False
            obj.__construct__()
            obj.__post_init__()
            cls._is_top_level["value"] = True

        return obj


def not_top_level(cls: ExtendedConstructorMeta):
    current_value = cls._is_top_level["value"]

    class ContextManager:
        def __enter__(self):
            cls._is_top_level["value"] = False

        def __exit__(self, *args):
            cls._is_top_level["value"] = current_value

    return ContextManager()
