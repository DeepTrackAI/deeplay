from typing import Type, TypeVar

T = TypeVar("T")


class ExtendedConstructorMeta(type):
    _is_top_level = {"value": True, "current_root_module": None}

    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        """Construct an instance of a class whose metaclass is Meta."""

        __user_config = kwargs.pop("__user_config", None)
        obj = cls.__new__(cls, *args, **kwargs)
        # if isinstance(obj, cls):
        cls.__pre_init__(obj, *args, **kwargs)

        # if cls._is_top_level["value"]:
        with not_top_level(cls, obj):
            obj.__construct__()
            obj.__post_init__()

        return obj


def not_top_level(cls: ExtendedConstructorMeta, obj):
    current_value = cls._is_top_level["value"]

    class ContextManager:
        """Context manager that sets the value of _is_top_level to False for the duration of the context.

        NOTE: "current_root_module" is not currently used. It was intended to be used to set the root_module
        of the current object to the root_module of the parent object.
        """

        def __init__(self, obj):
            self.original_is_top_level = cls._is_top_level["value"]
            self.original_current_root_module = cls._is_top_level["current_root_module"]
            self.obj = obj

        def __enter__(self):
            if self.original_is_top_level:
                cls._is_top_level["value"] = False
                cls._is_top_level["current_root_module"] = self.obj

        def __exit__(self, *args):
            cls._is_top_level["value"] = self.original_is_top_level
            cls._is_top_level["current_root_module"] = self.original_current_root_module

    return ContextManager(obj)
