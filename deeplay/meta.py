from typing import Type, TypeVar
import dill

T = TypeVar("T")


class ExtendedConstructorMeta(type):
    _is_top_level = {
        "value": True,
        "current_root_module": None,
        "constructing_module": None,
    }

    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        """Construct an instance of a class whose metaclass is Meta."""

        # If the object is being constructed from a checkpoint, we instead
        # load the class from the pickled state and build it using the
        #
        if "__from_ckpt_application" in kwargs:
            assert "__build_args" in kwargs, "Missing __build_args in kwargs"
            assert "__build_kwargs" in kwargs, "Missing __build_kwargs in kwargs"

            _args = kwargs.pop("__build_args")
            _kwargs = kwargs.pop("__build_kwargs")

            app = dill.loads(kwargs["__from_ckpt_application"])
            app.build(*_args, **_kwargs)
            return app

        __user_config = kwargs.pop("__user_config", None)
        obj = cls.__new__(cls, *args, **kwargs)
        # if isinstance(obj, cls):
        cls.__pre_init__(obj, *args, **kwargs)

        if __user_config:
            obj._base_user_config = __user_config

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
            self.original_constructing_module = cls._is_top_level["constructing_module"]
            self.obj = obj

        def __enter__(self):
            if self.original_is_top_level:
                cls._is_top_level["value"] = False
                cls._is_top_level["current_root_module"] = self.obj
            cls._is_top_level["constructing_module"] = self.obj

        def __exit__(self, *args):
            cls._is_top_level["value"] = self.original_is_top_level
            cls._is_top_level["current_root_module"] = self.original_current_root_module
            cls._is_top_level["constructing_module"] = self.original_constructing_module

    return ContextManager(obj)
