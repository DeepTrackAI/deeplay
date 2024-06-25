from typing import Type, TypeVar
import dill

T = TypeVar("T")


class ExtendedConstructorMeta(type):
    _module_state = {
        "is_top_level": True,
        "current_root_module": None,
        "constructing_module": None,
    }

    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        """Construct an instance of a class whose metaclass is Meta."""

        # If the object is being constructed from a checkpoint, we instead
        # load the class from the pickled state and build it using the
        if "__from_ckpt_application" in kwargs:
            assert "__build_args" in kwargs, "Missing __build_args in kwargs"
            assert "__build_kwargs" in kwargs, "Missing __build_kwargs in kwargs"

            _args = kwargs.pop("__build_args")
            _kwargs = kwargs.pop("__build_kwargs")

            app = dill.loads(kwargs["__from_ckpt_application"])
            app.build(*_args, **_kwargs)
            return app

        # Otherwise, we construct the object as usual
        obj = cls.__new__(cls, *args, **kwargs)

        # We store the actual arguments used to construct the object
        object.__setattr__(
            obj,
            "_actual_init_args",
            {
                "args": args,
                "kwargs": kwargs,
            },
        )
        object.__setattr__(obj, "_config_tape", [])
        object.__setattr__(obj, "_is_calling_stateful_method", False)

        # First, we call the __pre_init__ method of the class
        cls.__pre_init__(obj, *args, **kwargs)

        # Next, we construct the class. The not_top_level context manager is used to
        # keep track of where in the object hierarchy we currently are.
        with not_top_level(cls, obj):
            obj.__construct__()
            obj.__post_init__()

        return obj


def not_top_level(cls: ExtendedConstructorMeta, obj):
    current_value = cls._module_state["is_top_level"]

    class ContextManager:
        """Context manager that sets the value of _module_state to False for the duration of the context.

        NOTE: "current_root_module" is not currently used. It was intended to be used to set the root_module
        of the current object to the root_module of the parent object.
        """

        def __init__(self, obj):
            self.original_module_state = cls._module_state["is_top_level"]
            self.original_current_root_module = cls._module_state["current_root_module"]
            self.original_constructing_module = cls._module_state["constructing_module"]
            self.obj = obj

        def __enter__(self):
            if self.original_module_state:
                cls._module_state["is_top_level"] = False
                cls._module_state["current_root_module"] = self.obj
            cls._module_state["constructing_module"] = self.obj

        def __exit__(self, *args):
            cls._module_state["is_top_level"] = self.original_module_state
            cls._module_state["current_root_module"] = self.original_current_root_module
            cls._module_state["constructing_module"] = self.original_constructing_module

    return ContextManager(obj)
