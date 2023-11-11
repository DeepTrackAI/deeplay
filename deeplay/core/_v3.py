import functools
import inspect
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Type,
    Union,
    get_type_hints,
    overload,
    Dict,
    Tuple,
    TypeVar,
    ParamSpec,
)

from torch import nn
from torch.nn.modules.module import Module
from deeplay.core.config import Config

T = TypeVar("T")


class ExtendedConstructorMeta(type):
    _is_top_level = {"value": True}

    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        """Construct an instance of a class whose metaclass is Meta."""

        __user_config = kwargs.pop("__user_config", None)
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
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


class DeeplayModule(nn.Module, metaclass=ExtendedConstructorMeta):
    _user_config: Dict[Tuple[str, ...], Any]
    _is_constructing: bool = False
    _is_building: bool = False

    __extra_configurables__: list[str] = []

    _args: tuple
    _kwargs: dict
    _actual_init_args: dict
    _has_built: bool
    _setattr_recording: set[str]

    @property
    def configurables(self):
        argspec = self.get_argspec()
        argset = {*argspec.args, *argspec.kwonlyargs, *self.__extra_configurables__}
        # TODO: should we do this?
        for name, value in self.named_modules():
            if isinstance(value, DeeplayModule):
                argset.add(name)

        return argset

    @property
    def kwargs(self):
        kwdict = self._kwargs.copy()
        for key, value in self._user_config.items():
            if len(key) == 1:
                kwdict[key[0]] = value

        return kwdict

    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = value

    def __pre_init__(self, *args, _args=(), **kwargs):
        super().__init__()

        self._actual_init_args = {
            "args": args,
            "_args": _args,
            "kwargs": kwargs,
        }
        print(kwargs)

        self._kwargs = self.build_arguments_from(*args, **kwargs)

        # Arguments provided as args are not configurable (though they themselves can be).
        self._args = _args
        self._is_constructing = False
        self._is_building = False
        self._user_config = {}
        self._has_built = False

        self._setattr_recording = set()

    def __init__(self, *args, **kwargs):  # type: ignore
        # We don't want to call the super().__init__ here because it is called
        # in the __pre_init__ method.
        ...

    def __post_init__(self):
        ...

    def build_arguments_from(self, *args, **kwargs):
        signature = self.get_signature()
        arguments = signature.bind(*args, **kwargs)
        arguments = arguments.arguments
        arguments.pop("self", None)
        return arguments

    def __construct__(self):
        with not_top_level(ExtendedConstructorMeta):
            self._is_constructing = True
            self.__init__(*self._args, **self.kwargs)
            self._is_constructing = False
            self.__post_init__()

    # def __build__(self):
    #     self._is_constructing = True
    #     self.build()
    #     self._is_constructing = False

    def configure(self, *args: str, **kwargs):
        if len(args) == 0:
            self._configure_kwargs(kwargs)

        else:
            if args[0] not in self.configurables:
                raise ValueError(
                    f"Unknown configurable {args[0]} for {self.__class__.__name__}. Available configurables are {self.configurables}."
                )

            if hasattr(getattr(self, args[0]), "configure"):
                getattr(self, args[0]).configure(*args[1:], **kwargs)
            elif len(args) == 2 and kwargs == {}:
                self._configure_kwargs({args[0]: args[1]})

            else:
                raise ValueError(
                    f"Unknown configurable {args[0]} for {self.__class__.__name__}. Available configurables are {self.configurables}."
                )

    def _configure_kwargs(self, kwargs):
        for name, value in kwargs.items():
            if name not in self.configurables:
                raise ValueError(
                    f"Unknown configurable {name} for {self.__class__.__name__}. Available configurables are {self.configurables}."
                )
            self._user_config[(name,)] = value
        self.__construct__()

    def take_user_configuration(self, config):
        self._user_config = config.copy()
        self.__construct__()

    def get_user_configuration(self):
        return self._user_config

    def give_user_configuration(self, receiver: "DeeplayModule", name):
        if self._user_config is not None:
            sub_config = {}
            for key, value in self._user_config.items():
                if len(key) > 1 and key[0] == name:
                    sub_config[key[1:]] = value
            receiver.take_user_configuration(sub_config)

    def collect_user_configuration(self):
        config = self.get_user_configuration()
        for name, value in self.named_modules():
            if name == "":
                continue
            if isinstance(value, DeeplayModule):
                name_as_tuple = tuple(name.split("."))
                local_user_config = value.get_user_configuration()
                for key, value in local_user_config.items():
                    config[name_as_tuple + key] = value

        return config

    def __setattr__(self, name, value):
        if self._is_constructing:
            if isinstance(value, DeeplayModule):
                self.give_user_configuration(value, name)
                value.__construct__()
            self._setattr_recording.add(name)
        super().__setattr__(name, value)

    def create(self):
        obj = self.new()
        obj = obj.build()
        return obj

    def build(self):
        for name, value in self.named_children():
            if isinstance(value, DeeplayModule):
                value = value.build()
                if value is not None:
                    try:
                        setattr(self, name, value)
                    except TypeError:
                        # torch will complain if we try to set an attribute
                        # that is not a nn.Module.
                        # We circumvent this by setting the attribute using object.__setattr__
                        object.__setattr__(self, name, value)

        self._has_built = True
        return self

    def new(self):
        user_config = self.collect_user_configuration()
        args = self._actual_init_args["args"]
        _args = self._actual_init_args["_args"]
        kwargs = self._actual_init_args["kwargs"]
        obj = ExtendedConstructorMeta.__call__(
            type(self),
            *args,
            __user_config=user_config,
            _args=_args,
            **kwargs,
        )
        return obj

    @classmethod
    def get_argspec(cls):
        spec = inspect.getfullargspec(cls.__init__)
        spec.args.remove("self")
        return spec

    @classmethod
    def get_signature(cls):
        sig = inspect.signature(cls.__init__)
        # remove the first parameter
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])
        return sig


T = TypeVar("T")
P = ParamSpec("P")


class External(DeeplayModule):
    __extra_configurables__ = ["classtype"]

    @property
    def kwargs(self):
        full_kwargs = super().kwargs
        classtype = full_kwargs.pop("classtype")

        # Since the classtype can be configured by the user, we need to
        # remove kwargs that are not part of the classtype's signature.

        signature = self.get_signature()
        signature_args = signature.parameters.keys()
        kwargs = {}
        for key, value in full_kwargs.items():
            if key in signature_args:
                kwargs[key] = value

        kwargs["classtype"] = classtype
        return kwargs

    def __pre_init__(self, classtype: type, *args, **kwargs):
        # Hack
        self.classtype = classtype
        super().__pre_init__(*args, classtype=classtype, **kwargs)

    def __init__(self, classtype, *args, **kwargs):
        super().__init__()
        self.classtype = classtype

    def build(self) -> nn.Module:
        args = self.kwargs
        args.pop("classtype", None)
        return self.classtype(**args)

    create = build

    def get_argspec(self):
        classtype = self.classtype
        if inspect.isclass(classtype):
            argspec = inspect.getfullargspec(classtype.__init__)
            argspec.args.remove("self")
        else:
            argspec = inspect.getfullargspec(classtype)

        return argspec

    def get_signature(self):
        classtype = self.classtype
        if issubclass(classtype, DeeplayModule):
            return classtype.get_signature()
        return inspect.signature(classtype)

    def build_arguments_from(self, *args, classtype, **kwargs):
        kwargs = super().build_arguments_from(*args, **kwargs)
        kwargs["classtype"] = classtype
        return kwargs

    @overload
    def configure(self, classtype: Callable[P, Any], **kwargs: P.kwargs) -> None:
        ...

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    def configure(self, classtype: Optional[type] = None, **kwargs):
        if classtype is not None:
            super().configure(classtype=classtype)

        super().configure(**kwargs)

    def __repr__(self):
        classkwargs = ", ".join(
            f"{key}={value}" for key, value in self.kwargs.items() if key != "classtype"
        )
        return f"{self.__class__.__name__}[{self.classtype.__name__}]({classkwargs})"


class Layer(External):
    def __pre_init__(self, cls: Type[nn.Module], *args, **kwargs):
        super().__pre_init__(cls, *args, **kwargs)

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, classtype: Callable[P, nn.Module], **kwargs: P.kwargs) -> None:
        ...

    configure = External.configure


class LayerList(DeeplayModule):
    def __pre_init__(self, *layers):
        super().__pre_init__(_args=layers)

    def __init__(self, *layers):
        super().__init__()
