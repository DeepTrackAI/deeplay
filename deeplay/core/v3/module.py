import inspect
from typing import Any, Dict, Tuple

import torch.nn as nn

from .meta import ExtendedConstructorMeta, not_top_level


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
        for name, value in self.named_children():
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
        params = self.get_signature().parameters

        arguments = {}

        # Assign positional arguments to their respective parameter names
        for param_name, arg in zip(params, args):
            arguments[param_name] = arg

        # Add/Override with keyword arguments
        arguments.update(kwargs)

        arguments.pop("self", None)
        return arguments

    def __construct__(self):
        with not_top_level(ExtendedConstructorMeta):
            self._modules.clear()
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
            print(name, value)
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
        _args = self._args
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
