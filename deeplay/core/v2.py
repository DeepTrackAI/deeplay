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
)

from torch import Tensor, nn
from torch.nn.modules.module import Module

from deeplay.core.config import Config


class _DeeplayModuleBase(nn.Module):
    ...


class CreateNewModuleOnInitMeta(type):
    def __new__(cls, name, bases, attrs):
        # make all class attributes (that are not methods) configurable
        # by making the properties that return config values

        new_attrs_dict = {"_class_modules": {}}

        for attr_name, attr in attrs.items():
            if isinstance(attr, _DeeplayModuleBase) and not attr_name.startswith("_"):
                new_attrs_dict["_class_modules"][attr_name] = attr
            else:
                new_attrs_dict[attr_name] = attr

        return super().__new__(cls, name, bases, new_attrs_dict)

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        obj.before_build()
        return obj


class DeeplayModule(_DeeplayModuleBase, metaclass=CreateNewModuleOnInitMeta):
    __builder_functions__ = []

    # class_config: Config
    @property
    def user_config(self):
        return self._user_config

    @property
    def class_config(self):
        return self._class_config

    @property
    def config(self):
        user_config_copy = Config(
            rules=self.user_config._rules.copy(),
            refs=self.user_config._refs.copy(),
            context=self.user_config._context,
        )
        merged = user_config_copy.merge(self.class_config, as_default=True)
        merged._context = user_config_copy._context
        return merged

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._class_config = Config()
        obj._user_config = Config()
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()
        # mimic dataclass behaviour

        annotations = get_type_hints(self)
        annotations.pop("user_config", None)
        annotations.pop("class_config", None)

        for key, value in self._class_modules.items():
            print("setting", key, value)
            value = value.new()
            for method_name in value.__builder_functions__:
                # We need to bind the method to the instance
                # so that it can access the instance attributes
                func = getattr(value, method_name)
                if func is not None:
                    # Not all builders need to be defined.
                    print("binding", method_name, func)
                    setattr(value, method_name, functools.partial(func, self))

            setattr(self, key, value)

        # create a signature from the annotations
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in annotations.keys()
            ]
        )
        # bind the passed in arguments to the signature
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"Error while initializing {type(self)}. Make sure all class attributes have type annotations"
            ) from e

        bound.apply_defaults()

        # set the attributes on the instance
        for name, value in bound.arguments.items():
            # print("setting", name, value)
            setattr(self, name, value)

    def configure(self, *args: Any, **kwargs: Any) -> None:
        """Configure a module."""
        self.configure_without_build(*args, **kwargs)
        self.build()

    def configure_without_build(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 0:
            name: str = args[0]

            if hasattr(self, name) and isinstance(getattr(self, name), DeeplayModule):
                getattr(self, name).configure_without_build(*args[1:], **kwargs)
            else:
                for k, v in kwargs.items():
                    # TODO: Check if this is a valid config name
                    self.user_config.set(name + "." + k, v)

            if len(args) == 2:
                self._configure_key_val(name, args[1])
        else:
            for k, v in kwargs.items():
                self._configure_key_val(k, v)

    def default(self, *args, **kwargs) -> None:
        if len(args) > 0:
            name: str | int | None = args[0]

            if isinstance(name, int):
                if len(args) == 2 and kwargs == {}:
                    self.class_config[name].set(None, args[1])
                    return
                else:
                    raise NotImplementedError(
                        "Integer defaults only supported with a single value."
                    )

            if (
                not name is None
                and hasattr(self, name)
                and isinstance(getattr(self, name), DeeplayModule)
            ):
                getattr(self, name).default(*args[1:], **kwargs)
            else:
                for k, v in kwargs.items():
                    # TODO: Check if this is a valid config name
                    self.class_config.set(name + "." + k, v)

            if len(args) == 2:
                self._default_key_val(name, args[1])
        else:
            for k, v in kwargs.items():
                self._default_key_val(k, v)

    def _configure_key_val(self, name, value):
        # print("configure", name, value)
        self.assert_is_valid_config_name(name)
        self.user_config.set(name, value)

    def _default_key_val(self, name, value):
        # print("default", name, value)
        self.assert_is_valid_config_name(name)
        self.class_config.set(name, value)

    def before_build(self):
        pass

    def build(self) -> "DeeplayModule":
        self.before_build()

        for name, value in self.config.get_parameters(create=True).items():
            if isinstance(value, DeeplayModule):
                value.update_user_config(getattr(self.user_config, name))
                value.build()
            setattr(self, name, value)

        for name, value in self.named_children():
            if isinstance(value, DeeplayModule):
                value.update_user_config(getattr(self.user_config, name))
                value.build()

        return self

    def update_user_config(self, new_config: Config):
        old_config = self.user_config

        if old_config._rules is not new_config._rules:
            # if old config has any rules, we add them to the new config
            new_config.merge(old_config)
        self._user_config = new_config

    def assert_is_valid_config_name(self, name: str):
        none_object = object()

        if name in self.__annotations__:
            return

        if name in self.__dict__:
            return

        if self.class_config.get(name, default=none_object) is not none_object:
            return

        if name in self._modules and isinstance(self._modules[name], DeeplayModule):
            return

        raise ValueError(f"Invalid config name {name}. ")

    def new(self):
        obj = type(self).__new__(type(self))
        class_config = Config(
            rules=self.class_config._rules.copy(),
            refs=self.class_config._refs.copy(),
            # Class config context should be empty
            # context=self.class_config._context,
        )
        obj._class_config = class_config
        DeeplayModule.__init__(obj)

        return obj

    def notify(self, parent):
        # By default we don't store reverse lineage information.
        # We only need to do this in very specific cases.
        pass

    # def __setattr__(self, name: str, value: Any) -> None:
    #     if isinstance(value, DeeplayModule):
    #         if name in self.__annotations__:
    #             # Here we must bypass the Module

    #     super().__setattr__(name, value)


class NamedModuleMixin(DeeplayModule):
    name: str

    @DeeplayModule.user_config.getter
    def user_config(self):
        return getattr(self._user_config, self.name)

    @DeeplayModule.class_config.getter
    def class_config(self):
        return getattr(self._class_config, self.name)

    @DeeplayModule.config.getter
    def config(self):
        user_config_copy = Config(
            rules=self._user_config._rules.copy(),
            refs=self._user_config._refs.copy(),
            context=self._user_config._context,
        )
        merged = user_config_copy.merge(self._class_config, as_default=True)
        merged._context = user_config_copy._context
        return getattr(merged, self.name)

    def update_user_config(self, new_config):
        new_config_context = new_config._context

        base, head = new_config_context.pop()
        assert (
            str(head) == self.name
        ), f"Wrong config context. Expected {self.name} got {head}."

        recontextualized_config = Config(
            rules=new_config._rules,
            refs=new_config._refs,
            context=base,
        )

        # Do normal update
        super().update_user_config(recontextualized_config)


class Layer(DeeplayModule):
    _: nn.Module

    def __init__(self, layer: Type[nn.Module], **kwargs):
        super().__init__()
        self.default(None, layer)
        for k, v in kwargs.items():
            self.default(k, v)

    def forward(self, x):
        return self._(x)

    def build(self):
        print(self.config)
        template = self.config.get(None)
        layer = self.config.build_object(template)
        self._ = layer
        return self

    @overload
    def configure(self, value: Type[nn.Module], **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    configure = DeeplayModule.configure

    def configure_without_build(self, value: Any = None, **kwargs: Any) -> None:
        if value is not None:
            self._configure_key_val(None, value)

        return super().configure_without_build(**kwargs)

    def assert_is_valid_config_name(self, name: str):
        if name is None:
            return

        self._assert_valid_module_argument(name)

    def _assert_valid_module_argument(self, name: str):
        module_type = self.config.get(None)
        signature = inspect.signature(module_type.__init__)
        if name in signature.parameters:
            return

        raise ValueError(f"Invalid module argument {name} for module {module_type}. ")

    # def __getattr__(self, name: str) -> Tensor | Module:
    #     if name == "_":
    #         return super().__getattr__(name)

    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self._, name)


class layerlist(NamedModuleMixin, nn.ModuleList, property):
    __builder_functions__ = ["builderfunc", "lengthfunc"]

    def __init__(self, builderfunc, lengthfunc: Optional[Callable[[], int]] = None):
        self.name = builderfunc.__name__
        super().__init__()
        self.builderfunc = builderfunc
        self.lengthfunc = lengthfunc

    def new(self):
        obj = super().new()
        obj.builderfunc = self.builderfunc
        obj.lengthfunc = self.lengthfunc
        obj.name = self.name
        return obj

    def length(self, func):
        lengthfuncname = func.__name__

        assert (
            lengthfuncname == self.builderfunc.__name__
        ), f"The name of the length function must be the same as the builder function for layerlist. Got {lengthfuncname} and {self.builderfunc.__name__}."
        # TODO: not sure if we shoudl set lengthfunc here. Might interfere with subclassing
        self.lengthfunc = func
        return type(self)(self.builderfunc, func)

    # def before_build(self):

    def build(self):
        if self.lengthfunc is None:
            self._raise_length_error()

        for idx in range(len(self)):
            res = self.builderfunc(idx)
            self.default(idx, res)

        items = []
        config = self.config
        for idx in range(len(self)):
            print(idx)
            res = config[idx].get(None)
            value = config[idx].build_object(res)
            if isinstance(value, DeeplayModule):
                value.update_user_config(self.user_config[idx])
                value.build()
            items.append(value)

        # clear previous modules
        self._modules.clear()

        # Register the items as modules
        for idx, item in enumerate(items):
            self[idx] = item

        return self

    def _raise_length_error(self):
        raise ValueError(
            f"Cannot get length of layerlist {self.name}. You must specify a length function."
        )

    def __len__(self):
        # We defer to lengthfunc instead of len(self._modules) because
        # lengthfunc is correct before build is called.
        if self.lengthfunc is None:
            return self._raise_length_error()

        return self.lengthfunc()

    def __get__(self, obj, objtype) -> "layerlist":
        return super().__get__(obj, objtype)


class LayerList(DeeplayModule, nn.ModuleList):
    def __init__(self, *layers: Layer | list[Layer]):
        super().__init__()

    def build(self):
        for idx in range(len(self)):
            layer = self[idx]
            layer.update_user_config(self.user_config[idx])
            layer.build()
        return self

    @overload
    def __getitem__(self, idx: int) -> Layer:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "LayerList":
        ...

    def __getitem__(self, idx: int | slice) -> "Layer | LayerList":
        return super().__getitem__(idx)  # type: ignore

    def __iter__(self) -> Iterator[Layer]:
        return super().__iter__()


class Sequential(LayerList):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x
