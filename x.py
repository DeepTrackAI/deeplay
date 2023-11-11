# %%
from typing import (
    Any,
    Type,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    ParamSpec,
    Literal,
    Union,
    overload,
    Optional,
    Dict,
    Iterable,
    Iterator,
)

import typing
import torch.nn as nn
import inspect

from torch.nn.modules.container import ModuleList
from torch.nn.modules.module import Module
import deeplay as dl

from dataclasses import dataclass

T = TypeVar("T")


class module_list:
    def __init__(self, length: int | property, name="", on_build: Callable = None):
        self.length = length
        self.on_build = on_build
        self.name = name

    def __call__(self, func) -> "module_list":
        name = func.__name__

        def on_build(other):
            if isinstance(self.length, property):
                length = self.length.__get__(other, type(other))
            else:
                length = self.length

            for idx in range(length):
                template = func(other, idx)
                template_config = template.config
                other.config = other.config.__getattr__(name)[idx](
                    template, template_config
                )

        return type(self)(self.length, name, on_build)

    def __get__(self, instance, owner) -> list:
        self.on_build(instance)
        length = (
            self.length.__get__(instance, owner)
            if isinstance(self.length, property)
            else self.length
        )
        return [
            instance.config.__getattr__(self.name)[idx].get(None)
            for idx in range(length)
        ]


class Config(dl.Config):
    def module_list(self, length: int | property) -> module_list:
        return module_list(length)


class MetaclassConfigDeferrer(type):
    def __new__(cls, name, bases, attrs):
        # make all class attributes (that are not methods) configurable
        # by making the properties that return config values

        class_config = attrs.get("config", None) or Config()
        instance_config = Config(
            rules=class_config._rules.copy(),
            refs=class_config._refs.copy(),
            context=class_config._context,
        )

        new_attrs = {}

        for attr_name, attr_value in attrs.items():
            if attr_name == "__annotations__":
                for k, v in attr_value.items():
                    if not isinstance(v, Config):
                        new_attrs[k] = cls.create_property(k)

            if (
                not inspect.isfunction(attr_value)
                and not attr_name.startswith("_")
                and not isinstance(attr_value, Config)
            ):
                print(attr_name, attr_value)
                new_attrs[attr_name] = cls.create_property(attr_name)

            else:
                new_attrs[attr_name] = attr_value

        new_attrs["config"] = instance_config

        print(new_attrs)

        return super().__new__(cls, name, bases, new_attrs)

    @classmethod
    def create_property(cls, name):
        def getter(self):
            return self.config.get(name)

        def setter(self, value):
            self.config.set(name, value)

        return property(getter, setter)


from deeplay.core.utils import match_signature as _match_signature


class Base(nn.Module):
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

    def new(self, name: str, *args, **kwargs):
        return self.config.new(name, *args, **kwargs)

    def default(self, name: str, value: Any):
        self.class_config.default(name, value)

    def configure(self, name: str, value: Any = None, **kwargs: Any) -> None:
        """Configure a module."""

        if not value is None:
            self.user_config.set(name, value)

        for k, v in kwargs.items():
            print(k, v)
            self.user_config.set(name + "." + k, v)

        self.build()

    def build(self):
        for name, value in self.config.get_parameters(create=True).items():
            if isinstance(value, Base):
                value.user_config = getattr(self.user_config, name)
                value.build()
            print("setting", name, value)
            setattr(self, name, value)

    def update_user_config(self, new_config):
        old_config = self.user_config

        if old_config._rules is not new_config._rules:
            # if old config has any rules, we add them to the new config
            new_config.merge(old_config)
        self.user_config = new_config


class Layer(Base):
    def __init__(self, name: str, layer: Type[nn.Module], **kwargs):
        super().__init__()
        self.name = name
        self.default(name, layer)
        for k, v in kwargs.items():
            self.default(name + "." + k, v)

    def forward(self, x):
        return getattr(self, self.name)(x)

    @overload
    def configure(self, value: Type[nn.Module], **kwargs: Any) -> None:
        ...

    @overload
    def configure(self, **kwargs: Any) -> None:
        ...

    def configure(self, value: Any = None, **kwargs: Any) -> None:
        super().configure(self.name, value, **kwargs)


a = Layer("norm", nn.Identity)
a.build()
a.configure(nn.BatchNorm2d, num_features=10)
print(a)
# %%


class LayerList(Base, nn.ModuleList):
    def __init__(self, *layers: Layer | list[Layer]):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]

        for layer in layers:
            self.append(layer)

    def build(self):
        for idx, layer in enumerate(self):
            layer.update_user_config(self.user_config[idx])
            layer.build()

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


# %%
import time

llist = LayerList(Layer("norm", nn.Identity))
llist.append(Layer("test", nn.BatchNorm2d, num_features=10))
# llist.build()
llist[0].configure(nn.Linear, in_features=10, out_features=10)
llist[1].configure(eps=1e-3)
# llist.build()
print(llist, llist[0].user_config)

# %%


class Sequential(LayerList):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


# %%
