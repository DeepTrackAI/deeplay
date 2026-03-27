import inspect
from typing import Any, Dict, Tuple, List, Set, Literal, Optional, Callable, Union
from warnings import warn
import dill
from typing_extensions import Self

import torch
import torch.nn as nn
from torch_geometric.data import Data

import copy
import inspect
import numpy as np

from .meta import ExtendedConstructorMeta, not_top_level
from .decorators import after_init, after_build, stateful
from functools import partial


def builder(cls, args, kwargs):
    # Builds a class given the arguments and keyword arguments.
    # Used to support kwargs when pickling in __reduce__.
    obj = cls(*args, **kwargs)
    obj.__construct__()
    return obj


class ConfigItem:

    @property
    def source_depth(self):
        if self.source is None:
            return -1
        return min(len(tag) for tag in self.source)

    def __init__(self, source: Optional[List[Tuple[str, ...]]], value: Any):
        self.source = source
        self.value = value

    def prefix(self, tags: Tuple[str, ...]):
        if self.source is None:
            return ConfigItem(self.source, self.value)
        else:
            return ConfigItem([tags + tag for tag in self.source], self.value)

    def __repr__(self):
        return f"ConfigItem(source={self.source}, value={self.value})"


class DetachedConfigItem:
    @property
    def source_depth(self):
        if self.source is None:
            return -1
        return min(len(tag) for tag in self.source.tags)

    def __init__(
        self,
        source: "DeeplayModule",
        value: Any,
    ):
        self.source = source
        self.value = value

    def prefix(self, tags: Tuple[str, ...]):
        return DetachedConfigItem(self.source, self.value)

    def __repr__(self):
        return f"DetachedConfigItem(value={self.value})"


class ConfigItemList(List[Union[ConfigItem, DetachedConfigItem]]): ...


class Config(Dict[Tuple[str, ...], ConfigItemList]):

    __hook_containers__ = [
        ("__user_hooks__",),
    ]

    def __init__(self, *args, **kwargs):
        self.hooks = {}
        super().__init__(*args, **kwargs)

    def update(self, __m):
        for key, value in __m.items():
            if key[-1] == "__user_hooks__":
                if key not in self.hooks:
                    self.hooks[key] = value.copy()
                else:
                    for hook_name, hooks in value.items():
                        self.hooks[key][hook_name] += hooks
            else:
                self._set_or_extend(key, value)

    def prefix(self, tags: List[Tuple[str, ...]]):
        d = Config()
        for tag in tags:
            for key, value in self.items():
                d._set(tag + key, ConfigItemList([v.prefix(tag) for v in value]))
            for hook_key, hooks in self.hooks.items():
                d.hooks[tag + hook_key] = hooks
        return d

    def take(
        self,
        tags: List[Tuple[str, ...]],
        keep_list=False,
        take_subconfig=False,
        trim_tags=False,
        trim_derived=False,
    ):
        res = Config()

        def matches_key(key, tag):
            if not take_subconfig:
                return len(key) == len(tag) + 1 and key[: len(tag)] == tag

            return len(key) >= len(tag) and key[: len(tag)] == tag

        def maybe_trim_derived(key, tag, value: ConfigItemList):
            if not trim_derived:
                return value
            return [
                item
                for item in value
                if not isinstance(item, DetachedConfigItem)
                and item.source_depth <= len(tag)
            ]

        def new_key(key, tag):
            if not trim_tags:
                return key
            return key[len(tag) :]

        for tag in tags:
            res.update(
                {
                    new_key(key, tag): maybe_trim_derived(key, tag, value)
                    for key, value in self.items()
                    if matches_key(key, tag)
                }
            )
            res.update(
                {
                    new_key(key, tag): value
                    for key, value in self.hooks.items()
                    if matches_key(key, tag)
                }
            )
        out = {}

        # Sort list by source, take the last item with the source closest to the root.
        if not keep_list:
            for key, itemlist in res.items():

                if len(itemlist) == 0:
                    continue

                if all(isinstance(item, DetachedConfigItem) for item in itemlist):
                    out[key] = itemlist[-1].value
                    continue

                # itemlist = [
                #    item
                #    for item in itemlist
                #    if not isinstance(item, DetachedConfigItem)
                # ]

                # filter out DetachedConfigItem

                itemdepth = [item.source_depth for item in itemlist]
                min_depth = min(itemdepth)
                itemlist = [
                    item
                    for item, depth in zip(itemlist, itemdepth)
                    if depth == min_depth
                ]
                out[key] = itemlist[-1].value
        else:
            out = {**res}

        out.update(res.hooks)

        return out

    def set_for_tags(
        self,
        tags: List[Tuple[str, ...]],
        key: str,
        value: Any,
        source: Optional[List[Tuple[str, ...]]] = None,
    ):
        for tag in tags:
            self._set_or_append(tag + (key,), ConfigItem(source, value))

    def add_detached_configuration(
        self, source: "DeeplayModule", child: "DeeplayModule", name: str, value: Any
    ):
        for tag in child.tags:
            self._set_or_append(tag + (name,), DetachedConfigItem(source, value))

    def try_attach_detached_configurations(self, target: "DeeplayModule"):
        for itemlist in self.values():
            for idx, item in enumerate(itemlist):
                if isinstance(item, DetachedConfigItem):
                    # Same hierarchy if have same root.
                    if item.source.root_module is target.root_module:
                        newitem = ConfigItem(item.source.tags, item.value)
                        itemlist[idx] = newitem

    def remove_derived_configurations(
        self, tags: Optional[List[Tuple[str, ...]]] = None
    ):
        if tags is None:
            match_tag = lambda key: True
        else:
            match_tag = lambda key: any(tag in key for tag in tags)
            assert all(isinstance(tag, tuple) for tag in tags), (
                f"Tags must be a list of tuples, but found {tags}. "
                "Please check the tags being used."
            )

        for k, itemlist in self.items():
            for item in itemlist:
                if (
                    item.source is not None
                    and isinstance(item, ConfigItem)
                    and match_tag(item.source)
                ):
                    itemlist.remove(item)

    def _set(
        self,
        key: Tuple[str, ...],
        value: Union[ConfigItem, DetachedConfigItem, ConfigItemList],
    ):
        assert isinstance(value, (ConfigItem, DetachedConfigItem, ConfigItemList)), (
            f"Value must be a ConfigItem, DetachedConfigItem or ConfigItemList, but found {type(value)}. "
            "Please check the value being set."
        )
        if isinstance(value, (ConfigItem, DetachedConfigItem)):
            value = ConfigItemList([value])
        self[key] = value

    def _set_or_append(
        self, key: Tuple[str, ...], value: Union[ConfigItem, DetachedConfigItem]
    ):
        assert isinstance(value, (ConfigItem, DetachedConfigItem)), (
            f"Value must be a ConfigItem, but found {type(value)}. "
            "Please check the value being set."
        )
        if key in self:
            self[key].append(value)
        else:
            self._set(key, value)

    def _set_or_extend(self, key: Tuple[str, ...], value: ConfigItemList):
        assert isinstance(value, ConfigItemList), (
            f"Value must be a ConfigItemList, but found {type(value)}. "
            "Please check the value being set."
        )
        if key in self:
            self[key].extend(value)
        else:
            self._set(key, value)

    def __setitem__(self, key: Tuple[str, ...], value: ConfigItemList):
        assert isinstance(value, ConfigItemList), (
            f"Value must be a ConfigItemList, but found {type(value)}. "
            "Please check the value being set."
        )
        super().__setitem__(key, value)


# class DerivedConfig(UserConfig):
#     """Derived configuration for a module.

#     Derived configuration differs from user configuration in that it is
#     automatically generated by the module itself, rather than being set
#     by the user. Derived configuration is typically used to store internal
#     module settings, such as default values or calculated attributes.

#     The derived configuration is stored in the module that creates the
#     configuration, not the target module. This ensure the configuration
#     has the correct lifecycle. Even if the target module is re-initialized,
#     the derived configuration will remain the same since it is stored in
#     the creating module. If the the creating module is re-initialized, the
#     derived configuration will be re-calculated from scratch.

#     The relation between the creating module and the target module is
#     established by the `tags` attribute, which uses the hierarchical
#     structure of the module to identify the target module.

#     However, the target module might not be attached to the same hierarchy
#     at the time of configuration. Then, no relation can be established.
#     In this case, the derived configuration is stored in the target module's
#     root module temporarily. When the target module is attached to a new
#     hierarchy, the derived configuration is moved to the original creating
#     module.
#     """

#     def __init__(self, *args, **kwargs):
#         self._detached_configurations = []
#         super().__init__(*args, **kwargs)


# class TemporaryDerivedConfig(dict):
#     """Temporary derived configuration for a module.

#     Temporary derived configuration is used to store derived configuration
#     for a module that is not yet attached to the core hierarchy. This
#     configuration is stored in the root module of the module until the
#     module is attached to the hierarchy.
#     """


def _create_forward_with_input_dict(
    old_forward,
    input_args: List[str],
    input_kwargs: Dict[str, str],
    output_args: Optional[Dict[str, int]],
):
    def forward_with_input_dict(self, x, overwrite_output: bool = True):
        if isinstance(x, dict):
            x = x.copy()
        elif isinstance(x, Data):
            x = x.clone()
        else:
            raise ValueError(
                f"Input must be a dictionary or torch-geometric Data object, but found {type(x)}. Please check if the module require an input/output mapping."
            )

        outputs = old_forward(
            self,
            *map(x.get, input_args),
            **{key: x.get(value) for key, value in input_kwargs.items()},
        )

        if not output_args:
            return outputs

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        expected_outputs = len(set(output_args.values()))
        assert len(outputs) == expected_outputs, (
            f"module {type(self).__name__} returned {len(outputs)} outputs, "
            f"but it should return {expected_outputs}"
        )

        if overwrite_output:
            x.update(
                dict(
                    map(
                        lambda key, value: (key, outputs[value]),
                        *zip(*output_args.items()),
                    )
                )
            )
            return x
        else:
            return {key: outputs[value] for key, value in output_args.items()}

    return forward_with_input_dict


class DeeplayModule(nn.Module, metaclass=ExtendedConstructorMeta):
    """
    A base class for creating configurable, extensible modules with dynamic initialization
    and argument management. This class is designed to be subclassed for specific functional
    implementations. It extends `nn.Module` and utilizes a custom meta-class, `ExtendedConstructorMeta`,
    for enhanced construction logic.

    Attributes
    ----------
    __extra_configurables__ : list[str]
        List of additional configurable attributes.
    configurables : set[str]
        A property that returns a set of configurable attributes for the module.
    kwargs : dict
        A property to get or set keyword arguments.


    Methods
    -------

    configure(*args: Any, **kwargs: Any)
        Configures the module with given arguments and keyword arguments.

    create()
        This method creates and returns a new instance of the module.
        Unlike build, which modifies the module in place, create
        initializes a fresh instance with the current configuration and state,
        ensuring that the original module remains unchanged

    build()
        build: "This method modifies the current instance of the module in place.
        It finalizes the setup of the module, applying any necessary configurations
        or adjustments directly to the existing instance."

    new()
        Creates a new instance of the module with collected user configuration.

    get_user_configuration()
        Retrieves the current user configuration of the module.

    get_argspec()
        Class method to get the argument specification of the module's initializer.

    get_signature()
        Class method to get the signature of the module's initializer.

    Example Usage
    -------------
    To subclass `DeeplayModule`, define an `__init__` method and other necessary methods:

    ```
    class MyModule(DeeplayModule):
        def __init__(self, param1, param2):
            super().__init__()
            # Initialize your module here
    ```

    To use the module:

    ```
    module = MyModule(param1=value1, param2=value2)
    module.configure(param1=some_value, param2=some_other_value)
    built_module = module.build()
    ```
    """

    __extra_configurables__: List[str] = []
    __config_containers__: List[str] = ["_user_config"]
    __parent_hooks__: Dict[Literal["before_build", "after_build", "after_init"], list]
    __constructor_hooks__: Dict[
        Literal["before_build", "after_build", "after_init"], list
    ]
    _is_building: bool = False
    _init_method = "__init__"
    _style_map: Dict[str, Callable] = {}

    _args: tuple
    _kwargs: dict
    _actual_init_args: dict
    _has_built: bool
    _setattr_recording: Set[str]
    _tag: Tuple[str, ...]
    _config_tape: list
    _is_calling_stateful_method: bool

    logs: Dict[str, Any]

    @property
    def tags(self) -> List[Tuple[str, ...]]:
        if self.root_module is self:
            return [()]

        tags = [
            tuple(name.split("."))
            for name, module in self.root_module.named_modules(remove_duplicate=False)
            if module is self
        ]
        tags = [() if tag == ("",) else tag for tag in tags]
        if not tags:
            raise RuntimeError(
                f"Module {type(self)} is not a child of root module: {self.root_module}"
            )
        return tags

    @property
    def configurables(self) -> Set[str]:
        argspec = self.get_argspec()
        argset = {*argspec.args, *argspec.kwonlyargs, *self.__extra_configurables__}
        # TODO: should we do this?
        for name, value in self.named_children():
            if isinstance(value, DeeplayModule):
                argset.add(name)

        return argset

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwdict = self._kwargs.copy()

        # User config should always override derived config
        for key, value in self._user_config.take(self.tags).items():
            if key[-1] not in [
                "__parent_hooks__",
                "__constructor_hooks__",
                "__user_hooks__",
            ]:
                kwdict[key[-1]] = value

        return kwdict

    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = value

    @property
    def __hooks__(
        self,
    ) -> Dict[Literal["before_build", "after_build", "after_init"], list]:
        """A dictionary of all hooks.
        Ordered __constructor_hooks__ > __parent_hooks__ > __user_hooks__"""
        try:
            all_hooks = {
                k: v + self.__parent_hooks__[k] + self.__user_hooks__[k]
                for k, v in self.__constructor_hooks__.items()
            }

            for key, value in all_hooks.items():
                all_hooks[key] = sorted(value, key=lambda x: x.id)
            return all_hooks
        except AttributeError as e:
            raise RuntimeError("Module has not been initialized properly") from e

    @property
    def __user_hooks__(
        self,
    ) -> Dict[Literal["before_build", "after_build", "after_init"], list]:
        """A dictionary of all user hooks.

        User hooks are hooks added after the creation of the root module.
        """
        user_config = self._user_config
        tags = self.tags
        for tag in tags:
            if tag + ("__user_hooks__",) in user_config.hooks:
                return user_config.hooks[tag + ("__user_hooks__",)]
        # not found, add
        __user_hooks__: Dict[
            Literal["before_build", "after_build", "after_init"], list
        ] = {
            "before_build": [],
            "after_build": [],
            "after_init": [],
        }
        for tag in tags:
            user_config.hooks[tag + ("__user_hooks__",)] = __user_hooks__

        return __user_hooks__

    @property
    def __active_hooks__(
        self,
    ) -> Dict[Literal["before_build", "after_build", "after_init"], list]:
        """Selects the dict of hooks that is currently relevant.

        If after the root module has been created, __user_hooks__ is returned.
        If inside the constructor, __constructor_hooks__ is returned.
        Else, __parent_hooks__ is returned.
        """
        if ExtendedConstructorMeta._module_state["is_top_level"]:
            return self.__user_hooks__
        if self.is_constructing:
            return self.__constructor_hooks__
        else:
            return self.__parent_hooks__

    @property
    def is_constructing(self) -> bool:
        return self._is_constructing if hasattr(self, "_is_constructing") else False

    @is_constructing.setter
    def is_constructing(self, value):
        self._is_constructing = value

    @property
    def root_module(self) -> "DeeplayModule":
        return self._root_module[0]

    @property
    def logs(self):
        if self.root_module is self:
            return self._logs
        return self.root_module.logs

    def set_root_module(self, value):
        self._root_module = (value,)

        for name, module in self.named_modules():
            module._root_module = (value,)
            if isinstance(module, DeeplayModule):
                module._user_config.try_attach_detached_configurations(module)
            # self._user_config.try_attach_detached_configurations(module)

    @property
    def _user_config(self) -> Config:
        if self.root_module is None:
            return self._base_user_config
        if self.root_module is self:
            return self._base_user_config

        return self.root_module._user_config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def __pre_init__(self, *args, _args=(), **kwargs):
        super().__init__()
        # Stored as tuple to avoid it being included in modules
        self._root_module = (self,)

        self._actual_init_args["_args"] = _args

        self._base_user_config = Config()

        self.__parent_hooks__ = {
            "before_build": [],
            "after_build": [],
            "after_init": [],
        }
        self.__constructor_hooks__ = {
            "before_build": [],
            "after_build": [],
            "after_init": [],
        }

        self._kwargs, variadic_args = self._build_arguments_from(*args, **kwargs)

        # Arguments provided as args are not configurable (though they themselves can be).
        self._args = _args + variadic_args
        self.is_constructing = False
        self._is_building = False

        self._has_built = False
        self._setattr_recording = set()

        self._computed_values: Dict[str, Callable] = {}

        self._logs = {}
        if hasattr(self, "validate_after_build"):
            self._validate_after_build()

    def __init__(self, *args, **kwargs):  # type: ignore
        # We don't want to call the super().__init__ here because it is called
        # in the __pre_init__ method.
        ...

    def __post_init__(self): ...

    @after_init
    def set_input_map(self, *args: str, **kwargs: str):
        self.__dict__.update(
            {"input_args": args, "input_kwargs": kwargs, "_input_mapped": True}
        )

    @after_init
    def set_output_map(self, *args: str, **kwargs: int):
        output_args = {arg: i for i, arg in enumerate(args)}
        output_args.update(kwargs)

        self.__dict__.update({"output_args": output_args, "_output_mapped": True})

    def _execute_mapping_if_valid(self):
        if getattr(self, "_input_mapped", False) and getattr(
            self, "_output_mapped", False
        ):
            self._set_mapping(
                self, self.input_args, self.input_kwargs, self.output_args
            )

    @staticmethod
    def _set_mapping(
        module,
        input_args: List[str],
        input_kwargs: Dict[str, str],
        output_args: Dict[str, int],
    ):
        # monkey patch the forward method to include dict
        # using type(module) to get the base implementation of forward.
        # This is necessary so that multiple calls to set_input_dict don't
        # chain the monkey patching.
        # We use partial to bind the instance to make it a method.
        module.forward = partial(
            _create_forward_with_input_dict(
                type(module).forward, input_args, input_kwargs, output_args
            ),
            module,
        )

    @after_init
    def replace(self, target: str, replacement: nn.Module):
        """
        Replaces a child module with another module.

        This method replaces the child module with the given name with the specified replacement module.
        It is useful for dynamically swapping out modules within a larger module or for replacing
        modules within a module that has already been built.

        Parameters
        ----------
        target : str
            The name of the child module to be replaced.
        replacement : DeeplayModule
            The replacement module.

        Raises
        ------
        ValueError
            Raised if the target module is not found among the module's children.

        Example Usage
        -------------
        To replace a child module with another module:
        ```
        module = ExampleModule()
        module.replace('child_module', ReplacementModule())
        ```
        """
        if target not in self._modules:
            raise ValueError(
                f"Cannot replace {target}. {target} is not a child module of {self.__class__.__name__}."
            )

        self._modules[target] = replacement

    @after_init
    def schedule(self, **schedulers):
        for attr, scheduler in schedulers.items():
            setattr(self, attr, scheduler)

    @after_init
    def schedule_linear(
        self,
        attr: str,
        start_value: float,
        end_value: float,
        n_steps: int,
        on_epoch: bool = False,
    ):
        from deeplay.schedulers import LinearScheduler

        setattr(self, attr, LinearScheduler(start_value, end_value, n_steps, on_epoch))

    @after_init
    def schedule_loglinear(
        self,
        attr: str,
        start_value: float,
        end_value: float,
        n_steps: int,
        on_epoch: bool = False,
    ):
        from deeplay.schedulers import LogLinearScheduler

        setattr(
            self, attr, LogLinearScheduler(start_value, end_value, n_steps, on_epoch)
        )

    @stateful
    def configure(self, *args: Any, **kwargs: Any):
        """
        Configures the module with specified arguments.

        This method allows dynamic configuration of the module's properties and behaviors. It can be
        used to set or modify the attributes and parameters of the module and, if applicable, its child
        modules. The method intelligently handles both direct attribute configuration and delegation to
        child modules' `configure` methods.

        Parameters
        ----------
        *args : Any
            Positional arguments specifying the configuration settings. When the first argument is a
            string matching a configurable attribute, the method expects either one or two arguments:
            the attribute name and, optionally, its value. If the attribute is itself a `DeeplayModule`,
            subsequent arguments are passed to its `configure` method.

        **kwargs : Any
            Keyword arguments for configuration settings. If provided, these are used to update the
            module's configuration directly.

        Raises
        ------
        ValueError
            Raised if a configuration key is not recognized as a valid configurable for the module or
            if the provided arguments do not match the expected pattern for configuration.

        Example Usage
        -------------
        To configure a single attribute:
        ```
        module.configure('attribute_name', attribute_value) # or
        module.configure(attribute_name=attribute_value)
        ```

        To configure multiple attributes using keyword arguments:
        ```
        module.configure(attribute1=value1, attribute2=value2)
        ```

        To configure a child module's attribute:
        ```
        module.configure('child_module_attribute', child_attribute=child_attribute_value) # or
        module.child_module.configure(child_attribute=child_attribute_value)
        ```
        """

        if self._has_built:
            raise RuntimeError(
                "Module has already been built. "
                "Please use create() to create a new instance of the module."
            )

        if len(args) == 0:
            self._configure_kwargs(kwargs)
        else:
            self._assert_valid_configurable(args[0])

            if hasattr(getattr(self, args[0]), "configure"):
                getattr(self, args[0]).configure(*args[1:], **kwargs)
            elif len(args) == 2 and not kwargs:
                self._configure_kwargs({args[0]: args[1]})

            else:
                raise ValueError(
                    f"Unknown configurable {args[0]} for {self.__class__.__name__}. "
                    "Available configurables are {self.configurables}."
                )
        return self

    def create(self) -> "Self":
        """
        Creates and returns a new instance of the module, fully initialized
        with the current configuration.

        This method differs from `build` in that it generates a new,
        independent instance of the module, rather than modifying the
        existing one. It's particularly relevant for subclasses of
        `dl.External`, where actual torch layers (like Linear, Sigmoid, ReLU,
        etc.) are instantiated during the build process. For these subclasses,
        `create` not only configures but also instantiates the specified torch
        layers. For most other objects, the `.build()` step, which is internally
        called in `create`, has no additional effect beyond configuration.

        Returns
        -------
        DeeplayModule
            A new instance of the `DeeplayModule` (or its subclass), initialized
            with the current module's configuration and, for `dl.External` subclasses,
            with instantiated torch layers.

        Example Usage
        -------------
        Creating a `dl.Layer` instance and then fully initializing it with `create`:
        ```
        layer = dl.Layer(nn.Linear, in_features=20, out_features=40)
        # At this point, `layer` has not instantiated nn.Linear
        built_layer = layer.create()
        # Now, `built_layer` is an instance of nn.Linear(in_features=20, out_features=40)
        ```
        """

        if self._has_built:
            warn(
                "Module has already been built. "
                "Please only use one of `build` or `create`.",
                RuntimeWarning,
            )
            return self

        obj = self.new()
        obj.build()
        return obj

    @stateful
    def build(self, *args, **kwargs) -> "DeeplayModule":
        """
        Modifies the current instance of the module in place, finalizing its setup.

        The `build` method is essential for completing the initialization of the module. It applies
        the necessary configurations and adjustments directly to the existing instance. Unlike `create`,
        which generates a new module instance, `build` works on the current module instance. This method
        is particularly crucial for subclasses of `dl.External`, as it triggers the instantiation of
        actual torch layers (like Linear, Sigmoid, ReLU, etc.) within the module. For most other objects,
        `build` primarily serves to finalize their configuration.

        Note that `build` is automatically called within the `create` method, ensuring that newly created
        instances are fully initialized and ready for use.

        Parameters
        ----------
        *args, **kwargs : Any
            Example input to the forward method used to


        Returns
        -------
        DeeplayModule
            The current instance of the module after applying all configurations and adjustments.

        Example Usage
        -------------
        Finalizing the setup of a module instance with `build`:
        ```
        module = ExampleModule(a=0)
        module.configure(a=1)
        built_module = module.build()
        # `built_module` is the same instance as `module`, now fully configured and initialized
        ```
        """

        from .external import Optimizer

        if args or kwargs:
            #
            with torch.no_grad():
                self(*args, **kwargs)

        self._run_hooks("before_build")

        for name, value in self.named_children():
            if isinstance(value, Optimizer):
                ...  # skip optimizers
            elif isinstance(value, DeeplayModule):
                if value._has_built:
                    continue
                if value is self:
                    # this is a recursive call to build
                    # should not happend
                    continue
                value = value.build()

                if value is not None:

                    try:
                        setattr(self, name, value)

                    except TypeError:

                        # torch will complain if we try to set an attribute
                        # that is not a nn.Module.
                        # We circumvent this by setting the attribute using object.__setattr__
                        object.__setattr__(self, name, value)

        self._execute_mapping_if_valid()
        self._has_built = True
        self._run_hooks("after_build")

        return self

    # @property
    # def __deepcopy__(self):
    #     if self._has_built:
    #         return lambda _: self
    #     else:
    #         return None

    def new(self, detach: bool = True, memo=None) -> Self:

        s = dill.dumps(self)
        obj = dill.loads(s)

        return obj

    def predict(
        self, x, *args, batch_size=32, device=None, output_device=None
    ) -> torch.Tensor:
        """
        Predicts the output of the module for the given input.

        This method is a wrapper around the `forward` method, which is used to predict the output of the module
        for the given input. It is particularly useful for making predictions on large datasets, as it allows
        for the specification of a batch size for processing the input data.

        Parameters
        ----------
        x : array-like
            The input data for which to predict the output. Should be an array-like object with the same length
            as the input data.
        *args : Any
            Positional arguments for the input data. Should have the same length as the input data.
        batch_size : int, optional
            The batch size for processing the input data. Defaults to 32.
        device : str, Device, optional
            The device on which to perform the prediction. If None, the model's device is used.
            Defaults to None.
        output_device : str, Device, optional
            The device on which to store the output. If None, the model's device is used.
            Defaults to None.

        Returns
        -------
        Any
            The output of the module for the given input data.

        Example Usage
        -------------
        To predict the output of a module for the given input data:
        ```
        module = ExampleModule()
        output = module.predict(input_data, batch_size=64)
        ```
        """
        if args:
            for arg in args:
                assert len(arg) == len(x), "All inputs must have the same length."

        if len(x) >= 1000:
            if isinstance(x, torch.Tensor) and x.device.type == "mps":
                device = x.device
                x = x.cpu().to(device)

            args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == "mps":
                    arg = arg.cpu().to(device)
                args.append(arg)
            args = tuple(args)
            # for _x in (x,) + args:
            #     if isinstance(_x, torch.Tensor) and _x.device.type == "mps":
            #         _x.to("cpu").to("mps")

        if device is None:
            device = self.device
        if output_device is None:
            output_device = device

        output_containers = []
        with torch.no_grad():
            for idx_0 in range(0, len(x), batch_size):
                idx_1 = min(idx_0 + batch_size, len(x))
                batch = [item[idx_0:idx_1] for item in [x, *args]]

                for i, item in enumerate(batch):
                    if not isinstance(item, torch.Tensor):
                        if isinstance(item, np.ndarray):
                            batch[i] = torch.from_numpy(item)
                        else:
                            batch[i] = torch.stack(item)
                    else:
                        batch[i] = item

                    if batch[i].dtype.is_floating_point:
                        if hasattr(self, "dtype"):
                            batch[i] = batch[i].to(self.dtype)
                        else:
                            batch[i] = batch[i].float()

                    batch[i] = batch[i].to(device)

                # ensure that all inputs are tuples
                res = self.forward(*batch)
                if not isinstance(res, tuple):
                    res = (res,)
                for i, item in enumerate(res):
                    output_container = (
                        output_containers[i] if i < len(output_containers) else None
                    )
                    if output_container is None:
                        output_container = torch.empty(
                            (len(x), *item.shape[1:]), device=output_device
                        )
                        output_containers.append(output_container)
                    output_container[idx_0:idx_1] = item.to(output_device)

        if len(output_containers) == 1:
            return output_containers[0]
        return tuple(output_containers)

    @classmethod
    def available_styles(cls):
        return list(cls._style_map.keys())

    def style(self, style: str, *args, **kwargs) -> Self:
        if style not in self._style_map:
            raise ValueError(
                f"Style {style} not found. Available styles are {self.available_styles()}"
            )
        self._style_map[style](self, *args, **kwargs)
        return self

    styled = style

    @classmethod
    def register_style(cls, func):
        cls._style_map = cls._style_map.copy()
        cls._style_map[func.__name__] = func
        return func

    @after_build
    def log_output(self, name: str):
        root = self._root_module[0]

        def forward_hook(module, input, output):
            root.logs[name] = output

        self.register_forward_hook(forward_hook)

    @after_build
    def log_input(self, name: str):
        root = self._root_module[0]

        def forward_hook(module, input):
            root.logs[name] = input

        self.register_forward_pre_hook(forward_hook)

    def log_tensor(self, name: str, tensor: torch.Tensor):
        """Stores a tensor in the root log.

        Allows for storing tensors in the root module's logs. Can be used in
        inference to make tensors globally available outside the direct
        forward pass.

        Parameters
        ----------
        name : str
            The name of the tensor to store.
        tensor : torch.Tensor
            The tensor to store in the logs.
        """
        self.logs[name] = tensor

    def initialize(
        self, initializer, tensors: Union[str, Tuple[str, ...]] = ("weight", "bias")
    ):
        if isinstance(tensors, str):
            tensors = (tensors,)
        for module in self.modules():
            if isinstance(module, DeeplayModule):
                module._initialize_after_build(initializer, tensors)
            else:
                initializer.initialize(module, tensors)

    @after_build
    def _initialize_after_build(self, initializer, tensors: Tuple[str, ...]):
        initializer.initialize(self, tensors)

    @after_build
    def _validate_after_build(self):
        if hasattr(self, "validate_after_build"):
            return self.validate_after_build()

    def register_before_build_hook(self, func):
        """
        Registers a function to be called before the module is built.

        Parameters
        ----------
        func : Callable
            The function to be called before the module is built. The function should take
            a single argument, which is the module instance.

        """
        self.__active_hooks__["before_build"].append(func)

    def register_after_build_hook(self, func):
        """
        Registers a function to be called after the module is built.

        Parameters
        ----------
        func : Callable
            The function to be called after the module is built. The function should take
            a single argument, which is the module instance.

        """

        self.__active_hooks__["after_build"].append(func)

    def register_after_init_hook(self, func):
        """
        Registers a function to be called after the module is initialized.

        Parameters
        ----------
        func : Callable
            The function to be called after the module is initialized. The function should take
            a single argument, which is the module instance.

        """
        self.__active_hooks__["after_init"].append(func)

    def _register_hook(self, hook_type, func):
        if ExtendedConstructorMeta._module_state["is_top_level"]:
            self.__active_hooks__[hook_type].append(ConfigItem(None, func))
        else:
            # If we are not currently constructing the top level module,
            # this means that this is a derived configuration.
            # Thus, we store the source of the derived configuration such
            # that it can be cleared at the correct time.

            current_constructing_module: DeeplayModule = (
                ExtendedConstructorMeta._module_state["constructing_module"]
            )
            self.__active_hooks__[hook_type].append(
                ConfigItem(current_constructing_module.tags, func)
            )

    def get_user_configuration(self):
        """
        Retrieves the current user configuration of the module.

        This method returns a dictionary containing the configuration settings provided by the user
        for this module. It is useful for inspecting the current state of module configuration,
        especially after multiple configurations have been applied or when the module's configuration
        needs to be examined or debugged.

        Returns
        -------
        dict
            A dictionary containing the current user configuration settings. The keys are tuples
            representing the configuration attributes, and the values are the corresponding settings.

        Example Usage
        -------------
        To retrieve the current configuration of a module:
        ```
        module = ExampleModule(a=0)
        module.configure(a=1)
        current_config = module.get_user_configuration()
        # current_config == {('a',): 1}
        ```
        """
        return self._user_config

    def get_from_user_config(self, key):
        v = self._user_config.take(self.tags)
        v = [value for k, value in v.items() if k[-1] == key]
        if not v:
            v = self._base_derived_config.take(self.tags)
            v = [value for k, value in v.items() if k[-1] == key]

        return v[-1]

    def get_subconfig(self):
        tags = self.tags
        return self._user_config.take(tags, keep_list=True, take_subconfig=True)

    def _configure_kwargs(self, kwargs):
        for name, value in kwargs.items():
            self._assert_valid_configurable(name)
            if ExtendedConstructorMeta._module_state["is_top_level"]:
                self._user_config.set_for_tags(self.tags, name, value)
            else:
                # If we are not currently constructing the top level module,
                # this means that this is a derived configuration.
                # Thus, we store the source of the derived configuration such
                # that it can be cleared at the correct time.

                current_constructing_module: DeeplayModule = (
                    ExtendedConstructorMeta._module_state["constructing_module"]
                )

                # check if self is a child of the constructing module
                # If it is, we only need to store the tags of the constructing module
                # and the tags of the target module.
                tags = [
                    tuple(name.split("."))
                    for name, module in current_constructing_module.named_modules(
                        remove_duplicate=False
                    )
                    if module is self
                ]
                tags = [() if tag == ("",) else tag for tag in tags]

                if tags:
                    self._user_config.set_for_tags(
                        self.tags, name, value, source=current_constructing_module.tags
                    )
                else:
                    # If self is not a child of the constructing module, we need to store
                    # the modules themselves. We will try to attach the derived configuration
                    # to the correct module when the module is attached to the correct hierarchy.

                    self._user_config.add_detached_configuration(
                        current_constructing_module, self, name, value
                    )
        self.__construct__()

    def _give_user_configuration(self, receiver: "DeeplayModule", name) -> bool:
        if receiver._user_config is self._user_config:
            if receiver.root_module is not receiver:
                receiver.set_root_module(self.root_module)
                return True
            return False
        if receiver.root_module is self.root_module:
            return False

        mytags = self.tags
        try:
            receivertags = receiver.tags
        except RuntimeError:
            raise ValueError(
                f"Receiver named ({mytags}) . {name}  is not a child of the root module of the sender."
            )

        # sort longest tag first
        receivertags.sort(key=lambda x: len(x), reverse=True)

        d = {}
        for _, model in self.named_modules():
            if isinstance(model, DeeplayModule):
                tags = model.tags
                subconfig = receiver._user_config.take(tags, keep_list=True)
                for key, value in subconfig.items():
                    for tag in receivertags:
                        if len(key) >= len(tag) and key[: len(tag)] == tag:
                            key = key[len(tag) :]
                            break

                    for tag in mytags:
                        d[tag + (name,) + key] = value

                    for citem in value:
                        if isinstance(citem, ConfigItem) and citem.source is not None:
                            citem.source = [
                                tag + (name,) + key
                                for key in citem.source
                                for tag in mytags
                            ]

        config_before = self._user_config.take(receivertags, take_subconfig=True)
        is_empty = len(config_before) == 0
        self._user_config.update(d)
        receiver.set_root_module(self.root_module)
        return not is_empty

    def __setattr__(self, name, value):
        if name == "_user_config" and hasattr(self, "_user_config"):
            if not isinstance(value, Config):
                raise ValueError("User configuration must be a Config instance.")
            super().__setattr__(name, value)

        super().__setattr__(name, value)

        if (
            hasattr(self, "is_constructing")
            and hasattr(self, "_root_module")
            and hasattr(self, "_has_built")
            and hasattr(self, "_is_calling_stateful_method")
            and hasattr(self, "_default_attribute_keys")
            and not self.is_constructing
            and not self._is_calling_stateful_method
            and name not in ("is_constructing",)
            and name not in self._default_attribute_keys
        ):
            if (
                isinstance(value, DeeplayModule)
                and value.root_module is not self.root_module
                and value.root_module is not value
            ):
                # This means that we are setting a submodule of another module
                # to this module.
                self._append_to_tape(
                    self, "_set_submodule", (name, value.root_module, value.tags), {}
                )
            else:
                self._append_to_tape(self, "__setattr__", (name, value), {})

        if self.is_constructing:
            if isinstance(value, DeeplayModule):
                # if not value._has_built:
                needs_rebuild = self._give_user_configuration(value, name)

                if needs_rebuild and not value._has_built:
                    value.__construct__()
                    # # root module should always be update to
                    # # ensure that logs are stored in the correct place
                    # value.set_root_module(self.root_module)

    def __getattr__(self, name):
        from deeplay.schedulers import BaseScheduler

        x = super().__getattr__(name)
        if self._has_built and isinstance(x, BaseScheduler):
            return x.__get__(self, type(self))
        return x

    @stateful
    def _set_submodule(self, name, module, tags):

        for _, submodule in module.named_modules():
            if hasattr(submodule, "tags") and submodule.tags == tags:
                m = submodule
                break

        if m is None:
            raise ValueError(f"Submodule with tags {tags} not found.")

        setattr(self, name, m)

    @stateful
    def set(self, name, value):
        """Sets an attribute for the module, allowing for persistent configuration
        across checkpoints.

        This method is intended for setting module parameters or configurations
        that should be restored when loading a module from a saved checkpoint.

        Parameters
        ----------
           name: str
               The name of the attribute to set.
           value: any
               The value to assign to the attribute.

        Example
        -------
        >>> module.set("attribute", 42)
        >>> module.attribute
        42

        """
        setattr(self, name, value)
        return self

    def _select_string(self, structure, selections, select, ellipsis=False):
        selects = select.split(",")
        selects = [select.strip() for select in selects]

        selects_and_slices = []

        for select in selects:
            slicer = slice(None)
            if "#" in select:
                select, slice_str = select.split("#")
                slice_str = slice_str.split(":")
                slice_ints = [int(item) if item else None for item in slice_str]

                if len(slice_str) == 1:
                    if slice_ints[0] is None:
                        slicer = slice(slice_ints[0], None)
                    elif slice_ints[0] == -1:
                        slicer = slice(-1, None)
                    else:
                        slicer = slice(slice_ints[0], slice_ints[0] + 1)
                elif len(slice_str) == 2:
                    slicer = slice(slice_ints[0], slice_ints[1])
                else:
                    slicer = slice(
                        slice_ints[0],
                        slice_ints[1],
                        slice_ints[2],
                    )
            selects_and_slices.append((select, slicer))

        new_selections = [[] for _ in range(len(selections))]
        for select, slicer in selects_and_slices:
            bar_selects = select.split("|")
            bar_selects = [bar_select.strip() for bar_select in bar_selects]

            for i, group in enumerate(selections):
                group = selections[i]
                new_group = []
                for item in group:
                    if ellipsis:
                        for name in structure:
                            if (
                                len(name) > len(item)
                                and name[-1] in bar_selects
                                and name[: len(item)] == item
                            ):
                                new_group.append(name)
                    else:
                        for bar_select in bar_selects:
                            new_select = item + (bar_select,)
                            if new_select in structure:
                                new_group.append(new_select)

                new_group = new_group[slicer]
                new_selections[i] += new_group

        for i, group in enumerate(new_selections):
            selections[i] = group

    def getitem_with_selections(self, selector, selections=None):
        if selections is None:
            selections = [[()]]

        names, _ = zip(*self.named_modules())

        structure = {}
        for name in names:
            if name == "":
                continue
            key = tuple(name.split("."))
            base = key[:-1]
            name = key[-1]

            if base not in structure:
                structure[base] = []

            structure[base].append(name)
            structure[key] = []

        if not isinstance(selector, tuple):
            selector = (selector,)

        selections = [[()]]
        idx = 0
        while idx < len(selector):
            # flatten selections
            new_selections = []
            for group in selections:
                new_selections += group
            selections = [[item] for item in new_selections]

            select = selector[idx]
            if isinstance(select, int):
                if select == -1:
                    select = slice(select, None)
                else:
                    select = slice(select, select + 1)
            if isinstance(select, str):
                self._select_string(structure, selections, select)
            if isinstance(select, type(...)):
                if idx + 1 < len(selector):
                    if not isinstance(selector[idx + 1], str):
                        raise RuntimeError("Ellipsis must be followed by a string")
                    idx += 1
                    select = selector[idx]
                    self._select_string(structure, selections, select, ellipsis=True)
                else:
                    for i, group in enumerate(selections):
                        group = selections[i]
                        new_group = []
                        for item in group:
                            for name in structure:
                                if name[: len(item)] == item:
                                    new_group.append(name)
                        selections[i] = new_group
            elif isinstance(select, slice):
                for i, group in enumerate(selections):
                    group = selections[i]
                    new_group = []
                    for item in group:
                        children = structure[item]
                        children = children[select]
                        new_group += [item + (child,) for child in children]

                    selections[i] = new_group

            idx += 1
        return Selection(self, selections)

    def __getitem__(self, selector) -> "Selection":
        return self.getitem_with_selections(selector)

    def _build_arguments_from(self, *args, **kwargs):
        argspec = self.get_argspec()

        params = argspec.args

        arguments = {}

        # Assign positional arguments to their respective parameter names
        for param_name, arg in zip(params, args):
            arguments[param_name] = arg

        variadic_args = ()
        if argspec.varargs is not None:
            variadic_args = args[len(params) :]

        # Add/Override with keyword arguments
        arguments.update(kwargs)

        arguments.pop("self", None)
        return arguments, variadic_args

    def _assert_valid_configurable(self, *args):
        if args[0] not in self.configurables:
            raise ValueError(
                f"Unknown configurable {args[0]} for {self.__class__.__name__}. "
                f"Available configurables are {self.configurables}."
            )

    def _run_hooks(self, hook_name, instance=None):
        if instance is None:
            instance = self
        hooks = self.__hooks__[hook_name]
        # remove duplicates based on id
        hooks = list({id(hook): hook for hook in hooks}.values())
        for hook in hooks:
            hook(instance)

    def __construct__(self):
        with not_top_level(ExtendedConstructorMeta, self):
            # Reset construction
            self._modules.clear()
            self._user_config.remove_derived_configurations(self.tags)

            self.is_constructing = True

            args, kwargs = self.get_init_args()
            getattr(self, self._init_method)(*(args + self._args), **kwargs)

            self._run_hooks("after_init")
            self.is_constructing = False
            self.__post_init__()

    def _cleanup_and_construct(self):
        self._modules.clear()
        self._user_config.remove_derived_configurations()
        self.__construct__()

    def _append_to_tape(self, obj, method_name, args, kwargs):
        if (
            self.root_module._is_calling_stateful_method
            or self.is_constructing
            or self.root_module.is_constructing
        ):
            return
        self._config_tape.append((obj.tags, method_name, args, kwargs))

    def _replay_tape(self):
        memo = {}
        for tags, method_name, args, kwargs in self._config_tape.copy():
            target = self._get_module_from_tags(tags)
            method = getattr(target, method_name)
            args, kwargs = self._copy_args_and_kwargs(args, kwargs, memo)
            with target.calling_stateful():
                method(*args, **kwargs)

    def _get_module_from_tags(self, tags):
        for name, module in self.root_module.named_modules():

            if name == "":
                name = ()
            else:
                name = tuple(name.split("."))

            if name in tags:
                return module

        raise ValueError(f"Module with tags {tags} not found in {self.root_module}")

    def calling_stateful(self):
        class Stateful:

            def __init__(st_self):
                st_self._is_calling_stateful_method_previous_state = (
                    self._is_calling_stateful_method
                )
                st_self._root_is_calling_stateful_method_previous_state = (
                    self.root_module._is_calling_stateful_method
                )

            def __enter__(st_self):
                self._is_calling_stateful_method = True
                self.root_module._is_calling_stateful_method = True

            def __exit__(st_self, *args):
                self._is_calling_stateful_method = (
                    st_self._is_calling_stateful_method_previous_state
                )
                self.root_module._is_calling_stateful_method = (
                    st_self._root_is_calling_stateful_method_previous_state
                )

        return Stateful()

    def __getstate__(self):
        args, kwargs = self._actual_init_args["args"], self._actual_init_args["kwargs"]
        _config_tape = self._config_tape

        # Ensure state dict is only saved for the root module
        # to avoid saving the same state dict multiple times
        if self.root_module is self:
            state_dict = self.state_dict()
        else:
            state_dict = None

        return {
            "args": args,
            "kwargs": kwargs,
            "_config_tape": _config_tape,
            "state_dict": state_dict,
        }

    def __setstate__(self, state):

        object.__setattr__(
            self,
            "_actual_init_args",
            {"args": state["args"], "kwargs": state["kwargs"]},
        )
        object.__setattr__(self, "_config_tape", [])
        object.__setattr__(self, "_is_calling_stateful_method", False)

        self.__pre_init__(*state["args"], **state["kwargs"])

        object.__setattr__(self, "_default_attribute_keys", set(self.__dict__.keys()))

        with not_top_level(ExtendedConstructorMeta, self):
            self.__construct__()
            self.__post_init__()

        object.__setattr__(
            self,
            "_user_attribute_keys",
            set(self.__dict__.keys()) - self._default_attribute_keys,
        )

        self._config_tape = state["_config_tape"]
        self._replay_tape()
        if state["state_dict"] is not None:
            self.load_state_dict(state["state_dict"])

    def _copy_args_and_kwargs(self, args, kwargs, memo=None):
        memo = {} if memo is None else memo
        new_args = []
        for arg in args:
            if isinstance(arg, DeeplayModule):
                new_args.append(arg.new(memo=memo))
            else:
                new_args.append(copy.deepcopy(arg, memo))

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DeeplayModule):
                new_kwargs[key] = value.new(memo=memo)
            else:
                new_kwargs[key] = copy.deepcopy(value, memo)

        return new_args, new_kwargs

    def get_init_args(self):
        argspec = self.get_argspec()
        signature = self.get_signature()

        args = ()
        kwargs = self.kwargs.copy()

        # extract positional arguments from kwargs
        # and put them in args
        if argspec.varargs is not None:
            for name, param in signature.parameters.items():
                if param.kind == param.VAR_POSITIONAL:
                    break
                if param.name in kwargs:
                    args += (kwargs.pop(param.name),)
        return args, kwargs

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "forward method not implemented for {}".format(self.__class__.__name__)
        )

    def __call__(self, *args, **kwargs):
        if not self._has_built:
            self._register_input(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _register_input(self, *args, **kwargs):
        self._update_computed_values(*args, **kwargs)

    def _update_computed_values(self, *args, **kwargs):
        new_values = {}
        for name, cb in self._computed_values.items():
            new_values[name] = cb(*args, **kwargs)
        if new_values:
            self.configure(**new_values)
        return new_values

    def computed(self, _x: Union[Callable, str], _f: Optional[Callable] = None):
        """Register a computed value.

        Can be used as a decorator or as a function.
        If the first argument is a string, it is assumed to be the name of the computed
        value. If the second argument is not set, the function is used as a decorator.
        If the first argument is a function, it is assumed to be the function to be
        computed, and the name of the function is used as the name of the computed value.

        Parameters
        ----------
        _x : Union[Callable, str]
            The function to be computed or the name of the computed value.
        _f : Optional[Callable], optional
            The function to be computed, by default None.

        Examples
        --------
        As a decorator:
        >>> @module.computed("in_features")
        >>> def f(x):
        >>>     return x.shape[1]

        As a function:
        >>> module.computed("in_features", lambda x: x.shape[1])

        As with a function
        >>> def in_features(x):
        >>>     return x.shape[1]
        >>> module.computed(in_features)


        """
        if isinstance(_x, str):
            name = _x
            if _f is None:
                return lambda f: self.computed(name, f)
            else:
                self._computed_values[name] = _f
        else:
            name = _x.__name__
            self._computed_values[name] = _x


class Selection(DeeplayModule):
    def __init__(self, model: nn.Module, selections: List[List[Tuple[str]]]):
        super().__init__()
        self.model = (model,)
        self.selections = selections
        self.first = _MethodForwarder(self, "first")
        self.all = _MethodForwarder(self, "all")

    def __getitem__(self, selector):
        return self.model[0].getitem_with_selections(selector, self.selections.copy())

    def __repr__(self):
        s = ""
        for selection in self.selections:
            for item in selection:
                s += ".".join(item) + "\n"
        s = "Selection(\n" + s + ")"
        return s

    def list_names(self):
        names = []
        for selection in self.selections:
            for item in selection:
                names.append(item)
        return names

    def filter(self, func: Callable[[str, nn.Module], bool]) -> "Selection":
        """Filter the selection based on a function that takes the module name (separated by .) and module as input.

        Parameters
        ----------
        func : Callable[[str, nn.Module], bool]
            A function that takes the module name (separated by .) and module as input and returns a boolean.

        Returns
        -------
        Selection
            A new selection with the modules that satisfy the condition.
        """
        new_selections = [selection.copy() for selection in self.selections]
        for n, module in self.model[0].named_modules():
            for selection in new_selections:
                for item in selection:
                    asstr = ".".join(item)
                    if asstr == n and not func(n, module):
                        selection.remove(item)

        return Selection(self.model[0], new_selections)

    def hasattr(
        self, attr: str, strict=True, include_layer_classtype: bool = True
    ) -> "Selection":
        """Filter the selection based on whether the modules have a certain attribute.

        Note, for layers, the attribute is checked in the layer's classtype
        (if include_layer_classtype is True). However, this does not include
        non-class attributes of the layer since they are not accessible from the
        layer's classtype. For example, Selection(Layer(nn.Conv2d)).hasattr("kernel_size")
        will return False, but Selection(Layer(nn.Conv2d)).hasattr("_conv_forward") will return True.

        Parameters
        ----------
        attr : str
            The attribute to check for.
        strict : bool, optional
            Whether to only accept real attributes and methods. This excludes properties. By default True
        include_layer_classtype : bool, optional
            Whether to check the attribute in the layer's classtype, by default True

        Returns
        -------
        Selection
            A new selection with the modules that have the attribute.
        """
        from deeplay.external import Layer

        def _filter_fn(name: str, module: nn.Module):
            if not hasattr(module, attr):
                if include_layer_classtype and isinstance(module, Layer):
                    return hasattr(module.classtype, attr)
                return False

            if strict:
                from deeplay.list import ReferringLayerList

                if isinstance(getattr(module, attr), ReferringLayerList):
                    return False
            return True

        return self.filter(_filter_fn)

    def isinstance(
        self, cls: type, include_layer_classtype: bool = True
    ) -> "Selection":
        """Filter the selection based on whether the modules are instances of a certain class.

        Note, for layers, the class is checked in the layer's classtype
        (if include_layer_classtype is True).

        Parameters
        ----------
        cls : type
            The class to check for.
        include_layer_classtype : bool, optional
            Whether to check the class in the layer's classtype, by default True

        Returns
        -------
        Selection
            A new selection with the modules that are instances of the class.
        """

        from deeplay.external import Layer

        return self.filter(
            lambda _, module: isinstance(module, cls)
            or (
                include_layer_classtype
                and isinstance(module, Layer)
                and isinstance(module.classtype, type)
                and issubclass(module.classtype, cls)
            )
        )

    def configure(self, *args, **kwargs):
        """Applies `DeeplayModule.configure` to all modules in the selection."""
        return self.all.configure(*args, **kwargs)

    def replace(self, *args, **kwargs):
        """Applies `DeeplayModule.replace` to all modules in the selection."""
        return self.all.replace(*args, **kwargs)

    def log_output(self, key):
        """Applies `DeeplayModule.log_output` to the first module in the selection."""
        return self.first.log_output(key)

    def log_input(self, key):
        """Applies `DeeplayModule.log_input` to the first module in the selection."""
        return self.first.log_input(key)

    def append(self, *args, **kwargs):
        """Applies `SequentialBlock.append` to all modules in the selection."""
        return self.all.append(*args, **kwargs)

    def prepend(self, *args, **kwargs):
        """Applies `SequentialBlock.prepend` to all modules in the selection."""
        return self.all.prepend(*args, **kwargs)

    def insert(self, *args, **kwargs):
        """Applies `SequentialBlock.insert` to all modules in the selection."""
        return self.all.insert(*args, **kwargs)

    def remove(self, *args, **kwargs):
        """Applies `SequentialBlock.remove` to all modules in the selection."""
        return self.all.remove(*args, **kwargs)

    def append_dropout(self, *args, **kwargs):
        """Applies `SequentialBlock.append_dropout` to all modules in the selection."""
        return self.all.append_dropout(*args, **kwargs)

    def prepend_dropout(self, *args, **kwargs):
        """Applies `SequentialBlock.prepend_dropout` to all modules in the selection."""
        return self.all.prepend_dropout(*args, **kwargs)

    def insert_dropout(self, *args, **kwargs):
        """Applies `SequentialBlock.insert_dropout` to all modules in the selection."""
        return self.all.insert_dropout(*args, **kwargs)

    def remove_dropout(self, *args, **kwargs):
        """Applies `SequentialBlock.remove_dropout` to all modules in the selection."""
        return self.all.remove_dropout(*args, **kwargs)

    def set_dropout(self, *args, **kwargs):
        """Applies `SequentialBlock.set_dropout` to all modules in the selection."""
        return self.all.set_dropout(*args, **kwargs)

    def set_input_map(self, *args, **kwargs):
        """Applies `DeeplayModule.set_input_map` to all modules in the selection."""
        return self.all.set_input_map(*args, **kwargs)

    def set_output_map(self, *args, **kwargs):
        """Applies `DeeplayModule.set_output_map` to all modules in the selection."""
        return self.all.set_output_map(*args, **kwargs)


class _MethodForwarder:

    def __init__(self, selection: Selection, mode: str):
        self.mode = mode
        self.selection = selection

    def _create_forwarder(self, name):
        def forwarder(*args, **kwargs):
            for selection in self.selection.selections:
                for item in selection:
                    for n, module in self.selection.model[0].named_modules():
                        if n == ".".join(item):
                            try:
                                v = getattr(module, name)(*args, **kwargs)
                                if self.mode == "first":
                                    return self
                            except AttributeError as e:
                                raise AttributeError(
                                    f"Module {module} does not have a method {name}. "
                                    "Use selection.hasattr('method_name') to filter modules that have the method."
                                ) from e
            return self

        return forwarder

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self._create_forwarder(name)

    def configure(self, *args, **kwargs):
        """See `DeeplayModule.configure`."""
        return self._create_forwarder("configure")(*args, **kwargs)

    def replace(self, *args, **kwargs):
        """See `DeeplayModule.replace`."""
        return self._create_forwarder("replace")(*args, **kwargs)

    def log_output(self, key):
        """See `DeeplayModule.log_output`."""
        return self._create_forwarder("log_output")(key)

    def log_input(self, key):
        """See `DeeplayModule.log_input`."""
        return self._create_forwarder("log_input")(key)

    def append(self, *args, **kwargs):
        """See `SequentialBlock.append`."""
        return self._create_forwarder("append")(*args, **kwargs)

    def prepend(self, *args, **kwargs):
        """See `SequentialBlock.prepend`."""
        return self._create_forwarder("prepend")(*args, **kwargs)

    def insert(self, *args, **kwargs):
        """See `SequentialBlock.insert`."""
        return self._create_forwarder("insert")(*args, **kwargs)

    def remove(self, *args, **kwargs):
        """See `SequentialBlock.remove`."""
        return self._create_forwarder("remove")(*args, **kwargs)

    def append_dropout(self, *args, **kwargs):
        """See `SequentialBlock.append_dropout`."""
        return self._create_forwarder("append_dropout")(*args, **kwargs)

    def prepend_dropout(self, *args, **kwargs):
        """See `SequentialBlock.prepend_dropout`."""
        return self._create_forwarder("prepend_dropout")(*args, **kwargs)

    def insert_dropout(self, *args, **kwargs):
        """See `SequentialBlock.insert_dropout`."""
        return self._create_forwarder("insert_dropout")(*args, **kwargs)

    def remove_dropout(self, *args, **kwargs):
        """See `SequentialBlock.remove_dropout`."""
        return self._create_forwarder("remove_dropout")(*args, **kwargs)

    def set_dropout(self, *args, **kwargs):
        """See `SequentialBlock.set_dropout`."""
        return self._create_forwarder("set_dropout")(*args, **kwargs)
