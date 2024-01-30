import inspect
from typing import Any, Dict, Tuple, List, Set, Literal, Optional

import torch.nn as nn
import copy
import inspect

from .meta import ExtendedConstructorMeta, not_top_level
from .decorators import after_init, after_build
from functools import partial


class UserConfig(dict):
    __hook_containers__ = [
        ("__user_hooks__",),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, __m):
        for key, value in __m.items():
            if key[-1] == "__user_hooks__":
                if key not in self:
                    self[key] = value.copy()
                else:
                    for hook_name, hooks in value.items():
                        self[key][hook_name] += hooks
            else:
                self[key] = value

    def prefix(self, tags: List[Tuple[str, ...]]):
        d = {}
        for tag in tags:
            for key, value in self.items():
                d[tag + key] = value
        return UserConfig(d)

    def take(self, tags: List[Tuple[str, ...]]):
        res = UserConfig()
        for tag in tags:
            res.update(
                {
                    key: value
                    for key, value in self.items()
                    if len(key) == len(tag) + 1 and key[: len(tag)] == tag
                }
            )

        return res

    def set_for_tags(self, tags: List[Tuple[str, ...]], key, value):
        for tag in tags:
            self[tag + (key,)] = value


def _create_forward_with_input_dict(
    old_forward,
    input_args: List[str],
    input_kwargs: Dict[str, str],
    output_args: Optional[Dict[str, int]],
):
    def forward_with_input_dict(self, x, overwrite_output: bool = True):
        assert isinstance(
            x, dict
        ), "Input must be a dictionary, but found {}. Please check if the module require an input/output mapping.".format(
            type(x)
        )
        x = x.copy()

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
                map(
                    lambda key, value: (key, outputs[value]),
                    *zip(*output_args.items()),
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

    _args: tuple
    _kwargs: dict
    _actual_init_args: dict
    _has_built: bool
    _setattr_recording: Set[str]
    _tag: Tuple[str, ...]

    @property
    def tags(self) -> List[Tuple[str, ...]]:
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
        return {
            k: v + self.__parent_hooks__[k] + self.__user_hooks__[k]
            for k, v in self.__constructor_hooks__.items()
        }

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
            if tag + ("__user_hooks__",) in user_config:
                return user_config[tag + ("__user_hooks__",)]
        # not found, add
        __user_hooks__: Dict[
            Literal["before_build", "after_build", "after_init"], list
        ] = {
            "before_build": [],
            "after_build": [],
            "after_init": [],
        }
        for tag in tags:
            user_config[tag + ("__user_hooks__",)] = __user_hooks__

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
        if ExtendedConstructorMeta._is_top_level["value"]:
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

    def set_root_module(self, value):
        self._root_module = (value,)
        for name, module in self.named_modules():
            module._root_module = (value,)

    @property
    def _user_config(self) -> UserConfig:
        if self.root_module is None:
            return self._base_user_config
        if self.root_module is self:
            return self._base_user_config

        return self.root_module._user_config

    def __pre_init__(self, *args, _args=(), **kwargs):
        super().__init__()
        self._root_module = (self,)

        self._actual_init_args = {
            "args": args,
            "_args": _args,
            "kwargs": kwargs,
        }

        self._base_user_config = UserConfig()

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

    def __init__(self, *args, **kwargs):  # type: ignore
        # We don't want to call the super().__init__ here because it is called
        # in the __pre_init__ method.
        ...

    def __post_init__(self):
        ...

    def set_input_map(self, *args: str, **kwargs: str):
        self.__dict__.update(
            {"input_args": args, "input_kwargs": kwargs, "_input_mapped": True}
        )
        self._execute_mapping_if_valid()

    def set_output_map(self, *args: str, **kwargs: int):
        output_args = {arg: i for i, arg in enumerate(args)}
        output_args.update(kwargs)

        self.__dict__.update({"output_args": output_args, "_output_mapped": True})
        self._execute_mapping_if_valid()

    def _execute_mapping_if_valid(self):
        if getattr(self, "_input_mapped", False) and getattr(
            self, "_output_mapped", False
        ):
            self._set_mapping(self.input_args, self.input_kwargs, self.output_args)

    @after_build
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

    def create(self):
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
        obj = self.new()
        obj = obj.build()
        return obj

    def build(self):
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

        self._run_hooks("before_build")

        for name, value in self.named_children():
            if isinstance(value, Optimizer):
                ...  # skip optimizers
            elif isinstance(value, DeeplayModule):
                if value._has_built:
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

        self._has_built = True
        self._run_hooks("after_build")

        return self

    def new(self):
        return copy.deepcopy(self)

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

        return v[-1]

    def _configure_kwargs(self, kwargs):
        for name, value in kwargs.items():
            self._assert_valid_configurable(name)
            self._user_config.set_for_tags(self.tags, name, value)
        self.__construct__()

    def _give_user_configuration(self, receiver: "DeeplayModule", name):
        if receiver._user_config is self._user_config:
            if receiver.root_module is not receiver:
                receiver.set_root_module(self.root_module)
            return
        if receiver.root_module is self.root_module:
            return

        mytags = self.tags
        receivertags = receiver.tags
        # sort longest tag first
        receivertags.sort(key=lambda x: len(x), reverse=True)

        d = {}
        for _, model in self.named_modules():
            if isinstance(model, DeeplayModule):
                tags = model.tags
                subconfig = receiver._user_config.take(tags)
                for key, value in subconfig.items():
                    for tag in receivertags:
                        if len(key) >= len(tag) and key[: len(tag)] == tag:
                            key = key[len(tag) :]
                            break

                    for tag in mytags:
                        d[tag + (name,) + key] = value

        self._user_config.update(d)
        receiver.set_root_module(self.root_module)

    def __setattr__(self, name, value):
        if name == "_user_config" and hasattr(self, "_user_config"):
            if not isinstance(value, UserConfig):
                raise ValueError("User configuration must be a UserConfig instance.")
            super().__setattr__(name, value)

        super().__setattr__(name, value)

        if self.is_constructing:
            if isinstance(value, DeeplayModule) and not value._has_built:
                self._give_user_configuration(value, name)
                value.__construct__()
            self._setattr_recording.add(name)

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
        for hook in self.__hooks__[hook_name]:
            hook(instance)

    def __construct__(self):
        with not_top_level(ExtendedConstructorMeta, self):
            self._modules.clear()

            self.is_constructing = True

            args, kwargs = self.get_init_args()
            getattr(self, self._init_method)(*(args + self._args), **kwargs)

            self._run_hooks("after_init")
            self.is_constructing = False
            self.__post_init__()

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

    @after_init
    def set_p(self, p):
        self.p = p


class Selection(DeeplayModule):
    def __init__(self, model, selections):
        super().__init__()
        self.model = model
        self.selections = selections

    def __getitem__(self, selector):
        return self.model.getitem_with_selections(selector, self.selections.copy())

    def __repr__(self):
        s = ""
        for selection in self.selections:
            for item in selection:
                s += ".".join(item) + "\n"
        return s

    def list_names(self):
        names = []
        for selection in self.selections:
            for item in selection:
                names.append(item)
        return names

    def configure(self, *args, **kwargs):
        for selection in self.selections:
            for item in selection:
                for name, module in self.model.named_modules():
                    if name == ".".join(item):
                        module.configure(*args, **kwargs)
