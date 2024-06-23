from __future__ import annotations
from typing import Any, Callable, Optional, TypeVar, overload
import inspect
from deeplay.module import DeeplayModule
from deeplay.meta import ExtendedConstructorMeta, not_top_level
from weakref import WeakKeyDictionary

import torch.nn as nn

T = TypeVar("T")


class External(DeeplayModule):
    __extra_configurables__ = ["classtype"]

    _init_method = "_actual_init"

    @property
    def kwargs(self):
        full_kwargs = super().kwargs
        classtype = full_kwargs.pop("classtype")

        # If classtype accepts **kwargs, we can pass all the kwargs to it.'
        argspec = self.get_argspec()
        if argspec.varkw is not None:
            kwargs = full_kwargs
            kwargs["classtype"] = classtype
            return kwargs

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
        self._non_classtype_args = args
        self._computed = {}
        super().__pre_init__(*args, classtype=classtype, **kwargs)
        self.assert_not_positional_only_and_variadic()

    def _actual_init(self, *args, **kwargs):
        self.classtype = kwargs.pop("classtype")
        self.assert_not_positional_only_and_variadic()

    def assert_not_positional_only_and_variadic(self):
        argspec = self.get_argspec()
        signature = self.get_signature()

        positional_only_args = [
            param
            for param in signature.parameters.values()
            if param.kind == param.POSITIONAL_ONLY
        ]

        has_variadic = argspec.varargs is not None

        if positional_only_args and has_variadic:
            raise TypeError(
                f"Cannot use both positional only arguments and *args with {self.__class__.__name__}. Consider wrapping the classtype in a wrapper class."
            )

    def build(self) -> nn.Module:
        self._run_hooks("before_build")

        kwargs = self.kwargs
        kwargs.pop("classtype", None)

        args = ()

        # check if classtype has *args variadic
        argspec = self.get_argspec()
        signature = self.get_signature()

        positional_only_args = [
            param.name
            for param in signature.parameters.values()
            if param.kind == param.POSITIONAL_ONLY
        ]

        # Any positional only arguments should be moved from kwargs to args
        for arg in positional_only_args:
            args = args + (kwargs.pop(arg),)

        if argspec.varargs is not None:
            args = args + self._non_classtype_args

        # Remove *args and **kwargs from kwargs
        for key in list(kwargs.keys()):
            if key in signature.parameters and (
                signature.parameters[key].kind == signature.parameters[key].VAR_KEYWORD
                or signature.parameters[key].kind
                == signature.parameters[key].VAR_POSITIONAL
            ):
                kwargs.pop(key)

        if self.classtype.__init__ is nn.Module.__init__:
            obj = self.classtype()
        else:
            obj = self.classtype(*args, **kwargs)

        if not isinstance(obj, DeeplayModule):
            obj._root_module = self._root_module
        self._execute_mapping_if_valid(obj)
        self._run_hooks("after_build", obj)
        return obj

    create = build

    def get_init_args(self):
        kwargs = self.kwargs.copy()
        # hack for external
        # classtype = kwargs.pop("classtype")
        return (), kwargs

    def get_argspec(self):
        classtype = self.classtype

        init_method = classtype.__init__ if inspect.isclass(classtype) else classtype
        argspec = inspect.getfullargspec(init_method)

        if "self" in argspec.args:
            argspec.args.remove("self")

        if (
            not argspec.args
            and inspect.isclass(classtype)
            and issubclass(classtype, nn.RNNBase)
        ):
            # This is a hack to get around torch RNN classes
            parent_init = classtype.__mro__[1].__init__
            argspec = inspect.getfullargspec(parent_init)
            argspec.args.remove("self")
            argspec.args.remove("mode")

        return argspec

    def get_signature(self):
        classtype = self.classtype

        if not inspect.isclass(classtype):
            return inspect.signature(classtype)
        elif issubclass(classtype, DeeplayModule):
            return classtype.get_signature()
        elif issubclass(classtype, nn.RNNBase):
            signature = inspect.signature(classtype.__mro__[1])
            params = list(signature.parameters.values())
            params.pop(0)  # corresponding "mode" in RNNBase
            return inspect.Signature(params)
        return inspect.signature(classtype)

    def build_arguments_from(self, *args, classtype, **kwargs):
        kwargs = super().build_arguments_from(*args, **kwargs)
        kwargs["classtype"] = classtype
        return kwargs

    @overload
    def configure(self, classtype, **kwargs) -> None: ...

    @overload
    def configure(self, **kwargs: Any) -> None: ...

    def configure(self, classtype: Optional[type] = None, **kwargs):
        if classtype is not None:
            super().configure(classtype=classtype)
        super().configure(**kwargs)

    def _assert_valid_configurable(self, *args):
        if self.get_argspec().varkw is not None:
            return
        return super()._assert_valid_configurable(*args)

    def _execute_mapping_if_valid(self, module):
        if getattr(self, "_input_mapped", False) and getattr(
            self, "_output_mapped", False
        ):
            self._set_mapping(
                module, self.input_args, self.input_kwargs, self.output_args
            )

    def __repr__(self):
        classkwargs = ", ".join(
            f"{key}={value}" for key, value in self.kwargs.items() if key != "classtype"
        )
        return f"{self.__class__.__name__}[{self.classtype.__name__}]({classkwargs})"

    # def __reduce__(self):
    #     # External object do not support
    #     return object.__reduce__(self)
