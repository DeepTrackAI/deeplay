from __future__ import annotations
from typing import Any, Callable, Optional, TypeVar, overload, ParamSpec
import inspect
from ..module import DeeplayModule

import torch.nn as nn

T = TypeVar("T")
P = ParamSpec("P")


class External(DeeplayModule):
    __extra_configurables__ = ["classtype"]

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
        super().__pre_init__(*args, classtype=classtype, **kwargs)

    def __init__(self, classtype, *args, **kwargs):
        super().__init__()
        self.classtype = classtype

    def build(self) -> nn.Module:
        kwargs = self.kwargs
        kwargs.pop("classtype", None)

        args = ()

        # check if classtype has *args variadic
        argspec = self.get_argspec()
        signature = self.get_signature()

          
        positional_only_args =[param.name
                                for param in signature.parameters.values()
                                if param.kind == param.POSITIONAL_ONLY]

        # Any positional only arguments should be moved from kwargs to args
        for arg in positional_only_args:
            args = args + (kwargs.pop(arg),)

        if argspec.varargs is not None:
            args = args + self._actual_init_args["args"]

        return self.classtype(*args, **kwargs)

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
