from turtle import st
from typing import List, Type, get_type_hints
import deeplay as dl
import os
import sys
import inspect
import importlib
import pkgutil


def main():

    # iterate all classes in deeplay and its submodules
    classes = get_classes(dl)

    def filter_cls(cls):
        if not issubclass(cls, dl.DeeplayModule):
            return False
        if cls.available_styles() == []:
            return False
        return True

    classes = list(filter(filter_cls, classes))

    for cls in classes:
        print(f"Generating stub for {cls.__name__}")
        path = find_module_path(cls)
        stub_path = create_base_stub(path, cls)
        add_imports(stub_path, cls)

    for cls in classes:
        path = find_module_path(cls)
        stub_path = path.replace(".py", ".pyi")
        print(f"Adding style overloads for {cls.__name__}")
        add_style_overloads(stub_path, cls)


def get_classes(module):
    """
    Given a module, return a list of all classes defined in that module and its submodules.
    """
    classes = []
    modules = [module]
    if hasattr(module, "__path__"):
        for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            # check if module is already imported
            if name in sys.modules:
                modules.append(sys.modules[name])
            else:
                try:
                    modules.append(importlib.import_module(name))
                except Exception as e:
                    print(f"Error importing {name}: {e}")

    for mod in modules:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__:
                classes.append(obj)
    return classes


def find_module_path(cls):
    module = cls.__module__
    path = sys.modules[module].__file__
    return path


def create_base_stub(path, cls):
    # call stubgen
    output = "."
    os.system(f"stubgen {path} -o {output}")
    return path.replace(".py", ".pyi")


def add_imports(stub_path, cls):
    with open(stub_path, "r") as f:
        lines = f.readlines()

    imports = [
        "from typing import Literal, Type, Union, Optional, overload\n",
        "from typing_extensions import Self\n",
    ]

    lines = imports + lines

    with open(stub_path, "w") as f:
        f.writelines(lines)


def add_style_overloads(stub_path, cls):
    with open(stub_path, "r") as f:
        lines = f.readlines()

    styles: List[str] = cls.available_styles()

    # Find line that starts with "class {cls.__name__}"
    for class_def_start, line in enumerate(lines):
        if line.startswith(f"class {cls.__name__}"):
            break

    # Find first line after class definition that is not indented
    for class_def_end, line in enumerate(
        lines[class_def_start + 1 :], class_def_start + 1
    ):
        if not line.startswith(" ") and not line.startswith("\t"):
            break

    up_to_class_def_end = lines[:class_def_end]
    after_class_def_end = lines[class_def_end:]

    for style in styles:
        style_overload = get_style_overload(cls, style)
        up_to_class_def_end.extend(style_overload)

    up_to_class_def_end.append(
        "    def style(self, style: str, **kwargs) -> Self: ...\n"
    )

    with open(stub_path, "w") as f:
        f.writelines(up_to_class_def_end + after_class_def_end)


def get_style_overload(cls: Type[dl.DeeplayModule], style: str):
    func = cls._style_map[style]
    style_overload = generate_mypy_stub(func)

    style_overload = [
        "    @overload\n",
        f'    def style(self, style: Literal["{style}"], {style_overload}) -> Self:',
    ]
    if func.__doc__ is not None:
        style_overload[-1] += "\n"
        doc = func.__doc__.split("\n")
        style_overload.append(f'        """{doc[0]}\n')
        style_overload.extend([f"    {line}\n" for line in doc[1:]])
        style_overload.append('    """\n')
    else:
        style_overload[-1] += " ...\n"
    return style_overload


def generate_mypy_stub(func):
    signature = inspect.signature(func)
    params = []
    # remove first param (self)
    signature = signature.replace(parameters=list(signature.parameters.values())[1:])

    type_hints = get_type_hints(func)
    for param in signature.parameters.values():
        param_str = param.name
        if param.annotation != inspect.Parameter.empty:
            param_type = type_hints.get(param.name, param.annotation)
            if getattr(param_type, "__origin__", None) is not None:
                param_str += f": {param.annotation}"
            else:
                param_str += f": {param_type.__name__}"
        if param.default != inspect.Parameter.empty:
            if param.default is None:
                default_str = "None"
            elif isinstance(param.default, str):
                default_str = f'"{param.default}"'
            elif str(param.default).startswith("<"):
                default_str = "..."
            else:
                default_str = str(param.default)
            param_str += f"={default_str}"
        params.append(param_str)
    params_str = ", ".join(params)
    return params_str


if __name__ == "__main__":
    main()
