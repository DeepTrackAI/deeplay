import inspect
import torch.nn as nn
import torch

__all__ = [
    "safe_call",
    "match_signature",
    "center_pad_to_largest",
    "center_crop_to_smallest",
    "center_pad",
    "center_crop",
]


def safe_call(cls, kwargs):
    """Safely call a callable object.

    If the object is a uninstantiated class, then it is instantiated
    with the given keyword arguments. If the object is already instantiated,
    then its __call__ method is called with the given keyword arguments.

    The object's signature is checked to see if it accepts the given keyword
    arguments. Only keyword arguments accepted by the object are passed.

    Parameters
    ----------
    cls : class
        The class to instantiate.
    kwargs : dict
        The keyword arguments to pass to the class constructor.

    Returns
    -------
    object
        The instantiated object.
    """
    signature = inspect.signature(cls.__init__ if inspect.isclass(cls) else cls)

    # if accepts **kwargs, then pass all kwargs
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return cls(**kwargs)

    valid_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return cls(**valid_kwargs)


def center_pad_to_largest(inputs):
    """Pad the inputs to the largest input.
    Two first dimensions are assumed to be batch and channel and are not padded.

    Parameters
    ----------
    inputs : list of torch.Tensor
        List of inputs.

    Returns
    -------
    list of torch.Tensor
        List of padded inputs.
    """

    max_shape = max(inp.shape[2:] for inp in inputs)
    return [center_pad(inp, max_shape) for inp in inputs]


def center_pad(inp, shape):
    """Pad the input to the given shape.
    Two first dimensions are assumed to be batch and channel and are not padded.

    Parameters
    ----------
    inp : torch.Tensor
        Input.
    shape : tuple of int
        Shape to pad to.

    Returns
    -------
    torch.Tensor
        Padded input.
    """
    pads = []
    for i, s in zip(inp.shape[2:], shape):
        before = (s - i) // 2
        after = s - i - before

        # reversed since we will reverse the list later
        pads.extend([after, before])

    # reversed since pad expects padding for the last dimension first
    pads = pads[::-1]

    return nn.functional.pad(inp, pads)


def center_crop_to_smallest(inputs):
    """Crop the inputs to the smallest input.
    Two first dimensions are assumed to be batch and channel and are not cropped.

    Parameters
    ----------
    inputs : list of torch.Tensor
        List of inputs.

    Returns
    -------
    list of torch.Tensor
        List of cropped inputs.
    """

    min_shape = min(inp.shape[2:] for inp in inputs)
    return [center_crop(inp, min_shape) for inp in inputs]


def center_crop(inp, shape):
    """Crop the input to the given shape.
    Two first dimensions are assumed to be batch and channel and are not cropped.

    Parameters
    ----------
    inp : torch.Tensor
        Input.
    shape : tuple of int
        Shape to crop to.

    Returns
    -------
    torch.Tensor
        Cropped input.
    """

    slices = []
    for i, s in enumerate(shape):
        before = (inp.shape[i + 2] - s) // 2
        after = before + s
        slices.append(slice(before, after))

    return inp[(...,) + tuple(slices)]


def as_kwargs(f, args, kwargs):
    # Get the signature of the function
    sig = inspect.signature(f)
    parameters = sig.parameters

    # Separate positional-only, regular, *args, and **kwargs parameters
    pos_only_params = [
        name
        for name, param in parameters.items()
        if param.kind == param.POSITIONAL_ONLY
    ]
    regular_params = [
        name
        for name, param in parameters.items()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    var_args = [
        name for name, param in parameters.items() if param.kind == param.VAR_POSITIONAL
    ]
    var_kwargs = [
        name for name, param in parameters.items() if param.kind == param.VAR_KEYWORD
    ]

    # Remove "self" from regular parameters if it is the first parameter
    if regular_params and regular_params[0] == "self":
        regular_params = regular_params[1:]

    # Map positional arguments to their corresponding keyword names
    args_as_kwargs = dict(zip(regular_params, args))

    # If there are more positional arguments than regular parameters and the function does not accept *args, raise an error
    if len(args) > len(regular_params) and not var_args:
        raise ValueError("Too many positional arguments provided for function.")

    # Merge the two dictionaries, with the user-provided kwargs taking precedence
    # This handles the case where a positional argument and a keyword argument refer to the same parameter
    combined_kwargs = {**args_as_kwargs, **kwargs}

    # If the function does not accept **kwargs, filter out any extra keyword arguments
    if not var_kwargs:
        combined_kwargs = {
            k: v for k, v in combined_kwargs.items() if k in regular_params
        }

    # Check for missing required arguments
    missing_args = [
        k
        for k, p in parameters.items()
        if p.default == p.empty
        and k not in combined_kwargs
        and p.kind not in [p.VAR_POSITIONAL, p.VAR_KEYWORD]
    ]

    # Allow 'self' to be missing
    if "self" in missing_args:
        missing_args.remove("self")

    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

    return combined_kwargs


def match_signature(func, args, kwargs):
    """Returns a dictionary of arguments that match the signature of func.
    This can be used to find the names of arguments passed positionally.
    """
    return as_kwargs(func, args, kwargs)
