import inspect
import torch.nn as nn
import torch

__all__ = ["safe_call"]


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
