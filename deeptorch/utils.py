import inspect


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
