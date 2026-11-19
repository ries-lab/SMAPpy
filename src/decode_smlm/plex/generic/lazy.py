import functools
from typing import Any, Optional, Union, Sequence

import torch

from . import mixin


def no_op_on(attr: str):
    """
    Modifies a method to no-op if specified class / instance attribute is None; original
    arguments are then returned. Only works for positional arguments.

    Args:
        attr: attribute to check

    Examples:
        >>> class Dummy:
        >>>    def __init__(self, factor):
        >>>        self._factor = factor
        >>>    @no_op_on("_factor")
        >>>    def multiply(self, /, x):
        >>>        return x * self._factor

    """

    def wrapping_method(fn):
        @functools.wraps(fn)
        def wrapped_method(self, *args):
            if getattr(self, attr) is None:
                return args if len(args) >= 2 else args[0]
            return fn(self, *args)

        return wrapped_method

    return wrapping_method


class To(mixin.ForwardCallAlias):
    def __init__(self, device: Union[str, torch.device]):
        """
        Moves stuff (recursively) to specified device.
        Arguments or childs need to implement `.to` method.

        Args:
            device: target device
        """
        self._device = device

    def forward(self, *args: tuple[Any]) -> Any:
        """
        Puts arguments to specified device. Positional arguments only.

        Args:
            *args:

        """
        out = tuple(self.kernel(arg) for arg in args)
        return out if len(out) >= 2 else out[0]

    def kernel(
        self, arg: Optional[Union[torch.Tensor, Sequence]]
    ) -> Optional[Union[torch.Tensor, Sequence]]:
        if arg is None:
            return None
        if hasattr(arg, "to"):
            return arg.to(self._device)
        if isinstance(arg, Sequence):
            return tuple(self.kernel(x) for x in arg)  # recurse
        if isinstance(arg, dict):
            return {k: self.kernel(v) for k, v in arg.items()}

        raise TypeError(
            f"Argument {arg} does not implement `.to` method "
            f"and could not be recursed."
        )


def unfold_magic_dict(x: Any) -> None:
    # prints `__dict__` dunder recursively
    if hasattr(x, "__dict__"):
        print(x.__dict__)
        for v in x.__dict__.values():
            unfold_magic_dict(v)


def forward_relay(method: str):
    """Relays a method call to a wrapped object, usage mostly as decorator"""

    def wrapper(instance):
        class Wrapped(type(instance)):
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, item):
                return getattr(self._wrapped, item)

            def forward(self, *args, **kwargs):
                return getattr(self._wrapped, method)(*args, **kwargs)

        return Wrapped(instance)

    return wrapper
