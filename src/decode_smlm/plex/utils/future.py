import builtins
import functools
import sys
from typing import Sequence

from packaging import version


def sys_check(version_min, fn_builtin):
    """
    Decorator to check the current python version and use a builtin function instead of
    the if the implemented function if the python version is equal or higher than the
    specified threshold.

    Args:
        version_min: threshold version, below which an own implementation of zip will be
         used
        fn_builtin: function to use in case current python version matches or exceeds
         that defined by the version variable

    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            v = sys.version_info
            if version.Version(f"{v.major}.{v.minor}.{v.micro}") >= version.parse(
                version_min
            ):
                return fn_builtin(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@sys_check(version_min="3.10", fn_builtin=builtins.zip)
def zip(*args: tuple[Sequence, ...], strict: bool = False) -> builtins.zip:
    """
    Future of zip (py 3.10).

    Note: Different to the original zip, args must be Sequence because we need to check
    the length of the sequences before the actual implementation.

    Args:
        *args:
        strict: whether all sequences must have the same length

    Returns:
        generator
    """
    if strict:
        if not all(len(args[0]) == len(a) for a in args):
            raise ValueError("All arguments must have same length.")
    return builtins.zip(*args)
