"""The ``@deprecated`` decorator of the `Deprecated` package, as DECODE-Plex called it.

decode_smlm.plex (docs/decode/plex-plan.md §3): a dependency for a decorator is replaced by this
one.  Both forms are taken, ``@deprecated`` and ``@deprecated(reason=..., version=..., action=...)``;
a call warns with a `DeprecationWarning`, or raises with ``action="error"``, as the package did.
"""
import functools
import warnings


def deprecated(*args, reason: str = "", version: str = "", action: str | None = None, **kwargs):
    def wrap(obj):
        what = f"{getattr(obj, '__qualname__', obj)} is deprecated"
        if version:
            what += f" (version {version})"
        if reason:
            what += f": {reason}"

        def check():
            if action == "error":
                raise DeprecationWarning(what)
            if action != "ignore":
                warnings.warn(what, DeprecationWarning, stacklevel=3)

        if isinstance(obj, type):
            init = obj.__init__

            @functools.wraps(init)
            def __init__(self, *a, **k):
                check()
                init(self, *a, **k)

            obj.__init__ = __init__
            return obj

        @functools.wraps(obj)
        def wrapped(*a, **k):
            check()
            return obj(*a, **k)

        return wrapped

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return wrap(args[0])
    return wrap
