"""The part of pydantic 1 that DECODE-Plex uses, without pydantic.

decode_smlm.plex (docs/decode/plex-plan.md §3).  DECODE-Plex pinned pydantic 1.10; pydantic 2
carries it only as ``pydantic.v1``, which does not run on Python 3.14, and SMAPpy supports 3.12 and
later.  What the code uses of it is small, so it is written here, keeping pydantic 1's behaviour
where the code depends on it:

- a model's fields are its annotations, through the class hierarchy; a field whose annotation
  admits None (``Optional[X]``, ``X | None``) is None when not given, any other field without a
  default is required; arguments that are not fields are ignored (pydantic's ``extra = ignore``);
- a field's value is coerced by its type: a type with ``__get_validators__`` (the tensors of
  `generic.models`) by its validators, a model from a dict, ``int``, ``float``, ``str`` and
  ``tuple[...]`` by conversion; anything else (``Any``, ``dict``, unions of such) is taken as it is;
- ``@validator("field")`` runs after the field is coerced, ``@root_validator`` on all values after
  the fields; a ``ValueError``, ``TypeError`` or ``AssertionError`` raised in either is reported as a
  `ValidationError`, as pydantic did;
- ``Model.construct(**values)`` builds without validation, ``validate_model(Model, values)``
  validates without building, returning ``(values, fields_set, error)``;
- ``@validate_arguments`` coerces a function's annotated arguments as a model's fields are, an
  argument whose annotation it does not know (a tensor, a camera) passed as it is -- pydantic's
  ``arbitrary_types_allowed``, which DECODE-Plex set where it mattered; ``StrictInt`` takes an
  ``int`` (not a bool) and nothing else.
"""
import functools
import inspect
import types
import typing
from pathlib import Path
from typing import Any, Union

DirectoryPath = Path


class ValidationError(ValueError):
    def __init__(self, errors: list[tuple[str, Exception]], model: type):
        self.errors = errors
        self.model = model
        lines = [f"{len(errors)} validation error(s) for {model.__name__}"]
        lines += [f"{loc}\n  {err}" for loc, err in errors]
        super().__init__("\n".join(lines))


def validator(*fields: str, **kwargs):
    def mark(fn):
        f = fn.__func__ if isinstance(fn, classmethod) else fn
        f.__validated_fields__ = fields
        return f

    return mark


def root_validator(fn=None, **kwargs):
    def mark(f):
        f = f.__func__ if isinstance(f, classmethod) else f
        f.__validated_root__ = True
        return f

    return mark(fn) if fn is not None else mark


def validate_arguments(fn=None, **kwargs):
    def wrap(f):
        sig = inspect.signature(f)
        hints = typing.get_type_hints(f)
        hints.pop("return", None)

        @functools.wraps(f)
        def wrapped(*args, **kw):
            bound = sig.bind(*args, **kw)
            errors = []
            for name, v in bound.arguments.items():
                if name in hints and sig.parameters[name].kind not in (
                        inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    try:
                        bound.arguments[name] = _coerce(hints[name], v)
                    except (ValueError, TypeError, AssertionError) as e:
                        errors.append((name, e))
            if errors:
                raise ValidationError(errors, type(f.__name__, (), {}))
            return f(*bound.args, **bound.kwargs)

        return wrapped

    return wrap(fn) if fn is not None else wrap


class StrictInt(int):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"value is not a valid integer: {v!r}")
        return v


_NONE = type(None)


def _admits_none(tp) -> bool:
    return tp is Any or (typing.get_origin(tp) in (Union, types.UnionType)
                         and _NONE in typing.get_args(tp))


def _coerce(tp, v):
    if v is None:
        if _admits_none(tp):
            return None
        raise TypeError("none is not an allowed value")
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not _NONE]
        if len(args) == 1:
            return _coerce(args[0], v)
        return v
    if hasattr(tp, "__get_validators__"):
        for fn in tp.__get_validators__():
            v = fn(v)
        return v
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        return v if isinstance(v, tp) else tp(**v)
    if origin is tuple:
        args = typing.get_args(tp)
        v = tuple(v)
        if args and args[-1] is not Ellipsis:
            if len(args) != len(v):
                raise ValueError(f"wrong tuple length {len(v)}, expected {len(args)}")
            return tuple(_coerce(a, x) for a, x in zip(args, v))
        return v
    if tp in (int, float, str):
        return tp(v)  # pydantic 1's int() too: 2.7 becomes 2
    return v


class BaseModel:
    __fields__: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        hints = typing.get_type_hints(cls)
        cls.__fields__ = {k: tp for k, tp in hints.items()
                          if not k.startswith("_") and typing.get_origin(tp) is not typing.ClassVar}
        cls.__defaults__ = {}
        for k in cls.__fields__:
            for c in cls.__mro__:
                if k in c.__dict__ and not callable(c.__dict__[k]):
                    cls.__defaults__[k] = c.__dict__[k]
                    break
        cls.__validators__, cls.__root_validators__ = {}, []
        for c in reversed(cls.__mro__):
            for f in c.__dict__.values():
                f = f.__func__ if isinstance(f, (classmethod, staticmethod)) else f
                for name in getattr(f, "__validated_fields__", ()):
                    cls.__validators__.setdefault(name, []).append(f)
                if getattr(f, "__validated_root__", False) and f not in cls.__root_validators__:
                    cls.__root_validators__.append(f)

    def __init__(self, **data):
        values, _, error = validate_model(type(self), data)
        if error:
            raise error
        self.__dict__.update(values)

    @classmethod
    def construct(cls, **values):
        obj = cls.__new__(cls)
        filled = {k: cls.__defaults__.get(k) for k in cls.__fields__}
        filled.update(values)
        obj.__dict__.update(filled)
        return obj

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"


def validate_model(model: type, data: dict):
    values, errors = {}, []
    for name, tp in model.__fields__.items():
        if name in data:
            v = data[name]
        elif name in model.__defaults__:
            v = model.__defaults__[name]
        elif _admits_none(tp):
            v = None
        else:
            errors.append((name, ValueError("field required")))
            continue
        try:
            v = _coerce(tp, v)
            for fn in model.__validators__.get(name, ()):
                v = fn(model, v)
        except (ValueError, TypeError, AssertionError) as e:
            errors.append((name, e))
            continue
        values[name] = v
    for fn in model.__root_validators__:
        try:
            values = fn(model, values)
        except (ValueError, TypeError, AssertionError) as e:
            errors.append(("__root__", e))
    return values, set(data) & set(model.__fields__), (ValidationError(errors, model) if errors else None)
