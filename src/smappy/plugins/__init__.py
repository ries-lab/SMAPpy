"""Plugins: a settings dataclass, a ``run``, and a place in the menu tree.

A plugin is what the GUI shows as one section and what a script calls as one
function.  Nothing here knows about Qt; :mod:`smappy.gui` reads the same
declarations to build its widgets.
"""
from __future__ import annotations

import dataclasses
import importlib
import typing
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from ..locs import Localizations

_META = "smappy"


@dataclass
class ParamInfo:
    """How a settings field is presented.  Everything is optional."""
    label: Optional[str] = None
    unit: Optional[str] = None
    help: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[Sequence] = None
    advanced: bool = False      # hidden behind "more" unless the plugin lists it
    hidden: bool = False        # scripting only


def param(default: Any = MISSING, *, default_factory: Any = MISSING, **info):
    """A dataclass ``field`` carrying :class:`ParamInfo` for the GUI."""
    metadata = {_META: ParamInfo(**info)}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def _unwrap_optional(tp) -> Tuple[Any, bool]:
    """``Optional[int]`` -> ``(int, True)``; anything else -> ``(tp, False)``."""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


@dataclass
class ParamSpec:
    """One settings field, resolved: its type, default and presentation."""
    name: str
    type: Any
    optional: bool          # may be None ("auto")
    default: Any
    info: ParamInfo


def param_specs(settings_cls: type, extra: Optional[Dict[str, ParamInfo]] = None
                ) -> Dict[str, ParamSpec]:
    """Every field of a settings dataclass with its presentation.

    ``extra`` supplies infos for fields that were not declared with `param`,
    so a plain dataclass such as `DriftSettings` needs no changes.
    """
    hints = typing.get_type_hints(settings_cls)
    specs = {}
    for f in fields(settings_cls):
        if not f.init:
            continue
        tp, optional = _unwrap_optional(hints.get(f.name, f.type))
        default = (f.default if f.default is not MISSING
                   else f.default_factory() if f.default_factory is not MISSING
                   else None)
        info = (extra or {}).get(f.name) or f.metadata.get(_META) or ParamInfo()
        specs[f.name] = ParamSpec(f.name, tp, optional, default, info)
    return specs


class Selection:
    """Which localizations a plugin should *look at*.

    The plugin still receives the whole table, because most of them write
    back to all of it.  ``mask`` is boolean over the table; ``layer`` and
    ``roi`` say where it came from, for the history and for plugins that care.
    """

    def __init__(self, mask: np.ndarray, layer: int = 0, roi=None, name: str = ""):
        self.mask = np.asarray(mask, dtype=bool)
        self.layer = layer
        self.roi = roi
        self.name = name

    @classmethod
    def all(cls, n: int) -> "Selection":
        return cls(np.ones(n, dtype=bool), name="all")

    @classmethod
    def from_filter(cls, filter, layer: int = 0) -> "Selection":
        return cls(filter.mask, layer=layer, name=str(filter))

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    def __len__(self) -> int:
        return int(self.mask.sum())

    def apply(self, locs: Localizations) -> Localizations:
        return locs[self.mask]

    def __str__(self) -> str:
        return f"{len(self)} localizations ({self.name or 'selection'})"


@dataclass
class Result:
    """What a plugin hands back.  Every part is optional."""
    locs: Optional[Localizations] = None    # replaces the session's table
    text: str = ""                          # shown, and logged
    plot: Optional[Callable] = None         # plot(ax) draws into a matplotlib axis
    data: Dict[str, Any] = field(default_factory=dict)  # anything else
    settings: Any = None                    # what was actually used


class Plugin:
    """Base class.  Subclasses set ``Settings`` and implement ``run``."""

    name: str = ""                 # set by `register` from the path's last part
    path: str = ""                 # "Analysis/Drift/COMET"
    description: str = ""
    Settings: type = None
    # presentation for fields that were not declared with `param`
    params: Dict[str, ParamInfo] = {}
    # fields shown by default; everything else goes under "more".  None: all
    # fields not marked advanced.
    main: Optional[Sequence[str]] = None

    def run(self, locs: Localizations, selection: Selection, settings,
            progress: Optional[Callable[[str], None]] = None) -> Result:
        raise NotImplementedError

    def __call__(self, locs: Localizations, selection: Optional[Selection] = None,
                 settings=None, **overrides) -> Result:
        """The scripting entry: ``plugin(locs, sel, radius_nm=30)``."""
        if settings is None:
            settings = self.Settings() if self.Settings else None
        if overrides:
            settings = dataclasses.replace(settings, **overrides)
        if selection is None:
            selection = Selection.all(len(locs))
        return self.run(locs, selection, settings)

    @classmethod
    def specs(cls) -> Dict[str, ParamSpec]:
        if cls.Settings is None:
            return {}
        specs = param_specs(cls.Settings, cls.params)
        if cls.main is not None:
            for name, spec in specs.items():
                spec.info.advanced = name not in cls.main
        return specs


# ------------------------------------------------------------------ registry

_REGISTRY: Dict[str, Type[Plugin]] = {}
_BUILTIN = ("smappy.plugins.drift_comet",)


def register(path: str):
    """Class decorator: put a plugin at ``"Tab/Group/Name"`` in the tree."""
    def wrap(cls: Type[Plugin]) -> Type[Plugin]:
        cls.path = path
        cls.name = cls.name or path.rsplit("/", 1)[-1]
        _REGISTRY[path] = cls
        return cls
    return wrap


def load_builtin() -> None:
    for module in _BUILTIN:
        importlib.import_module(module)


def available(prefix: str = "") -> Dict[str, Type[Plugin]]:
    load_builtin()
    return {p: c for p, c in sorted(_REGISTRY.items()) if p.startswith(prefix)}


def get(path: str) -> Type[Plugin]:
    load_builtin()
    return _REGISTRY[path]


def tree() -> Dict[str, Any]:
    """The registry as nested dicts, leaves being plugin classes."""
    out: Dict[str, Any] = {}
    for path, cls in available().items():
        node = out
        *groups, leaf = path.split("/")
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = cls
    return out
