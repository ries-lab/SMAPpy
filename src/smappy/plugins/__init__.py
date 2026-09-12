"""Plugins: a settings dataclass, a ``run``, and a place in the menu tree.

A plugin is what the GUI shows as one section and what a script calls as one
function.  Nothing here knows about Qt; :mod:`smappy.gui` reads the same
declarations to build its widgets.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from ..locs import Localizations
from .discovery import Diagnostic, PluginRef, scan

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
    choices: Optional[Sequence] = None   # values, or (value, label) pairs; may be a callable
    advanced: bool = False      # hidden behind "more" unless the plugin lists it
    hidden: bool = False        # scripting only
    kind: Optional[str] = None  # "open_file", "save_file", "dir": a path with a browse button
    file_filter: str = ""       # a Qt-style name filter for those, "TIFF (*.tif)"


def param(default: Any = MISSING, *, default_factory: Any = MISSING, **info):
    """A dataclass ``field`` carrying :class:`ParamInfo` for the GUI."""
    metadata = {_META: ParamInfo(**info)}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


_TEXT_TYPES = {"int": int, "float": float, "str": str, "bool": bool}


def _hint_from_text(annotation):
    """A best effort at a type from a string annotation."""
    if not isinstance(annotation, str):
        return annotation
    text = annotation.replace(" ", "")
    optional = False
    if text.endswith("|None"):
        text, optional = text[:-len("|None")], True
    elif text.startswith("Optional[") and text.endswith("]"):
        text, optional = text[len("Optional["):-1], True
    base = _TEXT_TYPES.get(text, Any)
    return typing.Optional[base] if optional and base is not Any else base


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
    # for a field that is itself a settings dataclass (a *part*): its specs
    children: Optional[Dict[str, "ParamSpec"]] = None


def param_specs(settings_cls: type, extra: Optional[Dict[str, ParamInfo]] = None
                ) -> Dict[str, ParamSpec]:
    """Every field of a settings dataclass with its presentation.

    A plugin need not have settings at all, so ``None`` and anything that is
    not a dataclass come back empty rather than raising.

    ``extra`` supplies infos for fields that were not declared with `param`,
    so a plain dataclass such as `DriftSettings` needs no changes.  A field
    that is itself a dataclass is a part; its own fields come back under
    ``children``, and ``extra`` reaches them with dotted keys, ``"fit.roisize"``.
    """
    if settings_cls is None or not dataclasses.is_dataclass(settings_cls):
        return {}
    try:
        hints = typing.get_type_hints(settings_cls)
    except TypeError:
        # a `X | None` annotation under `from __future__ import annotations`
        # cannot be evaluated before Python 3.10; read what we can from text
        hints = {f.name: _hint_from_text(f.type) for f in fields(settings_cls)}
    extra = extra or {}
    specs = {}
    for f in fields(settings_cls):
        if not f.init:
            continue
        tp, optional = _unwrap_optional(hints.get(f.name, f.type))
        default = (f.default if f.default is not MISSING
                   else f.default_factory() if f.default_factory is not MISSING
                   else None)
        info = extra.get(f.name) or f.metadata.get(_META) or ParamInfo()
        spec = ParamSpec(f.name, tp, optional, default, info)
        if dataclasses.is_dataclass(tp) and isinstance(tp, type):
            prefix = f.name + "."
            below = {k[len(prefix):]: v for k, v in extra.items() if k.startswith(prefix)}
            spec.children = param_specs(tp, below)
            if default is not None:        # the parent's default instance wins
                for child in spec.children.values():
                    child.default = getattr(default, child.name)
        specs[f.name] = spec
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

    def require(self, minimum: int, progress=None, what: str = "this") -> None:
        """Refuse an empty selection; warn about a thin one through ``progress``."""
        n = len(self)
        where = f" in the {self.roi}" if self.roi is not None else ""
        if n == 0:
            raise ValueError(f"no localizations selected{where}: loosen the filter"
                             + (" or clear the ROI" if self.roi is not None else ""))
        if n < minimum and progress:
            progress(f"warning: only {n} localizations{where}; {what} wants "
                     f"at least ~{minimum}")

    def __str__(self) -> str:
        return f"{len(self)} localizations ({self.name or 'selection'})"


class Context:
    """What a plugin is given to work with.

    `run(ctx, settings)` replaced `run(locs, selection, settings, progress,
    stream)` because that signature could only describe a plugin that
    transforms a table.  A loader has no input localizations, a simulator has no
    input at all, and an evaluator runs once per ROI -- all of which fit here
    without a second base class.

    Everything is optional, so a script can say `Context(locs=locs)` and a
    plugin called through `Plugin.__call__` never sees the difference.  When a
    session is given, `locs` and `selection` are read from it *once, now*: the
    GUI builds the context on its own thread and hands it to a worker, so
    reading them later would race with a live fit rebinding the table.
    """

    def __init__(self, session=None, locs: Optional[Localizations] = None,
                 selection: Optional["Selection"] = None, layer: int = 0,
                 progress: Optional[Callable[[str], None]] = None,
                 stream: Optional[Callable[[str, Any], None]] = None,
                 site=None, site_table: Optional[Sequence[Dict[str, Any]]] = None):
        self.session = session
        self.layer = layer
        self.site = site                    # the ROI, for a scope="site" plugin
        self.site_table = site_table        # the rows evaluation produced
        self._progress = progress
        self._stream = stream
        if locs is None:
            locs = session.locs if session is not None else Localizations({}, {})
        self.locs = locs
        if selection is None:
            selection = (session.selection(layer) if session is not None
                         else Selection.all(len(locs)))
        self.selection = selection

    @property
    def rois(self):
        """The ROI manager's project, or None outside a session."""
        return None if self.session is None else self.session.rois

    def report(self, text: str) -> None:
        """Say what is happening.  A no-op when nobody is listening."""
        if self._progress:
            self._progress(text)

    def emit(self, event: str, payload: Any) -> None:
        """Hand a partial result on while still running.

        ``("start", {"extent": ...})`` once, then ``("block", Localizations)``
        per finished block, for a plugin that produces localizations over
        minutes.  A no-op when nobody is listening.
        """
        if self._stream:
            self._stream(event, payload)

    def for_site(self, site, locs: Optional[Localizations] = None,
                 selection: Optional["Selection"] = None) -> "Context":
        """This context aimed at one ROI, for a ``scope = "site"`` plugin."""
        return Context(session=self.session,
                       locs=self.locs if locs is None else locs,
                       selection=self.selection if selection is None else selection,
                       layer=self.layer, progress=self._progress,
                       stream=self._stream, site=site, site_table=self.site_table)


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
    favorite: bool = True          # pinned in the shipped workspace
    # "locs": run once over the selection.  "site": run once per ROI, with
    # `ctx.site` set -- what makes a plugin an ROI evaluator.  It is a property
    # of the plugin, not of its folder, which is what lets the evaluation
    # window filter correctly when every plugin lives in one tree.
    scope: str = "locs"
    Settings: type = None
    # presentation for fields that were not declared with `param`
    params: Dict[str, ParamInfo] = {}
    # fields shown by default; everything else goes under "more".  None: all
    # fields not marked advanced.
    main: Optional[Sequence[str]] = None

    def __init_subclass__(cls, **kwargs):
        """Catch a plugin written against the old signature with a real error.

        `run(locs, selection, settings, ...)` would otherwise be handed a
        `Context` as its `locs` and the settings as its `selection`, and fail
        somewhere far away.
        """
        super().__init_subclass__(**kwargs)
        run = cls.__dict__.get("run")
        if run is None:
            return
        try:
            names = list(inspect.signature(run).parameters)
        except (TypeError, ValueError):
            return
        if len(names) > 1 and names[1] in ("locs", "localizations"):
            raise TypeError(
                f"{cls.__name__}.run takes {names[1]!r}: plugins now take a "
                "Context, `def run(self, ctx, settings)`.  ctx.locs and "
                "ctx.selection replace the first two arguments, ctx.report "
                "replaces progress, ctx.emit replaces stream.")

    def run(self, ctx: Context, settings) -> Result:
        """Do the work.

        `ctx` carries the localizations, the selection, the session and the
        ROI, and `ctx.report(text)` / `ctx.emit(event, payload)` report progress
        and hand partial results on.  See `Context`.
        """
        raise NotImplementedError

    def react(self, changed: str, settings) -> Optional[Dict[str, Any]]:
        """A parameter was edited: ``changed`` is its dotted name.

        Return values to set in reply, keyed by dotted name -- a camera read
        from the file that was just chosen, say -- or None.  The GUI calls this;
        a script may too."""
        return None

    def __call__(self, locs: Optional[Localizations] = None,
                 selection: Optional[Selection] = None,
                 settings=None, *, ctx: Optional[Context] = None,
                 progress: Optional[Callable[[str], None]] = None,
                 **overrides) -> Result:
        """The scripting entry: ``plugin(locs, sel, radius_nm=30)``.

        `locs` may be omitted entirely for a plugin that makes its own -- a
        loader or the simulator.
        """
        if settings is None:
            settings = self.Settings() if self.Settings else None
        if overrides:
            settings = dataclasses.replace(settings, **overrides)
        if ctx is None:
            ctx = Context(locs=locs, selection=selection, progress=progress)
        return self.run(ctx, settings)

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
#
# Two halves.  `_REGISTRY` is filled by `@register` when a module is imported;
# `_REFS` is filled by the scanner, which reads the files without importing
# them (see `discovery`).  The refs are what the GUI builds its tree from, so
# starting up costs one parse per plugin and no imports at all; `get` and
# `available` import on demand and are what a script uses.

_REGISTRY: Dict[str, Type[Plugin]] = {}
_REFS: Dict[str, PluginRef] = {}
_PROBLEMS: List[Diagnostic] = []
_SCANNED = False

BUILTIN_ROOT = Path(__file__).resolve().parent


def register(path: str):
    """Class decorator: put a plugin at ``"Tab/Group/Name"`` in the tree."""
    def wrap(cls: Type[Plugin]) -> Type[Plugin]:
        cls.path = path
        cls.name = cls.name or path.rsplit("/", 1)[-1]
        _REGISTRY[path] = cls
        return cls
    return wrap


def roots() -> List[Tuple[str, Path]]:
    """Where to look, in priority order: the shipped plugins, then the user's.

    A later root wins a collision, so a user can shadow a shipped plugin by
    putting their own at the same path.
    """
    from .. import config
    found = [("builtin", BUILTIN_ROOT)]
    for directory in config.plugin_roots():
        found.append((str(directory), directory))
    return found


def discover(force: bool = False) -> Dict[str, PluginRef]:
    """Scan the roots.  Cheap, and idempotent unless ``force``."""
    global _SCANNED, _REFS, _PROBLEMS
    if _SCANNED and not force:
        return _REFS
    _REFS, _PROBLEMS = scan(roots())
    _SCANNED = True
    return _REFS


def refs(prefix: str = "") -> Dict[str, PluginRef]:
    """Every plugin under ``prefix``, as refs -- nothing is imported."""
    return {p: r for p, r in discover().items() if p.startswith(prefix)}


def problems() -> List[Diagnostic]:
    """Files that should have held a plugin and did not.  For the GUI to show."""
    discover()
    return list(_PROBLEMS)


def get(path: str) -> Type[Plugin]:
    """The plugin class at ``path``, importing its module the first time."""
    if path in _REGISTRY:
        return _REGISTRY[path]
    ref = discover().get(path)
    if ref is None:
        raise KeyError(path)
    return ref.load()


def available(prefix: str = "") -> Dict[str, Type[Plugin]]:
    """Every plugin under ``prefix``, as classes -- so this imports them all.

    Prefer `refs` for anything that only needs to *show* the plugins.
    """
    return {p: get(p) for p in sorted(refs(prefix))}


def tree(prefix: str = "") -> Dict[str, Any]:
    """The registry as nested dicts, the leaves being `PluginRef`s."""
    out: Dict[str, Any] = {}
    for path, ref in refs(prefix).items():
        node = out
        *groups, leaf = path.split("/")
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = ref
    return out
