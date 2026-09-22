"""Plugins: a settings dataclass, a ``run``, and a place in the menu tree.

A plugin is what the GUI shows as one section and what a script calls as one
function.  Nothing here knows about Qt; :mod:`smappy.gui` reads the same
declarations to build its widgets.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing
from dataclasses import MISSING, dataclass, field, fields, replace
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
    # "open_file", "save_file", "dir": a path with a browse button.
    # "text": a line that is given the width of the form, for something
    # written rather than dialled -- an expression, a list of names
    kind: Optional[str] = None
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


def settings_values(settings) -> Dict[str, Any]:
    """A settings dataclass as a flat map of dotted names.

    The same shape a form saves, so a workspace, a pipeline file and a plugin
    panel all speak one language.
    """
    out: Dict[str, Any] = {}
    if not dataclasses.is_dataclass(settings):
        return out
    for f in fields(settings):
        if not f.init:
            continue
        value = getattr(settings, f.name)
        if dataclasses.is_dataclass(value):
            out.update({f"{f.name}.{k}": v for k, v in settings_values(value).items()})
        else:
            out[f.name] = value
    return out


def settings_from(settings_cls: Optional[type], values: Optional[Dict[str, Any]] = None):
    """Build settings from a flat dotted map, tolerantly.

    Saved values outlive the plugin that wrote them, so a name the plugin no
    longer has is dropped and a name it has gained keeps its default: a renamed
    field must cost that field, not the whole pipeline.
    """
    if settings_cls is None or not dataclasses.is_dataclass(settings_cls):
        return None
    nested: Dict[str, Dict[str, Any]] = {}
    flat: Dict[str, Any] = {}
    for key, value in (values or {}).items():
        head, dot, rest = key.partition(".")
        if dot:
            nested.setdefault(head, {})[rest] = value
        else:
            flat[head] = value
    kwargs = {}
    specs = param_specs(settings_cls)
    for name, spec in specs.items():
        if spec.children is not None:
            child = settings_from(spec.type, nested.get(name))
            if child is not None:
                kwargs[name] = child
        elif name in flat:
            kwargs[name] = flat[name]
    try:
        return settings_cls(**kwargs)
    except (TypeError, ValueError):
        return settings_cls()          # a value the class refuses costs the map


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
                 site=None, site_table: Optional[Sequence[Dict[str, Any]]] = None,
                 rois=None):
        self.session = session
        self.layer = layer
        self.site = site                    # the ROI, for a scope="site" plugin
        self.site_table = site_table        # the rows evaluation produced
        self._rois = rois                   # a project without a session
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
        """The ROI manager's project: given explicitly, else the session's.

        The evaluation driver passes it directly, because it runs a pipeline
        over a project that may not belong to any session.
        """
        if self._rois is not None:
            return self._rois
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
                       stream=self._stream, site=site, site_table=self.site_table,
                       rois=self._rois)


@dataclass
class Plot:
    """One figure a result offers, and how much room it wants.

    A plugin that draws one thing passes a plain ``plot(ax)`` and never meets
    this class.  One that draws several panels declares how many and is handed
    the figure instead of an axis, because the alternative -- taking
    ``ax.figure``, deleting the axis it was given and repopulating what is
    left -- only works while the figure belongs to nobody else.  It does not
    when the figure is a page of small multiples, where each plot gets a
    *subfigure*: same `subplots` call, a piece of a page rather than a window.

    So the two rules are: say how many panels you draw, and never set the
    figure's size or its layout engine -- `size` is the hint for that, in
    inches, and the window it lands in decides what to do with it.
    """
    draw: Callable                     # draw(ax), or draw(figure) when panels > 1
    name: str = ""                     # "" is the main figure
    panels: int = 1                    # how many axes `draw` makes
    size: Optional[Tuple[float, float]] = None   # width, height in inches

    def draw_into(self, figure) -> None:
        """Draw into a cleared figure, or into a subfigure of a page."""
        if self.panels <= 1:
            self.draw(figure.subplots())
        else:
            self.draw(figure)


def _as_plot(plot, name: str = "") -> Plot:
    """A `Plot`, or the plain callable that means a one-axis plot."""
    if isinstance(plot, Plot):
        return plot if plot.name == name or not name else replace(plot, name=name)
    return Plot(draw=plot, name=name)


@dataclass
class PreflightChoice:
    """One way out of a preflight question, and what taking it runs.

    ``settings`` is what the run uses when this is chosen -- normally the ones
    the form holds with a few fields replaced, which is how a question can
    offer a cheaper way of doing the same thing rather than only a yes and a
    no.  None runs what was asked for.
    """
    label: str
    settings: Any = None
    help: str = ""


@dataclass
class PreflightQuestion:
    """A preflight question with more than two answers.

    `Plugin.preflight` may return a plain string, which is the yes-or-no it
    always was.  This is for the case where the honest answer to "this will
    take an afternoon" is neither yes nor no but "not like that": the choices
    are offered beside the plain yes, each carrying the settings it would run
    with, and cancelling is always there.
    """
    text: str
    choices: Sequence[PreflightChoice] = ()
    run_label: str = "Run anyway"


@dataclass
class Result:
    """What a plugin hands back.  Every part is optional."""
    locs: Optional[Localizations] = None    # replaces the session's table
    # loaded files, as (locs, info, grouped): a loader runs in a worker thread
    # and must not touch the session, so it hands them back and `Session.apply`
    # adds them on the thread that owns the session.
    files: Sequence = ()
    text: str = ""                          # shown, and logged
    # plot(ax) draws into a matplotlib axis; a `Plot` says more than that
    plot: Optional[Any] = None
    # further figures, by name: one plugin may have more than one thing to
    # show, and two views of the same decision belong in two tabs rather
    # than in one crowded axis
    plots: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)  # anything else
    settings: Any = None                    # what was actually used

    def figures(self) -> List[Plot]:
        """Everything there is to draw, as `Plot`s, the main one first."""
        found = [_as_plot(self.plot)] if self.plot is not None else []
        return found + [_as_plot(draw, name) for name, draw in self.plots.items()
                        if draw is not None]


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
    # what the Preview button's tooltip says, when the generic sentence is
    # not enough; empty means the generic one
    preview_help: str = ""
    # Whether a run of this plugin goes into the session's log.  `None` -- the
    # default -- means *if it changed something*: the log is the provenance of
    # the localizations, and what it has to carry is what cannot be worked out
    # again from the file in front of you.  A measurement can always be
    # repeated, so logging it records only that somebody looked; a correction
    # cannot, so it must be recorded, with the numbers it used.  What a
    # measurement is worth keeping goes with its *result* instead
    # (`Plugin.keep`, whose settings the session stores beside it).
    #
    # True says log it anyway, for a run that changes something the log cannot
    # see -- writing a file, exporting a picture.  False says never.
    logged: Optional[bool] = None

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

    # a plugin that can show its work before committing to it overrides
    # `preview`; the GUI grows a button for it when it is overridden, and
    # nothing when it is not.  Dropping the `frame` argument says the preview
    # is of the whole selection rather than of one frame -- see
    # `preview_wants_frame` -- and the GUI then asks for no frame number
    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """One frame, drawn rather than saved: is this set up right?

        Same settings and the same code as `run`, on a single frame, returning
        a `Result` that is only looked at -- never applied to the session.
        """
        raise NotImplementedError

    @classmethod
    def has_preview(cls) -> bool:
        return cls.preview is not Plugin.preview

    @classmethod
    def preview_wants_frame(cls) -> bool:
        """Whether the preview is of one *frame*.

        A fit previews frame 17; a plugin that works on the finished table --
        a colour assignment previewing its histogram -- previews the whole
        selection and has no frame to be asked for.  The GUI reads this to
        decide whether the Preview button comes with a frame number.
        """
        if not cls.has_preview():
            return False
        try:
            return "frame" in inspect.signature(cls.preview).parameters
        except (TypeError, ValueError):
            return True

    def preflight(self, ctx: Context, settings):
        """Anything to put to the user before `run` starts, or None to start.

        Returning a string makes the GUI ask it, and run only on a yes; a
        `PreflightQuestion` offers named alternatives beside that yes, each
        with the settings it would run with.  This is called on the thread
        that owns the session, *before* the worker exists, so it has to be
        quick -- it is where a run whose cost can be known in a fraction of a
        second says so, rather than finding out over the following twenty
        minutes.  `ctx.report` from here reaches the log.
        """
        return None

    def keep(self, result: Result) -> Optional[Dict[str, Any]]:
        """What of this result is worth saving with the localization file.

        A tool's figure outlives the run that made it: a drift correction
        subtracts a curve and the corrected table no longer says what the
        curve was, so reopening the file next week and pressing *Plot* should
        still draw it.  Return plain, JSON-able data -- the curve, the few
        numbers behind the figure -- and never the table, which is in the file
        already.  None (the default) keeps nothing, and the plot is then
        offered only for as long as the session that made it lasts.
        """
        return None

    def restore(self, saved: Dict[str, Any]) -> Optional[Result]:
        """A result carrying the figure again, from what `keep` wrote.

        Read tolerantly: a file may have been written by another version, and
        a curve that cannot be read back is a plot that is not offered, never
        a file that fails to open.
        """
        return None

    def hints(self, settings) -> Optional[Dict[str, Any]]:
        """What the settings' *automatic* fields currently resolve to.

        Keyed by dotted name, like `react`, but these are never written into
        the settings: the GUI shows them greyed in the fields that are set to
        auto, so "auto" says what it will do rather than only that it will do
        something.  Return None when nothing can be resolved yet.
        """
        return None

    def active(self, settings) -> Optional[Dict[str, bool]]:
        """Which fields the current settings actually read, by dotted name.

        A plugin with alternative methods has parameters that belong to one of
        them; the GUI greys out the rest, so that a number which does nothing
        does not look like a number which does.  Nothing is changed by this --
        a greyed field keeps its value and still comes back from the form -- so
        it is presentation, and a script may ignore it.  Return None when every
        field is always live.
        """
        return None

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
