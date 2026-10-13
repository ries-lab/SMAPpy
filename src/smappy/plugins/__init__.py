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
from ..params import _META, ParamInfo, param  # noqa: F401  (re-exported)
from .discovery import Diagnostic, PluginRef, scan

def settings_hints(settings_cls: type) -> Dict[str, Any]:
    """`typing.get_type_hints`, with the one failure a plugin author meets
    explained: under ``from __future__ import annotations``, Python 3.9
    cannot evaluate ``X | None``, and a form built from guesses at the text
    would be wrong in ways nobody sees."""
    try:
        return typing.get_type_hints(settings_cls)
    except TypeError as exc:
        raise TypeError(f"{settings_cls.__name__}: its annotations cannot be "
                        f"evaluated ({exc}); on Python 3.9 write Optional[X] "
                        "rather than X | None") from exc


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


def param_specs(settings_cls: type) -> Dict[str, ParamSpec]:
    """Every field of a settings dataclass with its presentation.

    A plugin need not have settings at all, so ``None`` and anything that is
    not a dataclass come back empty rather than raising.

    How a field is shown is declared on the field, with `param` -- the one
    way, so a settings class shown by two plugins (`DriftSettings` in COMET
    and in a fitter's *after the fit*) is shown the same in both.  A field
    that is itself a dataclass is a part; its own fields come back under
    ``children``.
    """
    if settings_cls is None or not dataclasses.is_dataclass(settings_cls):
        return {}
    hints = settings_hints(settings_cls)
    specs = {}
    for f in fields(settings_cls):
        if not f.init:
            continue
        tp, optional = _unwrap_optional(hints.get(f.name, f.type))
        default = (f.default if f.default is not MISSING
                   else f.default_factory() if f.default_factory is not MISSING
                   else None)
        info = f.metadata.get(_META) or ParamInfo()
        spec = ParamSpec(f.name, tp, optional, default, info)
        if dataclasses.is_dataclass(tp) and isinstance(tp, type):
            spec.children = param_specs(tp)
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

    A part may also come as a nested dictionary, ``{"fit": {"roisize": 13}}``:
    that is how the session's log has always written settings (`asdict`), and
    reading it back is what makes a logged run one that can be repeated.
    """
    if settings_cls is None or not dataclasses.is_dataclass(settings_cls):
        return None
    return _from_specs(settings_cls, param_specs(settings_cls), values)


def _from_specs(settings_cls: type, specs: Dict[str, "ParamSpec"],
                values: Optional[Dict[str, Any]]):
    """`settings_from` over specs whose defaults are already resolved.

    A part's defaults are its *parent's* -- a settings class may declare
    ``part: PartSettings = field(default_factory=lambda: PartSettings(n=3))``
    -- and `param_specs` carries them down; building a part from its own
    class instead quietly reset them, so the form and a script made
    different settings from the same values.
    """
    nested: Dict[str, Dict[str, Any]] = {}
    flat: Dict[str, Any] = {}
    for key, value in (values or {}).items():
        head, dot, rest = key.partition(".")
        if dot:
            nested.setdefault(head, {})[rest] = value
        else:
            flat[head] = value
    kwargs = {}
    for name, spec in specs.items():
        if spec.children is not None:
            if isinstance(flat.get(name), dict):
                # the nested spelling; a dotted key given as well wins
                nested[name] = {**flat[name], **nested.get(name, {})}
            child = _from_specs(spec.type, spec.children, nested.get(name))
            if child is not None:
                kwargs[name] = child
        elif name in flat:
            kwargs[name] = flat[name]
        elif spec.default is not None:
            kwargs[name] = spec.default     # the resolved default, the parent's
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

    def require(self, minimum: int, warn=None, what: str = "this") -> None:
        """Refuse an empty selection; warn about a thin one through ``warn``.

        Pass `Context.warn`, not `Context.report`: a progress line is
        overwritten by the next one, and this one was, before anyone read it.
        """
        n = len(self)
        where = f" in the {self.roi}" if self.roi is not None else ""
        if n == 0:
            raise ValueError(f"no localizations selected{where}: loosen the filter"
                             + (" or clear the ROI" if self.roi is not None else ""))
        if n < minimum and warn:
            warn(f"only {n} localizations{where}; {what} wants "
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

    `locs` and `selection` are the table the layer *shows* -- grouped or not
    -- unless `grouping` says which (the plugin's requirement, `Plugin.grouping`,
    or a chain's choice): a measurement measures what is on screen.  A plugin
    that hands back a new table declares ``grouping = "ungrouped"``, since
    there is no way from one row per blink back to the localizations; one
    that measures blinks and writes onto the localizations reads both, the
    second through `ungrouped` (DBSCAN).
    """

    def __init__(self, session=None, locs: Optional[Localizations] = None,
                 selection: Optional["Selection"] = None, layer: int = 0,
                 progress: Optional[Callable[[str], None]] = None,
                 stream: Optional[Callable[[str, Any], None]] = None,
                 region=None, site_table: Optional[Sequence[Dict[str, Any]]] = None,
                 rois=None, grouping: Optional[str] = None,
                 evaluations: Sequence[Tuple[str, str]] = (),
                 stop=None, writer_finished=None, pinned=None, rows=None,
                 region_id: Optional[str] = None):
        self.session = session
        # layer -> {"ungrouped"/"grouped": (locs, selection)}, taken when the
        # session's table was still being appended to (`Session.context`), or
        # cut to one region (`ROIProject.region_context`): then `table` and
        # `layer_selection` answer from these and never read the session's
        # table.  An entry may say which of the two it ``"shows"``, for a
        # context with no session to ask.
        self._pinned = pinned
        # a region context's rows of the session's table, for a selection
        # `layer_selection` has to build afresh (``without`` a filter)
        self._rows = rows
        self.layer = layer
        # "grouped", "ungrouped", or None for each layer's own: what a chain
        # (or the plugin's own requirement) has decided `table` should hand
        # out.  See `table`.
        self.grouping = grouping
        # whether `table` has handed out a grouped table: a chain refuses a
        # new table from a step that worked on one row per blink
        self.served_grouped = False
        # the region this run is confined to (`regions.Region`): an ROI of
        # the ROI manager, or None for a cluster or a run over the view.  In
        # a region context `locs` is the region's rows, and every layer's
        # selection is that layer's filter over them (`docs/regions.md`).
        self.region = region
        # which region, when this run is one of a run over many (an ROI's id,
        # a cluster's): set for a cluster too, which has no shape
        self.region_id = region_id
        self.site_table = site_table        # the rows evaluation produced
        # (name, evaluator path) of the evaluations a chain has made before
        # this step, oldest first, each name as *results from* spells it
        # (`region_results.Evaluation`): an analysis of regions in a chain
        # reads its own chain's results unless told otherwise
        # (`roi.choose_evaluation`)
        self.evaluations = list(evaluations)
        self._rois = rois                   # a project without a session
        # Two `threading.Event`s for a run that lasts as long as an
        # acquisition, set from another thread; None when nobody can.
        # `stop` asks it to end early and keep what it has -- a fit then
        # closes its file and returns the table so far; the panel's Stop
        # button sets it (`stopped`, `Stopped`).  `writer_finished`
        # says a live source is complete, which ends the watch on it after
        # one more read instead of after an idle timeout (`smappy.io.watch`).
        self.stop = stop
        self.writer_finished = writer_finished
        self._progress = progress
        self._stream = stream
        self.warnings: List[str] = []       # what `warn` was told, in order
        given = locs is not None
        if locs is None:
            locs = session.locs if session is not None else Localizations({}, {})
        if selection is None:
            selection = (session.selection(layer) if session is not None
                         else Selection.all(len(locs)))
        # the ungrouped table and its selection, whatever `locs` is
        self._ungrouped = (locs, selection)
        self.locs, self.selection = locs, selection
        if grouping != "ungrouped" and (pinned is not None
                                        or (session is not None and not given)):
            self.locs, self.selection = self._shown()

    def _shown(self) -> Tuple[Localizations, "Selection"]:
        """The table this run works on, by the rule of `table`, for `locs`.

        Where the layer's own choice is grouped but no grouped table can be
        had -- one cut to a region while the fit is still adding to it, or
        a cluster of a table linked before the cluster column was -- the
        ungrouped table is the answer rather than a refusal: every
        measurement would otherwise fail in such a region.  Asked for
        explicitly (a chain step set to grouped), `table` refuses.
        """
        try:
            return self.table()
        except ValueError:
            if self.grouping == "grouped":
                raise
            self.served_grouped = False
            return self._ungrouped

    def ungrouped(self) -> Tuple[Localizations, "Selection"]:
        """The ungrouped table and the layer's selection over it.

        `locs` when the plugin asks for it (``grouping = "ungrouped"``).  For
        a plugin that follows the layer and still writes onto the
        localizations: it measures `locs` -- blinks, on a grouped layer --
        and hands back a new table made from this one (DBSCAN maps each
        blink's cluster onto its localizations through `group_id`).
        """
        return self._ungrouped

    @property
    def per_region(self) -> bool:
        """Whether this is one region of a run over many -- an ROI or a
        cluster -- rather than a run on the view.  A plugin may then make an
        empty region an answer, or leave out what only the whole field has."""
        return self.region_id is not None

    @property
    def stopped(self) -> bool:
        """Whether the run has been asked to stop (`stop`).

        A plugin checks it between steps of its own -- time windows,
        regions, blocks of frames -- and then either returns what it has,
        with ``Result(stopped=True)`` so the panel says "stopped", or
        raises `Stopped` when a part of the work is worth nothing (half a
        drift estimate).  One that never checks simply runs to the end.
        """
        return self.stop is not None and self.stop.is_set()

    @property
    def rois(self):
        """The ROI manager's project: given explicitly, else the session's.

        The evaluation driver passes it directly, because it runs a pipeline
        over a project that may not belong to any session.
        """
        if self._rois is not None:
            return self._rois
        return None if self.session is None else self.session.rois

    def table(self, layer: Optional[int] = None
              ) -> Tuple[Localizations, "Selection"]:
        """The table to work on and what is selected in it: ``(locs, selection)``.

        Grouped or not, as decided -- lowest priority first -- by the layer's
        own grouping, the chain's, the chain step's, and the plugin's
        requirement (`Plugin.grouping`); `grouping` here is the outcome of all
        but the first, None leaving it to the layer.  This is SMAP's
        ``getloc(..., 'grouping', ...)``.  For the context's own layer it is
        `locs` and `selection`; this is how a plugin reads another layer.

        Without a session there is no grouped table to hand out, and linking
        one here with settings nobody chose would be a guess; that is refused
        with a message rather than answered with the ungrouped table.
        """
        index = self.layer if layer is None else layer
        if self._pinned is not None:
            return self._pinned_table(index)
        if self.session is None:
            if self.grouping == "grouped":
                raise ValueError("merged blinks need a session: merge the "
                                 "localizations first (group.group), or run "
                                 "this on the localizations")
            return self.locs, self.selection
        locs, selection = self.session.table(
            index, self.grouping,
            progress=lambda text, _f: self.report(f"merging blinks: {text}"))
        if locs is not self.session.locs:
            self.served_grouped = True
        return locs, selection

    def _pinned_table(self, index: int) -> Tuple[Localizations, "Selection"]:
        session = self.session
        if session is not None and session.layers[index].is_image:
            index = session.first_locs_layer()
        if index not in self._pinned:
            raise ValueError(f"there is no layer {index + 1} here")
        tables = self._pinned[index]
        if self.grouping is not None:
            grouped = self.grouping == "grouped"
        elif "shows" in tables:
            grouped = tables["shows"] == "grouped"
        else:
            grouped = session.layers[index].grouped
        if not grouped:
            return tables["ungrouped"]
        if "grouped" not in tables:
            if self.region is not None:
                raise ValueError(f"layer {index + 1} has no merged blinks to cut "
                                 "to the region: tick merge blinks first, or run "
                                 "this on the localizations")
            raise ValueError("no merged blinks while the fit is still adding "
                             "localizations: tick merge blinks on the layer "
                             "first, or run this on the localizations")
        self.served_grouped = True
        return tables["grouped"]

    def layer_selection(self, layer: int,
                        without: Tuple[str, ...] = ()) -> "Selection":
        """Layer ``layer``'s selection over the ungrouped table (`ungrouped`).

        `Session.selection`, except that it agrees with `locs` while a fit is
        still appending: a plugin that reads several layers -- one answer per
        layer -- asks here rather than the session, whose table has grown
        since this context was made.
        """
        if self.session is None:
            if layer != self.layer or without:
                raise ValueError("other layers need a session")
            return self.selection
        if self._pinned is None:
            return self.session.selection(layer, without)
        index = (self.session.first_locs_layer()
                 if self.session.layers[layer].is_image else layer)
        if not without:
            if index not in self._pinned:
                raise ValueError(f"there is no layer {layer + 1} here")
            return self._pinned[index]["ungrouped"][1]
        if self._rows is not None:
            # a region: the layer's own filter without those, over its rows --
            # not the session's selection, which is cut to the drawn ROI
            sel = self.session.layers[index].selection(index, without)
            sel.mask = np.asarray(sel.mask, dtype=bool)[self._rows]
            sel.roi = self.region
            sel.name = f"{sel.name}, {self.region or self.region_id}"
            return sel
        sel = self.session.selection(index, without)
        sel.mask = sel.mask[:len(self.locs)]      # rows are only ever appended
        return sel

    def report(self, text: str) -> None:
        """Say what is happening.  A no-op when nobody is listening."""
        if self._progress:
            self._progress(text)

    def warn(self, text: str) -> None:
        """Say something the user must not miss, and keep it.

        A progress line is overwritten by the next one, which is how an EM-gain
        mismatch reported at the start of a fit vanished under its frame
        count.  A warning goes to whoever listens as a ``("warning", text)``
        event -- the panel keeps it as a line of its own -- or as a progress
        line when only progress is listened to, and it stays in `warnings`
        for the plugin to put in what it returns.
        """
        self.warnings.append(text)
        if self._stream:
            self._stream("warning", text)
        elif self._progress:
            self._progress(f"warning: {text}")

    def emit(self, event: str, payload: Any) -> None:
        """Hand a partial result on while still running.

        ``("start", {"extent": ...})`` once, then ``("block", Localizations)``
        per finished block, for a plugin that produces localizations over
        minutes.  A no-op when nobody is listening.
        """
        if self._stream:
            self._stream(event, payload)


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


# Where the plugins that make a table of their own sit -- from a file, from
# images or from nothing: they never read the one there is.
SOURCES = ("Localize/", "File/Load/", "File/Simulate/")


def refuse_grouped_table(ctx: "Context", result: "Result", plugin: "Plugin",
                         name: str = "") -> None:
    """Refuse a new table made from the grouped one.

    A plugin that follows the layer reads blinks on a grouped layer, and a
    table it hands back made from them would replace the localizations with
    one row per blink -- the data gone, with an undo step as the only way
    back.  A plugin that changes the table declares ``grouping =
    "ungrouped"``; one written before plugins followed the layer is told so
    here rather than allowed to do that.  A fit or a loader makes its own
    table and is not asked.
    """
    if (result is None or result.locs is None or not ctx.served_grouped
            or str(getattr(plugin, "path", "")).startswith(SOURCES)):
        return
    name = name or getattr(plugin, "name", "") or type(plugin).__name__
    raise ValueError(
        f"{name} ran on merged blinks and handed back a new table: "
        "there is no way from one row per blink back to the localizations, "
        "so run it with blinks not merged (a plugin that changes the table declares "
        "grouping = \"ungrouped\")")


class Stopped(Exception):
    """A run that was asked to stop (`Context.stop`) and has nothing worth
    keeping: the session is left as it was, and the panel says "stopped"
    rather than "failed".  A run that has something -- the frames fitted, the
    regions measured -- returns it instead, with ``Result(stopped=True)``."""


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
    # a window of the GUI's (`Session.window_openers`) that this choice opens
    # *instead of* running, with ``settings`` (None: the form's): the way to
    # put right what the question is about -- a camera to add to the
    # database -- rather than a way to run.  Never the dialog's default.
    window: Optional[str] = None


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
    # empty: no plain run is offered -- the question is which, not whether,
    # and the GUI asks again once one is chosen, for the next such question
    run_label: str = "Run anyway"


@dataclass
class Notice:
    """Something about the settings as they stand that the user should see
    before running: a line under the form, and a button that does something
    about it.

    `window` names a window of the GUI's (`Session.window_openers`), opened
    with the form's settings, as `Plugin.window` does -- so the plugin says
    what is wrong and where it is put right without importing Qt.  A GUI with
    no such window shows the line and no button.

    ``kind`` is how loudly: "warning" (orange, bold) for something that will
    go wrong or be guessed if left as it is -- a camera the database does not
    know -- and "info" (grey) for something worth knowing that is not a
    fault, such as what a save leaves out by choice.  Drawing every line as
    a warning taught people to stop reading them.
    """
    text: str
    action: str = ""                 # the button's text
    window: Optional[str] = None
    # what the window is opened with; None: the form's settings.  A chain's
    # step says its own here, since the window wants a fitter's settings and
    # the form holds the chain's.
    settings: Any = None
    kind: str = "warning"            # or "info"


@dataclass
class FormState:
    """What the form shows about the settings as they stand, beyond their
    values: one answer to one question (`Plugin.describe`), asked after every
    edit and whenever the session changes what a list offers.

    Every part is presentation: nothing here is written into the settings,
    and a script may ignore all of it.  Keys are dotted field names, as in
    `Plugin.react`.

    * ``hints`` -- what fields left on *auto* currently resolve to, shown
      greyed in them, so "auto" says what it will do.
    * ``active`` -- which fields the settings actually read; the rest are
      greyed out (a greyed field keeps its value), so that a number which
      does nothing does not look like one which does.
    * ``choices`` -- ``[(value, label), ...]`` for fields whose list depends
      on the session: the evaluations on the ROIs, the cameras in the
      database.  The field declares a callable ``choices`` for a form with no
      session; a value the list does not have is kept and shown as
      unavailable, never swapped for the first entry.
    * ``notice`` -- a line under the form, with a button that does something
      about it (`Notice`).
    * ``titles`` -- a part's title, in place of its label: a summary of what
      it holds, for a part that starts folded.
    """
    hints: Dict[str, Any] = field(default_factory=dict)
    active: Dict[str, bool] = field(default_factory=dict)
    choices: Dict[str, Sequence] = field(default_factory=dict)
    notice: Optional[Notice] = None
    # a part's title by its dotted name, in place of its label: what the part
    # holds, said while it is folded ("camera: auto - iXon 888, ...")
    titles: Dict[str, str] = field(default_factory=dict)

    def under(self, key: str, title: str = "", settings: Any = None) -> "FormState":
        """This state as part ``key`` of a larger form -- a chain's step: the
        fields prefixed, and the notice saying whose it is and opening its
        window with the step's own ``settings``."""
        notice = self.notice
        if notice is not None:
            notice = replace(
                notice, text=f"{title}: {notice.text}" if title else notice.text,
                settings=notice.settings if notice.settings is not None else settings)
        return FormState(
            hints={f"{key}.{k}": v for k, v in self.hints.items()},
            active={f"{key}.{k}": v for k, v in self.active.items()},
            choices={f"{key}.{k}": v for k, v in self.choices.items()},
            notice=notice,
            titles={f"{key}.{k}": v for k, v in self.titles.items()})

    def update(self, other: "FormState") -> None:
        """Add ``other``'s parts; the first notice stays, as the panel has
        room for one line."""
        self.hints.update(other.hints)
        self.active.update(other.active)
        self.choices.update(other.choices)
        if self.notice is None:
            self.notice = other.notice


@dataclass
class Result:
    """What a plugin hands back.  Every part is optional.

    Two kinds of part.  What the run found -- its text, its figures, its
    numbers (`data`, `values`, `arrays`) -- is looked at and kept.  What it
    asks the session to do beyond replacing the table -- open files, set
    filter bounds, add ROIs, keep camera frames -- is a typed field each,
    below the line, which `Session.apply` (or a chain, a block, the batch)
    reads.  They were string keys of `data` once, and a key misspelt or
    renamed on one side did nothing at all; a misspelt keyword now raises a
    `TypeError` where the result is made.
    """
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
    # the plugin's own numbers and payload, for a script or a test: nothing
    # reads a key of it to decide what to do
    data: Dict[str, Any] = field(default_factory=dict)
    # What a run in one region stores as that region's results
    # (`docs/regions.md`): `values` the numbers, a column each of the region
    # table; `arrays` named arrays -- a fixed-size one (eight corner counts)
    # or one of the region's own length.  Only these are kept per region;
    # `data` and the figures are for the run that made them.
    values: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, Any] = field(default_factory=dict)
    settings: Any = None                    # what was actually used
    # more for the log entry, beside the path, the text and the settings: a
    # chain puts its steps here, so the history shows what each one did
    log: Dict[str, Any] = field(default_factory=dict)
    # what the figures are of, for the window's title beside the plugin's
    # name: "NeNA, CRLB, FRC of layer 1, 30000 localizations"
    title: str = ""

    # ---------------------------------------------- what the session is asked
    # The `files` join the session's rather than replacing them (a loader's
    # *append*); the second and later files always join.
    append: bool = False
    # The file this run wrote -- a saved table, an exported picture, a
    # simulation's recipe, a fit's output.  A table made from nothing is
    # named after it, a chain passes it on, and the batch records it.
    written: Optional[str] = None
    # The table handed back is the one in `written`, so the session is not
    # "unsaved" after it: a fit that wrote its own output.
    saved: bool = False
    # Camera frames a fit kept (`smappy.rawframes`): they replace the
    # session's raw frames, as the fit's table replaces its table.
    images: Sequence = ()
    # Filter bounds ``{column: (lo, hi)}`` set on every localization layer:
    # for a column the plugin wrote to be filtered on.  It cannot set them
    # itself, since a filter belongs to the table, and the table is the one
    # `apply` has just put in place.
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Layer set-ups, bounds and grouping per layer
    # (`Session.set_layer_configs`): the Render tab's work done by a plugin
    # (`Chain/Layers`).
    layers: Optional[List[Dict[str, Any]]] = None
    # The ROIs this run added to the ROI manager's project (a segmenter, or
    # *Column to ROIs*), an empty list when it found none; None for a run
    # that adds no ROIs.  The manager redraws, and a preview is dropped.
    rois: Optional[List[Any]] = None
    # A segmenter's preview: the ROIs it would add, shown grey in the
    # manager and not added (`Session.show_preview`).
    roi_preview: Optional[Dict[str, Any]] = None
    # The whole ROI project as a chain left it (`ROIProject.to_dict`): the
    # ROIs its segmenters found and the runs its evaluators made, put into
    # the project the session has.
    roi_project: Optional[Dict[str, Any]] = None
    # What was measured per region of an id column, ``{column: {name:
    # table}}``, added to `Session.id_regions`.
    id_regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # A block's tables per region, which a block around it keeps as the
    # children of its own region (`region_runs.RegionScratch`).
    region_tables: Dict[str, Any] = field(default_factory=dict)
    # ``(name, plugin path)`` of the evaluations this run left on the
    # regions, so that a later step of a chain reads them without asking.
    evaluations: Sequence[Tuple[str, str]] = ()
    # The table change as a recipe (`ColumnRecipe`), which can follow a table
    # a fit is still appending to; `locs` is then the recipe applied.
    recipe: Optional["ColumnRecipe"] = None
    # The run was asked to stop (`Context.stop`) and this is what it had:
    # the panel says "stopped" rather than "done".
    stopped: bool = False

    def figures(self) -> List[Plot]:
        """Everything there is to draw, as `Plot`s, the main one first."""
        found = [_as_plot(self.plot)] if self.plot is not None else []
        return found + [_as_plot(draw, name) for name, draw in self.plots.items()
                        if draw is not None]


class ColumnRecipe:
    """A table change that can follow a fit while it is still appending.

    A table change cannot go into a table a fit is still adding to
    (`Session.apply`) -- unless it is a recipe: columns computed from the
    other columns, which the session can work out for the table so far and
    then for every block as it arrives, and once more for the finished
    table.  A plugin that can be one puts it in its result as
    ``Result.recipe``, and says so before the run in
    `Plugin.column_recipe_for`.  The Math Parser's expression is one, and the
    colour assignment, with the modes it found, is another.

    `columns` must work on any table with the columns it reads -- a block of
    a hundred rows as well as the whole table -- and must not write a
    position: the spatial index of a live table is extended, never rebuilt.
    ``whole`` says the values of earlier rows depend on later ones (a
    reduction, ``z_nm - median(z_nm)``): the column is then recomputed over
    the whole table on every append, not only for the new block.
    """

    name = ""                       # the plugin's, for the log
    whole = False

    def columns(self, locs: Localizations) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def final(self, locs: Localizations) -> Localizations:
        """The finished table with the recipe applied: `columns`, unless the
        recipe has something better to do once the whole table is there."""
        return Localizations({**locs.columns, **self.columns(locs)},
                             dict(locs.metadata))


class Plugin:
    """Base class.  Subclasses set ``Settings`` and implement ``run``."""

    name: str = ""                 # set by `register` from the path's last part
    path: str = ""                 # "Analysis/Drift/COMET"
    description: str = ""
    favorite: bool = True          # pinned in the shipped workspace; for a
                                   # per-region plugin, in the default
                                   # evaluation pipeline
    # Whether `run` hands back `Result.values`, and so can be run once per
    # region and kept as that region's row (`docs/regions.md`).  Not a kind
    # of plugin -- such a plugin runs on the view like any other -- but what
    # the evaluation window and a chain need to know before running it,
    # which is why it is declared rather than found out.
    per_region: bool = False
    Settings: type = None
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
    # Bumped when a change moves the numbers.  A chain records the version of
    # every step it ran, and a batch re-runs a file whose steps' versions have
    # changed and skips one whose have not: a fix that leaves this alone is a
    # fix the old results are taken to agree with.
    version: str = "1"
    # "grouped" or "ungrouped" for a plugin that can only work on one of the
    # two tables; None for one that takes whichever the layer, the chain or
    # the user says (see `Context.table`).  A requirement, not a preference:
    # it wins over every other choice, and the GUI greys the choice out.
    grouping: Optional[str] = None
    # Whether the result's text opens in a window of its own.  A run reports
    # in the panel's output box, where the rest of the GUI's status is, and a
    # window that pops up with a summary in it is one more thing to close.
    # True only for a plugin whose text *is* the result -- the history, a log
    # nobody reads four lines at a time.  The Text button offers the window
    # either way.
    text_window: bool = False
    # Whether `run` hands back a new table (`Result.locs`, or files that
    # replace it): True for a correction, a filter, a new column, a loader or
    # a fit; False for a measurement, an ROI tool or an export.  Known before
    # the run, which is what lets the GUI ask first rather than after minutes
    # of work -- while a fit is still appending, a table change cannot go
    # into the session (`Session.apply`).  Required: a class that does not
    # say is refused when it is defined (`__init_subclass__`).  There used to
    # be a third answer, "not said, ask after the run", and it was a dialog
    # offering to redo minutes of work in another window.  Where the answer
    # depends on the settings, override `changes_locs_for` too.
    changes_locs: bool
    # A window of the GUI's that this plugin opens instead of running: the
    # panel's button becomes *Open* and hands the settings to whatever the
    # GUI registered under this name (`Session.window_openers`).  For a tool
    # that is interactive through and through -- the bead calibration, with
    # its bead table -- and which a form and a Run button cannot be; the
    # plugin is what lets it sit in a tab, and it imports no Qt to do so.
    window: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        # At definition, not at the first run: a dropped-in plugin that does
        # not say then fails to load with the reason, in the list of problems,
        # rather than surprising someone during a fit.  Inherited counts -- the
        # loaders share theirs with their base.
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "changes_locs", None), bool):
            raise TypeError(
                f"{cls.__qualname__} must say whether it changes the table: "
                "set changes_locs = True (it hands back a new table or files) "
                "or False (it measures, draws or writes elsewhere)")
        # the old name of `follow_roi`: a plugin that still says it would
        # lose its tick without a word
        if "live" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__}: `live` is `follow_roi` now")

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
    # Cheap enough to redo while the ROI is dragged.  A plugin that says so
    # gets a *follow ROI* tick in the GUI, which measures again between drag
    # events -- its `preview` if it has one, otherwise `run`, looked at and
    # never applied -- what makes a measurement something one aims with
    # rather than something one asks for and reads afterwards.  Only claim it
    # for work measured in tens of milliseconds on a normal selection.  (It
    # was `live` once; "live" now means only a fit still appending.)
    follow_roi: bool = False

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

    def changes_locs_for(self, settings) -> bool:
        """Whether a run with ``settings`` changes the table; see `changes_locs`."""
        return self.changes_locs

    def column_recipe_for(self, settings) -> bool:
        """Whether a run with ``settings`` hands back a `ColumnRecipe`, and so
        can change a table a fit is still appending to.  False by default."""
        return False

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
        curve was, so reopening the file next week and pressing *Show result* should
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

    def describe(self, ctx: Optional["Context"], settings) -> FormState:
        """What the form should show about ``settings`` beyond their values:
        hints for the fields on *auto*, which fields are read, the lists that
        depend on the session, and a notice (`FormState`).

        Asked after every edit and when the session changes, so it has to be
        cheap: a fitter says here that the file's camera is not in the
        database, while the file is being chosen rather than when the run
        fails for want of a conversion.  ``ctx`` is the session's context, or
        None when there is no session -- a part that needs one is then left
        out.  One hook rather than one per part, so that a chain forwards its
        steps' with one call and none is forgotten.
        """
        return FormState()

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
        return param_specs(cls.Settings)


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


def chains_dir() -> Path:
    """Where **Save chain** writes, and a folder of chains discovery reads.

    The one folder read without the user naming it (`config.plugin_roots` is
    deliberately explicit): a chain holds no code, so reading one runs nothing
    the user did not write.
    """
    from .. import config
    return config.config_dir() / "chains"


def chain_roots() -> List[Tuple[str, Path]]:
    return [("chains", chains_dir())]


def discover(force: bool = False) -> Dict[str, PluginRef]:
    """Scan the roots.  Cheap, and idempotent unless ``force``."""
    global _SCANNED, _REFS, _PROBLEMS
    if _SCANNED and not force:
        return _REFS
    _REFS, _PROBLEMS = scan(roots(), chain_roots())
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
