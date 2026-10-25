"""*After the fit*: what is run over a finished table, as one step.

Every fitter has an *after the fit* part with these settings, and runs them
once its last frame is in, before the table is saved; the same settings are
this plugin, which runs them on the session's table -- the fitter's button
*Run these now*, for a fit that finished without them, or a chain.

What a finished fit wants depends on what was fitted, so the steps are fixed
rather than a chain of anything, each shown only where it applies, its
settings only while it is chosen:

* *astigmatic z* -- z from the widths of an elliptical Gaussian fit
  (`astigmatic_z.py`).  First, so that a drift correction after it can
  correct z too.
* *drift correction* -- none, RCC or COMET (`drift_rcc.py`, `drift_comet.py`).
* *assign colours* -- a channel per localization from its photon ratio, for
  a two-colour fit (`assign_colors.py`).  Last of the three: the correction
  moves positions and the assignment reads photons, so the order changes
  neither answer, but a colour histogram is what one looks at next.
* *then run chain* -- any saved chain, for whatever else the lab does to
  every fit.  A chain that fits is refused: it would fit again.

Each is the shipped plugin with its own settings -- the same code the
Analysis tab runs, so a fit that finishes itself and a table finished by
hand afterwards give the same numbers.

A step whose input the table does not have is passed over, not failed:
colours on a one-colour fit, z on widths that were never fitted.  That is
what lets one part serve every fitter, and the form hides those steps where
they cannot apply (`shown`).  A step that runs and fails is a note and
nothing more: these run after a fit, and a colour histogram with one mode or
a drift estimate on too few localizations must cost its own step, not the
twenty minutes that produced the table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..drift import DriftSettings
from ..locs import Localizations
from ..rcc import RCCSettings
from . import Context, FormState, Plugin, Result, param, register
from .assign_colors import AssignColorSettings
from .astigmatic_z import WIDTHS, AstigmaticZSettings

CHAIN_FILTER = "Chains (*.chain.yaml);;All files (*)"


@dataclass
class AfterFitSettings:
    """The steps in the order they run; each step's settings follow its switch."""
    astigmatic_z: bool = param(False, label="astigmatic z",
                               help="z from the widths of the elliptical Gaussian, before any "
                                    "drift correction so that it can correct z "
                                    "(Analysis/After fit/Astigmatic z)")
    astigmatic: AstigmaticZSettings = param(default_factory=AstigmaticZSettings,
                                            label="z calibration", collapsed=True)
    drift: str = param("none", label="drift correction",
                       choices=(("none", "none"), ("rcc", "RCC"), ("comet", "COMET")),
                       help="estimate the drift from the finished table and subtract it from "
                            "every localization.  Off by default: it is minutes of work on a "
                            "dataset it cannot see beforehand")
    rcc: RCCSettings = param(default_factory=RCCSettings, label="RCC drift",
                             collapsed=True)
    comet: DriftSettings = param(default_factory=DriftSettings, label="COMET drift",
                                 collapsed=True)
    assign_colors: bool = param(True, label="assign colours",
                                help="split the localizations by their photon ratio and "
                                     "write `channel` (Analysis/After fit/Assign colours)")
    colors: AssignColorSettings = param(default_factory=AssignColorSettings,
                                        label="colour assignment", collapsed=True)
    chain: str = param("", label="then chain", kind="open_file", file_filter=CHAIN_FILTER,
                       help="a saved chain run last, on the finished table; empty: none")

    def steps(self) -> List[tuple]:
        """``(plugin, settings)`` in the order they run; empty for none.

        The plugins are imported here rather than at the top of the module:
        their settings are needed to declare the fields above, the plugins
        themselves not until somebody runs one.
        """
        out = []
        if self.astigmatic_z:
            from .astigmatic_z import AstigmaticZ
            out.append((AstigmaticZ(), self.astigmatic))
        if self.drift == "rcc":
            from .drift_rcc import RCCDrift
            out.append((RCCDrift(), self.rcc))
        elif self.drift == "comet":
            from .drift_comet import CometDrift
            out.append((CometDrift(), self.comet))
        if self.assign_colors:
            from .assign_colors import AssignColors
            out.append((AssignColors(), self.colors))
        if self.chain:
            cls = chain_plugin(self.chain)
            out.append((cls(), cls.Settings()))
        return out

    def summary(self, two_channel: bool = True, widths: bool = True) -> str:
        """What the folded part's title says: the steps that are on, of those
        the table can take (`applicable`)."""
        on = applicable(self, two_channel, widths)
        names = (["astigmatic z"] if on.astigmatic_z else []) + \
            ([{"rcc": "RCC", "comet": "COMET"}[on.drift]] if on.drift != "none" else []) + \
            (["colours"] if on.assign_colors else []) + \
            ([_chain_name(on.chain)] if on.chain else [])
        return "after the fit: " + (", ".join(names) or "nothing")


def chain_plugin(path: str) -> type:
    """The chain file at ``path`` as a plugin class, refused if a step fits:
    it would run on a table, open the movie and fit it again."""
    from ..chain import load_class
    cls = load_class(path)
    fits = [step.title() for _, step, plugin in cls.steps
            if plugin.path.startswith("Localize/")]
    if fits:
        raise ValueError(f"the chain {_chain_name(path)!r} fits ({', '.join(fits)}): "
                         "after the fit there is nothing to fit")
    return cls


def _chain_name(path: str) -> str:
    from pathlib import Path
    from ..chain import chain_title
    return chain_title(Path(path))


def applicable(settings: AfterFitSettings, two_channel: bool,
               widths: bool) -> AfterFitSettings:
    """``settings`` with the steps the table cannot take turned off: colours
    without two channels' photons, astigmatic z without the widths."""
    return replace(settings, assign_colors=settings.assign_colors and two_channel,
                   astigmatic_z=settings.astigmatic_z and widths)


def table_can(locs: Localizations, colors: AssignColorSettings) -> Dict[str, bool]:
    """What the table has for the steps that need something: two channels'
    photons (as Assign colours finds them), and the elliptical widths."""
    from .assign_colors import channel_columns
    try:
        channel_columns(locs, colors.channel1, colors.channel2)
        two = True
    except ValueError:
        two = False
    return {"two_channel": two, "widths": all(c in locs for c in WIDTHS)}


def shown(settings: AfterFitSettings, two_channel: bool, widths: bool) -> Dict[str, bool]:
    """Which fields the form shows: a step only where it applies, its
    settings only while it is on, and inside them what their own plugin
    shows (the file or the typed curves of Astigmatic z)."""
    out = {"astigmatic_z": widths,
           "astigmatic": widths and settings.astigmatic_z,
           "rcc": settings.drift == "rcc", "comet": settings.drift == "comet",
           "assign_colors": two_channel,
           "colors": two_channel and settings.assign_colors}
    from .astigmatic_z import AstigmaticZ
    try:
        inner = AstigmaticZ().describe(None, settings.astigmatic).shown
    except Exception:
        inner = {}
    out.update({f"astigmatic.{name}": on for name, on in inner.items()})
    return out


@dataclass
class Finished:
    """The outcome of the steps: what to save, and what to say."""
    locs: Localizations
    changed: bool = False               # is the streamed file now out of date?
    notes: List[str] = field(default_factory=list)
    plots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


def with_session_merge(settings: AfterFitSettings, session) -> AfterFitSettings:
    """The drift steps merge blinks as the session does, as they would when run
    from the Analysis tab (`drift.merge_from_session`)."""
    from ..drift import merge_from_session
    return replace(settings, rcc=merge_from_session(settings.rcc, session),
                   comet=merge_from_session(settings.comet, session))


def finish_localizations(locs: Localizations, settings: AfterFitSettings,
                         report=None) -> Finished:
    """Run the steps over a finished table.

    A plain function over a table, so a script can finish a file the same way
    the Localize tab finishes a fit.  The steps the table cannot take are
    passed over (`applicable`); one that fails is a note and nothing more.
    """
    finished = Finished(locs)
    can = table_can(locs, settings.colors)
    try:
        steps = applicable(settings, **can).steps()
    except Exception as error:          # a chain file that cannot be read
        finished.notes.append(f"after the fit: {type(error).__name__}: {error}")
        steps = applicable(replace(settings, chain=""), **can).steps()
    for plugin, sub in steps:
        label = plugin.name or plugin.path.rsplit("/", 1)[-1]
        if report:
            report(f"{label}: starting")
        try:
            result = plugin.run(Context(locs=finished.locs, progress=report), sub)
        except Exception as error:
            finished.notes.append(f"{label} failed -- "
                                  f"{type(error).__name__}: {error}")
            continue
        if result.locs is not None:
            finished.locs = result.locs
            finished.changed = True
            finished.history.append(
                {"time": datetime.now().isoformat(timespec="seconds"),
                 "what": plugin.path, "text": result.text,
                 "settings": asdict(sub) if is_dataclass(sub) else sub,
                 "changed": True})
        # the drift curve and the colour histogram are the whole reason for
        # looking at a finished fit, so they travel with the fit's own result
        for figure in result.figures():
            name = f"{label}: {figure.name}" if figure.name else label
            finished.plots[name] = figure
        finished.notes.append(f"{label}: {result.text.splitlines()[0]}"
                              if result.text else label)
    return finished


@register("Analysis/After fit/After the fit")
class AfterFit(Plugin):
    """The fitters' *after the fit*, on the session's table."""

    description = ("What a fitter runs once its last frame is in -- astigmatic z, drift "
                   "correction, colours, a chain -- on the open localizations.")
    Settings = AfterFitSettings
    version = "1"
    changes_locs = True
    # the steps write onto every localization and measure drift on all of them
    grouping = "ungrouped"

    def describe(self, ctx: Optional[Context], settings: AfterFitSettings) -> FormState:
        """The steps the open table can take, and what *auto* merges by."""
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        can = (table_can(locs, settings.colors) if locs is not None and len(locs)
               else {"two_channel": True, "widths": True})
        state = FormState(shown=shown(settings, **can))
        from ..drift import merge_values
        dx, dt = merge_values(ctx.session if ctx is not None else None)
        for part in ("rcc", "comet"):
            state.hints[f"{part}.group_dx_nm"] = dx
            state.hints[f"{part}.group_dt"] = dt
        return state

    def run(self, ctx: Context, settings: AfterFitSettings) -> Result:
        locs, _ = ctx.ungrouped()
        used = with_session_merge(settings, ctx.session)
        finished = finish_localizations(locs, used, ctx.report)
        text = "\n".join(finished.notes) or "nothing to run on this table"
        return Result(locs=finished.locs if finished.changed else None, text=text,
                      plots=finished.plots, settings=used,
                      log={"steps": finished.history} if finished.history else {})
