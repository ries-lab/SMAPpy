"""Binding kinetics: a site's time trace, its binding events and qPAINT.

*Time Course* is the acquisition's: medians over a hundred windows of the
whole field, a window of fewer than ten localizations dropped.  A DNA-PAINT
site has a few hundred localizations in a few dozen events, and the
question is about it alone: did it bind at a steady rate all through, how
long were its bindings and the gaps between them, how many docking strands
does it carry.  This answers that, once per ROI or cluster or on a drawn
ROI, as Picasso's pick kinetics and qPAINT do (Jungmann et al. 2016); the
array functions and what was ported and changed are in
`smappy/binding_kinetics.py`.

The figure is the trace: each event a stem at its start, as high as its
localizations, over the cumulative count against the diagonal of a
constant rate -- what the ROI manager shows while Up and Down step through
the ROIs and Space un-ticks the bad ones.  A row keeps the events (a few
hundred numbers) so that figure comes back without measuring again.

*qPAINT* (`Analysis/Regions/Summarize/qPAINT`) reads the regions' results:
the bright and dark times pooled over them, the qPAINT index per region,
and the influx calibrated on regions of known site number (an ROI keyword),
which turns every index into sites.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import NO_FRAME_TIME, Context, FormState, Plot, Plugin, Result, param, register
from .. import binding_kinetics as bk
from .blinking import _localizations
from .roi import EVALUATION_HELP, SiteAnalysisPlugin, auto_only

PATH = "Analysis/Measure/Binding kinetics"
TIME_AXES = (("frames", "frames"), ("seconds", "seconds"))


# ------------------------------------------------------------ the acquisition

# each table's per-file frame span, by the identity of its frame column: a
# block over 10^4 clusters asks once per cluster, and a min and max over a
# table of 10^7 frames each time would be minutes
_SPANS: Dict[int, Tuple[Any, Dict[int, Tuple[int, int]]]] = {}


def table_spans(locs) -> Dict[int, Tuple[int, int]]:
    """Each file's first and last frame in a whole table, remembered while
    its frame column lives."""
    if "frame" not in locs or not len(locs):
        return {}
    frame = locs["frame"]
    key = id(frame)
    found = _SPANS.get(key)
    if found is not None and found[0]() is frame:
        return found[1]
    file = locs["filenumber"] if "filenumber" in locs else None
    spans = bk.spans_of(frame, file)
    try:
        _SPANS[key] = (weakref.ref(frame, lambda _, key=key: _SPANS.pop(key, None)),
                       spans)
    except TypeError:                     # a column that takes no weak reference
        pass
    return spans


def acquisition(ctx: Context, locs, files, frames: Optional[int]
                ) -> Optional[Dict[int, Tuple[int, int]]]:
    """Each file's first and last frame of the acquisition: ``frames`` when
    set; else the file's ``n_frames`` (a simulation records it); else the
    file's first and last localization in the session's table.  None when
    none of these says -- a script's table of one region -- and the dark
    times at the ends are then not counted."""
    from ..metadata import file_metadata
    if frames:
        return {int(f): (0, int(frames) - 1) for f in files}
    out = {}
    for f in files:
        n = file_metadata(locs.metadata, int(f)).get("n_frames")
        if n:
            out[int(f)] = (0, int(n) - 1)
    missing = [int(f) for f in files if int(f) not in out]
    if missing and ctx.session is not None:
        whole = table_spans(ctx.session.locs)
        for f in missing:
            if f in whole:
                out[f] = whole[f]
    if any(int(f) not in out for f in files):
        return None
    return out


# --------------------------------------------------------------------- drawing

def draw_trace(figure, first, last, n, span, seconds: float = 1.0,
               unit: str = "frames", title: str = "") -> None:
    """Two panels on one time axis: each event a stem at its first frame as
    high as its localizations, and the cumulative events against the
    diagonal a constant rate would climb."""
    first, last, n = (np.asarray(a, float) for a in (first, last, n))
    top, bottom = figure.subplots(2, 1, sharex=True, squeeze=True)
    t = first * seconds
    if len(t):
        top.vlines(t, 0, n, color="0.25", lw=1)
        top.plot(t, n, "o", ms=2.5, color="0.25")
        top.hlines(n, t, (np.asarray(last, float) + 1) * seconds, color="0.25", lw=1)
    top.set_ylabel("localizations\nper event", fontsize=8)
    top.set_ylim(bottom=0)
    lo, hi = span
    k = len(first)
    bottom.step(np.r_[lo, first, hi + 1] * seconds, np.r_[0, np.arange(1, k + 1), k],
                where="post", color="#1f77b4", lw=1.2, label="events")
    bottom.plot(np.array([lo, hi + 1]) * seconds, [0, len(first)], "--", color="0.5",
                lw=0.9, label="constant rate")
    bottom.set_ylabel("events so far", fontsize=8)
    bottom.set_xlabel(f"time ({unit})")
    bottom.set_xlim(lo * seconds, (hi + 1) * seconds)
    bottom.legend(fontsize=7, frameon=False, loc="upper left")
    if title:
        top.set_title(title, fontsize=7.5, color="0.25")


def summary_line(values: Dict[str, float]) -> str:
    """One line of what a region measured, for the text and the title."""
    parts = [f"{values['n_events']} events, {values['n_localizations']} localizations"]
    if np.isfinite(values.get("tau_bright_frames", np.nan)):
        parts.append(f"bright {values['tau_bright_frames']:.3g} frames")
    if np.isfinite(values.get("tau_dark_frames", np.nan)):
        parts.append(f"dark {values['tau_dark_frames']:.4g} frames")
    if np.isfinite(values.get("qpaint_index", np.nan)):
        parts.append(f"qPAINT index {values['qpaint_index']:.3g} "
                     f"± {values['qpaint_index_err']:.2g} /frame")
    if np.isfinite(values.get("n_sites", np.nan)):
        parts.append(f"{values['n_sites']:.2f} ± {values['n_sites_err']:.2f} sites")
    return "; ".join(parts)


# ---------------------------------------------------------------- the plugin

@dataclass
class BindingKineticsSettings:
    link_gap: int = param(bk.LINK_GAP, label="link gap", unit="frames", min=0,
                          help="frames without a localization inside one binding "
                               "event; a longer gap is a dark time")
    estimator: str = param("ml", label="estimator", choices=bk.ESTIMATORS,
                           help="how the mean bright and dark times are found")
    influx: Optional[float] = param(None, label="influx", unit="1/frame", min=0.0,
                                    help="bindings per frame of one site, from qPAINT's "
                                         "calibration; empty: no site count")
    time_axis: str = param("frames", label="time in", choices=TIME_AXES,
                           help="the unit of the trace's time axis")
    windows: int = param(bk.WINDOWS, label="windows", min=2, advanced=True,
                         help="windows of the trace kept for Curves")
    frames: Optional[int] = param(None, label="acquisition", unit="frames", min=1,
                                  advanced=True,
                                  help="auto: the file's frames, else its first to "
                                       "last localization")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="auto: the file's, else its exposure")




def _seconds(settings: BindingKineticsSettings, locs) -> Optional[float]:
    from ..metadata import table_frame_time
    return settings.frame_time_ms or table_frame_time(locs)


@register("Analysis/Measure/Binding kinetics")
class BindingKinetics(Plugin):
    """A site's time trace and binding events: bright and dark times, the
    qPAINT index and, with an influx, its binding sites; values to curate
    sticky or partial sites by.  On the view, the selection is one site."""

    Settings = BindingKineticsSettings
    changes_locs = False
    per_region = True
    follow_roi = True
    version = "1"

    def describe(self, ctx, settings: BindingKineticsSettings) -> FormState:
        state = FormState()
        if ctx is not None and settings.time_axis == "seconds" and \
                not settings.frame_time_ms:
            from ..metadata import table_frame_time
            from . import frame_time_notice
            try:
                if not table_frame_time(ctx.locs):
                    state.notice = frame_time_notice()
            except Exception:                              # noqa: BLE001
                pass
        return state

    def run(self, ctx: Context, settings: BindingKineticsSettings) -> Result:
        locs, blink = _localizations(ctx)
        if "frame" not in locs:
            raise ValueError("the table has no frame column: "
                             f"it has {', '.join(sorted(locs.keys()))}")
        if not len(locs) and not ctx.per_region:
            raise ValueError("no localizations to measure: draw an ROI round a site")
        frame = np.asarray(locs["frame"])
        file = (np.asarray(locs["filenumber"], np.int64) if "filenumber" in locs
                else np.zeros(len(locs), np.int64))
        files = np.unique(file)
        spans = acquisition(ctx, locs, files, settings.frames)
        ft = _seconds(settings, locs)
        seconds = settings.time_axis == "seconds"
        if seconds and not ft:
            raise ValueError(NO_FRAME_TIME)
        values, arrays, events = bk.region_kinetics(
            frame, blink, file, settings.link_gap, spans, settings.estimator,
            settings.influx, ft, settings.windows)
        if spans is None:
            spans = bk.spans_of(frame, file)
        line = summary_line(values)
        text = line
        if spans and len(spans) == 1 and values["n_events"]:
            text += (f"; frames {values['first_frame']}-{values['last_frame']}, "
                     f"mean at {values['mean_frame_rel']:.2f} of the acquisition, "
                     f"{100 * values['peak_window_fraction']:.0f} % in its busiest "
                     f"twentieth")
        if not ctx.per_region and values["n_events"] > 1000:
            text += ("; the selection is taken as one site: draw an ROI round one, "
                     "or run per region")
        span = (min(lo for lo, _ in spans.values()), max(hi for _, hi in spans.values())) \
            if spans else (0, 0)
        scale = ft / 1000.0 if seconds else 1.0
        unit = "s" if seconds else "frames"
        trace = {"first": events.first.tolist(), "last": events.last.tolist(),
                 "n": events.n.tolist(), "span": list(span), "seconds": scale,
                 "unit": unit, "title": line}

        def plot(figure) -> None:
            draw_trace(figure, trace["first"], trace["last"], trace["n"], span,
                       seconds=scale, unit=unit, title=line)

        return Result(text=text, values=values, arrays=arrays, settings=settings,
                      data={"events": events, "trace": trace},
                      plot=Plot(draw=plot, panels=2, size=(6.0, 4.4)),
                      title="binding kinetics")

    def keep(self, result: Result):
        """The events (three numbers each) and the span: what the trace is
        drawn from, a few hundred numbers for a site, never its
        localizations (`docs/regions.md`, "Figures")."""
        trace = (result.data or {}).get("trace")
        return dict(trace) if trace else None

    def restore(self, saved):
        try:
            first, last, n = saved["first"], saved["last"], saved["n"]
            span = tuple(int(v) for v in saved["span"])
            scale, unit = float(saved.get("seconds", 1.0)), str(saved.get("unit", "frames"))
            title = str(saved.get("title", ""))
        except (KeyError, TypeError, ValueError):
            return None
        return Result(plot=Plot(draw=lambda figure: draw_trace(
            figure, first, last, n, span, seconds=scale, unit=unit, title=title),
            panels=2, size=(6.0, 4.4)), title="binding kinetics")


# ------------------------------------------------------------- qPAINT summary

def pooled_kinetics(totals, periods, bright, influx: Optional[float] = None,
                    known_sites: float = 1.0) -> Dict[str, float]:
    """Pooled over regions: the qPAINT index from their dark frames and
    periods (`binding_kinetics.pooled_index`), the mean bright time of
    their events, and the influx it gives for ``known_sites`` sites each."""
    index, err = bk.pooled_index(totals, periods)
    bright = np.asarray(bright, float)
    bright = bright[np.isfinite(bright)]
    out = {"index": index, "index_err": err,
           "tau_bright_frames": float(bright.mean()) if len(bright) else float("nan"),
           "influx": index / known_sites, "influx_err": err / known_sites}
    if influx:
        out["influx"], out["influx_err"] = float(influx), 0.0
    return out


def draw_qpaint(figure, bright, dark, index, sites, calibration, pooled,
                influx: float) -> None:
    """Four panels: the bright and dark times of every region's events with
    the pooled means, the qPAINT index per region and the sites it gives,
    the calibration regions drawn apart."""
    from .blinking import integer_edges
    axes = figure.subplots(2, 2, squeeze=False).ravel()
    dark_tau = 1.0 / pooled["index"] if pooled["index"] else np.nan
    for ax, values, label, tau, name, start in (
            (axes[0], bright, "bright time (frames)", pooled["tau_bright_frames"],
             "mean", 1),
            (axes[1], dark, "dark time (frames)", dark_tau, "calibration", 0)):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if len(values):
            edges = integer_edges(values, start=start)
            ax.stairs(np.histogram(values, edges)[0], edges, color="0.3")
            ax.set_yscale("log")
        if np.isfinite(tau):
            ax.axvline(tau, color="#d62728", lw=1, ls="--", label=f"{name} {tau:.4g}")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel(label)
        ax.set_ylabel("events")
    index, sites = np.asarray(index, float), np.asarray(sites, float)
    calibration = np.asarray(calibration, bool)
    for ax, values, label in ((axes[2], index, "qPAINT index (1/frame)"),
                              (axes[3], sites, "binding sites")):
        ok = np.isfinite(values)
        if ok.any():
            edges = np.histogram_bin_edges(values[ok],
                                           bins=int(min(30, max(5, ok.sum() // 3))))
            ax.hist([values[ok & ~calibration], values[ok & calibration]], edges,
                    stacked=True, color=["0.6", "#1f77b4"],
                    label=["regions", "calibration"])
            if calibration.any() and (~calibration).any():
                ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel(label)
        ax.set_ylabel("regions")
    axes[3].set_title(f"influx {influx:.3g} /frame per site", fontsize=8)


def _no_lists():
    return [("", "every region")]


@dataclass
class QpaintSettings:
    evaluation: str = param("", label="results from", choices=auto_only,
                            help=EVALUATION_HELP)
    known_sites: float = param(1.0, label="known sites", min=0.001,
                               help="binding sites in each calibration region")
    keyword_list: str = param("", label="calibrate on", choices=_no_lists,
                              help="an ROI keyword list naming the calibration "
                                   "regions; every region: all of them")
    keyword: str = param("", label="keyword",
                         help="the keyword of the calibration regions")
    influx: Optional[float] = param(None, label="influx", unit="1/frame", min=0.0,
                                    advanced=True,
                                    help="auto: calibrated here; set to use a known one")


@register("Analysis/Regions/Summarize/qPAINT")
class Qpaint(SiteAnalysisPlugin):
    """qPAINT over the regions: their bright and dark times pooled, the
    influx calibrated on regions of known site number, and the sites of
    every region from its qPAINT index (Binding kinetics)."""

    Settings = QpaintSettings
    name = "qPAINT"
    changes_locs = False
    version = "1"
    evaluator = PATH

    def describe(self, ctx, settings: QpaintSettings) -> FormState:
        state = super().describe(ctx, settings)
        if ctx is None:
            return state
        lists = list(getattr(ctx.rois, "annotation_lists", {}) or {})
        state.choices["keyword_list"] = [("", "every region")] + [(n, n) for n in lists]
        # the calibration is typed in, or found on a keyword's regions
        calibrating = not settings.influx
        state.active = {"known_sites": calibrating, "keyword_list": calibrating,
                        "keyword": calibrating and bool(settings.keyword_list)}
        return state

    def run(self, ctx: Context, settings: QpaintSettings) -> Result:
        from .curves import Curves
        from .npc import column
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("no binding kinetics in the regions yet: add Binding "
                             "kinetics to the per-region steps and run them")
        label = None if ev is None else ev.label

        def values(name):
            key = column(rows, name, label)
            if key is None:
                raise ValueError(f"the region results have no {name}: run Binding "
                                 "kinetics again")
            return np.array([np.nan if r.get(key) is None else r.get(key) for r in rows],
                            float)

        index = values("qpaint_index")
        totals, periods = values("dark_total_frames"), values("dark_periods")
        if settings.keyword_list:
            key = column(rows, settings.keyword_list, None) or settings.keyword_list
            calibration = np.array([str(r.get(key, "")) == settings.keyword for r in rows])
            if not calibration.any() and not settings.influx:
                raise ValueError(f"no region has the keyword {settings.keyword!r} in "
                                 f"{settings.keyword_list!r} to calibrate on")
        else:
            calibration = np.ones(len(rows), bool)
        bright, dark = [], []
        if ev is not None:
            wanted = {str(r["roi_id"]) for r in rows}
            for region in Curves().regions(ctx, ev):
                if region.id in wanted:
                    bright.append(region.arrays.get("bright_frames", np.zeros(0)))
                    dark.append(region.arrays.get("dark_frames", np.zeros(0)))
        bright = np.concatenate(bright) if bright else np.zeros(0)
        dark = np.concatenate(dark) if dark else np.zeros(0)
        pooled = pooled_kinetics(totals[calibration], periods[calibration], bright,
                                 settings.influx, settings.known_sites)
        influx = pooled["influx"]
        if not (np.isfinite(influx) and influx > 0):
            raise ValueError("the calibration regions have too few events for an influx")
        sites = index / influx
        what = "ROIs" if ev is None or ev.on_rois else "regions"
        lines = [f"{len(rows)} {what}; influx {influx:.4g}"
                 + (f" ± {pooled['influx_err']:.2g}" if pooled["influx_err"] else "")
                 + " /frame per site"
                 + ("" if settings.influx else
                    f", from {int(calibration.sum())} {what} of {settings.known_sites:g} "
                    "sites"),
                 f"binding sites: mean {np.nanmean(sites):.2f}, median "
                 f"{np.nanmedian(sites):.2f}, sd {np.nanstd(sites):.2f}",
                 f"pooled bright time {pooled['tau_bright_frames']:.3g} frames"]
        ids = [r["roi_id"] for r in rows]
        data = {"influx": influx, "influx_err": pooled["influx_err"],
                "pooled": pooled, "sites": dict(zip(map(str, ids), sites.tolist())),
                "calibration": [str(i) for i, c in zip(ids, calibration) if c]}

        def plot(figure) -> None:
            draw_qpaint(figure, bright, dark, index, sites, calibration, pooled, influx)

        return Result(text="\n".join(lines), data=data, settings=settings,
                      plot=Plot(draw=plot, panels=4, size=(7.5, 5.6)), title="qPAINT")
