"""*Trace inspector*: one trace at a time, as SMAP's Cluster_MINFLUX_Roi --
its positions and its quality attributes against time, the spectrum of its
positions, and the numbers that say whether it is one still molecule: the
spread, a robust spread and the precision from consecutive differences per
axis, the time between localizations, the medians of efo, cfr and photons.

``per_region``: in the ROI manager or a block over ``tid`` it inspects each
trace and its numbers go to the region table; on the view it takes the
longest trace the selection holds (or the one *trace* names), or, with
*gallery*, the longest few side by side.  The numbers are
`smappy.traces`'.

**What Trace statistics marked is drawn, not measured.**  Where *Trace
statistics* wrote its marks onto the localizations (*write the tail*:
`in_tail`, and `after_cfr` past a cfr cut), the inspector shades those
stretches of time and greys their points, so the tail rule can be judged on
the trace it was applied to.  The numbers stay those of the whole trace: the
inspector is where one looks at what a rule would leave out, and leaving it
out here would hide exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register
from .id_statistics import _id_choices, id_column

QUALITY = ("efo", "cfr", "photons", "fbg")
#: the marks *Trace statistics* writes (*write the tail*), drawn as shaded
#: stretches of time: column, what the legend calls it, its shade
MARKS = (("in_tail", "tail", "0.82"), ("after_cfr", "after cfr", "#e8a39c"))
#: the colour of a marked localization's points
MARKED = "0.65"


@dataclass
class TraceInspectorSettings:
    id_column: str = param("", label="id attribute", choices=_id_choices,
                           help="the traces; auto: tid, else track_id, else cluster_id")
    trace: int = param(-1, label="trace", min=-1,
                       help="-1: the longest in the selection; else this id")
    window: int = param(20, label="running median", unit="localizations", min=1,
                        help="the window of the running median drawn over the positions")
    gallery: int = param(0, label="gallery", min=0, max=24,
                         help="0: one trace; else this many of the longest, side by side")


@register("Analysis/Tracking/Trace inspector")
class TraceInspector(Plugin):
    """One trace inspected: positions and quality against time with a
    running median, the positions' spectrum, and its spread, robust spread
    and consecutive-difference precision per axis; or a gallery of the
    longest traces."""

    Settings = TraceInspectorSettings
    changes_locs = False
    grouping = "ungrouped"
    per_region = True
    follow_roi = True
    # 2: the trace's axis among the values (angle_deg forward in time, the
    # spread along and across, the centre), drawn over the ROI and proposed
    # as the ROI's angle, a line and its centre
    version = "2"

    def describe(self, ctx: Optional[Context], settings: TraceInspectorSettings) -> FormState:
        state = FormState()
        state.active["trace"] = settings.gallery == 0
        return state

    def run(self, ctx: Context, settings: TraceInspectorSettings) -> Result:
        from .. import traces as T
        locs = ctx.locs
        column = id_column(locs, settings.id_column)
        clock = "time_s" in locs
        time = np.asarray(locs["time_s"] if clock else locs["frame"], np.float64)
        mask = np.asarray(ctx.selection.mask, bool)
        r = T.runs(locs[column], time, mask)
        if not len(r.ids):
            if ctx.per_region:
                return Result(values={"n_locs": 0}, settings=settings)
            raise ValueError(f"no trace of {column} in the selection")
        if settings.gallery and not ctx.per_region:
            return self._gallery(ctx, settings, r, column, T)
        if settings.trace >= 0 and not ctx.per_region:
            hit = np.flatnonzero(r.ids == settings.trace)
            if not len(hit):
                raise ValueError(f"no trace {column} {settings.trace} in the selection")
            i = int(hit[0])
        else:
            i = int(np.argmax(r.n))
        rows = r.order[r.start[i]:r.start[i] + r.n[i]]
        t = time[rows]
        axes = [a for a in ("x", "y", "z") if f"{a}_nm" in locs]
        pos = {a: np.asarray(locs[f"{a}_nm"], np.float64)[rows] for a in axes}
        values = trace_values(t, pos, {q: np.asarray(locs[q], np.float64)[rows]
                                       for q in QUALITY if q in locs})
        values["trace"] = float(r.ids[i])
        # the trace's axis: which way it went, what an ROI of it is oriented
        # by (*Write results* (to the ROIs)), and drawn over the ROI image
        overlays, proposals = [], {}
        if "x" in pos and "y" in pos and len(rows) >= 3:
            one = T.runs(np.zeros(len(rows), np.int64), t)
            a = T.axes(one, pos["x"][one.order], pos["y"][one.order])
            axis = {k: float(a[k][0]) for k in ("centre_x_nm", "centre_y_nm",
                                                 "std_long_nm", "std_short_nm",
                                                 "angle_deg")}
            values.update(axis)
            if np.all(np.isfinite(list(axis.values()))):
                half = 2.0 * axis["std_long_nm"]          # +-2 sd: the bulk of it
                u = np.array([np.cos(np.radians(axis["angle_deg"])),
                              np.sin(np.radians(axis["angle_deg"]))]) * half
                c = np.array([axis["centre_x_nm"], axis["centre_y_nm"]])
                overlays.append({"kind": "line", "points": [(c - u).tolist(),
                                                            (c + u).tolist()],
                                 "label": "axis"})
                # what an ROI of this trace could take on: its direction as
                # the ROI's frame, the axis as a line, the trace's centre
                proposals = {"angle": axis["angle_deg"],
                             "line:axis": [(c - u).tolist(), (c + u).tolist()],
                             "centre": c.tolist()}
        unit = "s" if clock else "frame"
        lines = [f"{column} {r.ids[i]}: {len(rows)} localizations over "
                 f"{values['duration']:.3g} {unit}" + ("s" if unit == "frame" else ""),
                 "; ".join(f"{a}: sd {values[f'{a}_sd_nm']:.2f}, robust {values[f'{a}_robust_nm']:.2f}, "
                           f"consecutive {values[f'{a}_consecutive_nm']:.2f} nm" for a in axes[:2])]
        quality = {q: np.asarray(locs[q], np.float64)[rows] for q in QUALITY if q in locs}
        # an attribute never measured here (NaN) or not used (0) is not drawn
        quality = {q: v for q, v in quality.items()
                   if np.isfinite(v).any() and np.nanmax(np.abs(v)) > 0}
        marks = trace_marks(locs, rows)
        for label, flag in marks.items():
            lines.append(f"{label} (Trace statistics): {int(flag.sum())} of {len(rows)} "
                         "localizations, shaded")

        def draw(figure):
            draw_trace(figure, t, pos, quality, settings.window, unit, marks)

        n_panels = 2 + bool(quality)
        return Result(text="\n".join(lines), values=values,
                      plot=Plot(draw, panels=n_panels, size=(4.0 * n_panels, 3.2)),
                      settings=settings, title="trace inspector", overlays=overlays,
                      proposals=proposals if ctx.per_region else {})

    def _gallery(self, ctx, settings, r, column, T):
        locs = ctx.locs
        time = np.asarray(locs["time_s"] if "time_s" in locs else locs["frame"], np.float64)
        x = np.asarray(locs["x_nm"], np.float64)[r.order]
        y = np.asarray(locs["y_nm"], np.float64)[r.order]
        a = T.axes(r, x, y)
        xr, yr = T.rotate(r, x, y, a)
        pick = np.argsort(-r.n)[:settings.gallery]

        def run_of(i):
            return slice(int(r.start[i]), int(r.start[i] + r.n[i]))

        shown = [(int(r.ids[i]), time[r.order][run_of(i)], xr[run_of(i)], yr[run_of(i)],
                  marked(trace_marks(locs, r.order[run_of(i)]), int(r.n[i])))
                 for i in pick]

        def draw(figure):
            n = len(shown)
            grid = figure.subplots(2, n, squeeze=False)
            for k, (tid, t, u, v, off) in enumerate(shown):
                on = ~off
                grid[0, k].plot(u[off], v[off], ".", ms=1.5, color=MARKED)
                grid[0, k].plot(u[on], v[on], ".", ms=1.5, color="0.3")
                grid[0, k].set_aspect("equal", adjustable="datalim")
                grid[0, k].set_title(f"{column} {tid}", fontsize=7)
                grid[1, k].plot((t - t[0])[off], u[off], ".", ms=1.5, color=MARKED)
                grid[1, k].plot((t - t[0])[on], u[on], ".", ms=1.5, color="0.3")
                grid[1, k].tick_params(labelsize=6)
                grid[0, k].tick_params(labelsize=6)
            grid[0, 0].set_ylabel("across (nm)", fontsize=7)
            grid[1, 0].set_ylabel("along (nm)", fontsize=7)

        return Result(text=f"the {len(shown)} longest traces of {column}, each turned onto "
                           "its long axis (top) and along it against time (bottom)",
                      plot=Plot(draw, panels=len(shown), size=(2.2 * len(shown), 4.4)),
                      settings=settings, title="trace gallery")


def trace_values(t, positions, quality) -> dict:
    """One trace's numbers: per axis the sd (ddof 1), a robust sd (1.4826 x
    the median absolute deviation) and the precision from consecutive
    differences (std(diff)/sqrt(2)); the number, the duration and the median
    time between localizations; the median of each quality attribute."""
    out = {"n_locs": float(len(t)), "duration": float(t[-1] - t[0]) if len(t) else np.nan,
           "dt_median": float(np.median(np.diff(t))) if len(t) > 1 else np.nan}
    for a, v in positions.items():
        v = v[np.isfinite(v)]
        out[f"{a}_sd_nm"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        out[f"{a}_robust_nm"] = (float(1.4826 * np.median(np.abs(v - np.median(v))))
                                 if len(v) else np.nan)
        out[f"{a}_consecutive_nm"] = (float(np.diff(v).std(ddof=1) / np.sqrt(2))
                                      if len(v) > 2 else np.nan)
    for q, v in quality.items():
        v = v[np.isfinite(v)]
        out[f"{q}_median"] = float(np.median(v)) if len(v) else np.nan
    return out


def trace_marks(locs, rows) -> dict:
    """``{label: flag}`` for the marks of `MARKS` the table has, over the
    trace's ``rows`` in time order; a mark no row of the trace carries is
    left out, so an untouched trace draws as it did."""
    out = {}
    for column, label, _ in MARKS:
        if column in locs:
            flag = np.asarray(locs[column])[rows] > 0
            if flag.any():
                out[label] = flag
    return out


def marked(marks: dict, n: int) -> np.ndarray:
    """Any mark, per localization: the points drawn grey."""
    out = np.zeros(n, bool)
    for flag in marks.values():
        out |= flag
    return out


def spans(t, flag) -> list:
    """The stretches of time ``flag`` is on, as ``(start, end)`` pairs: each
    run of marked localizations from its first time to the next unmarked
    one's, so the shading meets the trace where the mark ends."""
    flag = np.asarray(flag, bool)
    if not flag.any():
        return []
    edges = np.diff(np.r_[0, flag.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(float(t[a]), float(t[min(b, len(t) - 1)])) for a, b in zip(starts, ends)]


def shade(ax, t0, marks: dict) -> None:
    """Each mark's stretches behind the curves, labelled once."""
    shades = {label: colour for _, label, colour in MARKS}
    for label, flag in marks.items():
        for k, (a, b) in enumerate(spans(t0, flag)):
            ax.axvspan(a, b, color=shades[label], alpha=0.5, lw=0, zorder=0,
                       label=label if k == 0 else None)


def running_median(v, window: int):
    """The median over ``window`` points centred on each (fewer at the ends)."""
    v = np.asarray(v, np.float64)
    if window <= 1 or len(v) < 3:
        return v
    from scipy.ndimage import median_filter
    return median_filter(v, size=min(window, len(v)), mode="nearest")


def draw_trace(figure, t, pos, quality, window: int, unit: str,
               marks: Optional[dict] = None) -> None:
    marks = marks or {}
    n = 2 + bool(quality)
    axs = np.atleast_1d(figure.subplots(1, n))
    t0 = t - t[0]
    shade(axs[0], t0, marks)
    off = marked(marks, len(t))
    for k, (a, v) in enumerate(pos.items()):
        c = f"C{k}"
        axs[0].plot(t0[~off], (v - np.nanmean(v))[~off], ".", ms=1.5, color=c, alpha=0.5)
        axs[0].plot(t0[off], (v - np.nanmean(v))[off], ".", ms=1.5, color=MARKED, alpha=0.5)
        axs[0].plot(t0, running_median(v, window) - np.nanmean(v), color=c, lw=1, label=a)
    axs[0].set_xlabel(f"time ({unit}{'s' if unit == 'frame' else ''})")
    axs[0].set_ylabel("position - mean (nm)")
    axs[0].legend(fontsize=6, frameon=False)
    # the spectrum of the positions, on a regular grid where the clock is irregular
    for k, (a, v) in enumerate(pos.items()):
        ok = np.isfinite(v)
        if ok.sum() < 8:
            continue
        dt = float(np.median(np.diff(t[ok])))
        grid = np.arange(t[ok][0], t[ok][-1], dt) if dt > 0 else t[ok]
        vv = np.interp(grid, t[ok], v[ok])
        vv = vv - vv.mean()
        amp = 2 * np.abs(np.fft.rfft(vv)) / len(vv)
        f = np.fft.rfftfreq(len(vv), dt) if dt > 0 else np.arange(len(amp))
        axs[1].semilogy(f[1:], amp[1:], lw=0.7, color=f"C{k}", label=a)
    axs[1].set_xlabel("frequency (Hz)" if unit == "s" else "frequency (1/frame)")
    axs[1].set_ylabel("amplitude (nm)")
    axs[1].legend(fontsize=6, frameon=False)
    if quality:
        ax = axs[2]
        shade(ax, t0, marks)
        for k, (q, v) in enumerate(quality.items()):
            ok = np.isfinite(v)
            if not ok.any():
                continue
            scale = np.nanmedian(np.abs(v[ok])) or 1.0
            ax.plot(t0[ok], v[ok] / scale, ".", ms=1.5, color=f"C{k}", alpha=0.4)
            ax.plot(t0, running_median(np.where(ok, v, np.nanmedian(v)), window) / scale,
                    color=f"C{k}", lw=1, label=f"{q} (/{scale:.3g})")
        ax.set_xlabel(f"time ({unit}{'s' if unit == 'frame' else ''})")
        ax.set_ylabel("relative to its median")
        ax.legend(fontsize=6, frameon=False)
