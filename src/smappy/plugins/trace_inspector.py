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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register
from .id_statistics import _id_choices, id_column

QUALITY = ("efo", "cfr", "photons", "fbg")


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
    version = "1"

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
        unit = "s" if clock else "frame"
        lines = [f"{column} {r.ids[i]}: {len(rows)} localizations over "
                 f"{values['duration']:.3g} {unit}" + ("s" if unit == "frame" else ""),
                 "; ".join(f"{a}: sd {values[f'{a}_sd_nm']:.2f}, robust {values[f'{a}_robust_nm']:.2f}, "
                           f"consecutive {values[f'{a}_consecutive_nm']:.2f} nm" for a in axes[:2])]
        quality = {q: np.asarray(locs[q], np.float64)[rows] for q in QUALITY if q in locs}
        # an attribute never measured here (NaN) or not used (0) is not drawn
        quality = {q: v for q, v in quality.items()
                   if np.isfinite(v).any() and np.nanmax(np.abs(v)) > 0}

        def draw(figure):
            draw_trace(figure, t, pos, quality, settings.window, unit)

        n_panels = 2 + bool(quality)
        return Result(text="\n".join(lines), values=values,
                      plot=Plot(draw, panels=n_panels, size=(4.0 * n_panels, 3.2)),
                      settings=settings, title="trace inspector")

    def _gallery(self, ctx, settings, r, column, T):
        locs = ctx.locs
        time = np.asarray(locs["time_s"] if "time_s" in locs else locs["frame"], np.float64)
        x = np.asarray(locs["x_nm"], np.float64)[r.order]
        y = np.asarray(locs["y_nm"], np.float64)[r.order]
        a = T.axes(r, x, y)
        xr, yr = T.rotate(r, x, y, a)
        pick = np.argsort(-r.n)[:settings.gallery]
        shown = [(int(r.ids[i]), time[r.order][r.start[i]:r.start[i] + r.n[i]],
                  xr[r.start[i]:r.start[i] + r.n[i]], yr[r.start[i]:r.start[i] + r.n[i]])
                 for i in pick]

        def draw(figure):
            n = len(shown)
            grid = figure.subplots(2, n, squeeze=False)
            for k, (tid, t, u, v) in enumerate(shown):
                grid[0, k].plot(u, v, ".", ms=1.5, color="0.3")
                grid[0, k].set_aspect("equal", adjustable="datalim")
                grid[0, k].set_title(f"{column} {tid}", fontsize=7)
                grid[1, k].plot(t - t[0], u, ".", ms=1.5, color="0.3")
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


def running_median(v, window: int):
    """The median over ``window`` points centred on each (fewer at the ends)."""
    v = np.asarray(v, np.float64)
    if window <= 1 or len(v) < 3:
        return v
    from scipy.ndimage import median_filter
    return median_filter(v, size=min(window, len(v)), mode="nearest")


def draw_trace(figure, t, pos, quality, window: int, unit: str) -> None:
    n = 2 + bool(quality)
    axs = np.atleast_1d(figure.subplots(1, n))
    t0 = t - t[0]
    for k, (a, v) in enumerate(pos.items()):
        c = f"C{k}"
        axs[0].plot(t0, v - np.nanmean(v), ".", ms=1.5, color=c, alpha=0.5)
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
