"""What changes during an acquisition: any quantity against time.

SMAP's ``Analyze/measure/StatsVsTime``, widened.  An acquisition that
degrades -- the density thins out as the dye bleaches, the background climbs
with the activation laser, the precision drifts as the focus lock gives up --
shows it first as a trend against time, and the trend is what to look at
before trusting anything measured over the whole movie.

**Series.**  Any mix of: the number of localizations per frame, a column's
median (or mean) in each time window -- photons, `background`, `xy_err_nm`,
`sigma_nm`, `z_nm`, anything numeric -- and the image tags the fit kept
(`smappy.frametags`: the piezo position, the UV pulse, ...).  SMAP had a
fixed set (photons, lifetime, background, PSF width, precision, falloff);
here a column is a column, so a column another program wrote or Math Parser
made is a series too.  The tags make this the one place to see how the data
follow what the microscope did, which Image Tags shows one tag at a time.

**Windows.**  The frames are cut into *windows* of equal length (SMAP's
*framewindows*), and a statistic of a window is of the localizations in it:
a median rather than SMAP's fitted exponential mean of the photons, because
it needs no model and is the same statistic for every column.  A window with
fewer than `MIN_IN_WINDOW` localizations gives no point -- a median of three
is noise drawn as a trend.  Per file instead of per window when *x* is the
file number (SMAP's *files*).

**Layers.**  Each visible layer is its own line, read as the layer shows it
-- its filter, the ROI, merged blinks or not (merged, the count is blinks
per frame) -- since the comparison is usually between them.  The tags belong
to the acquisition, not to a layer, and are drawn once.

**One graph.**  With *draw* on *one graph*, every series is divided by its
own largest value, SMAP's "all" axis: the shapes compare, the numbers do not,
which is what the panels are for.

**Time.**  *x* in seconds is the frame times the *frame time*, which defaults
to the camera's exposure recorded with the fit -- the exposure, not the
cycle time, which smappy is not told; set it when the two differ.  A table
with its own clock (MINFLUX's `time_s`) uses that instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register

#: the series that is not a column: localizations (or blinks) per frame
COUNT = "localizations per frame"
#: a tag's series name: what the list shows and the settings keep
TAG = "tag: "
#: a window with fewer localizations than this gives no point
MIN_IN_WINDOW = 10


# ------------------------------------------------------------- the algorithm

def window_edges(first: float, last: float, windows: int) -> np.ndarray:
    """``windows`` equal windows over ``[first, last]`` of a continuous clock
    (MINFLUX's seconds), the last one closed."""
    windows = max(int(windows), 1)
    if not last > first:
        last = first + 1
    return np.linspace(first, last, windows + 1)


def frame_windows(first: int, last: int, windows: int) -> Tuple[np.ndarray, np.ndarray]:
    """About ``windows`` windows of a whole number of frames over frames
    ``first`` to ``last``: ``(edges, frames per window)``.

    Whole frames because a window of 19.99 frames holds 19 or 20 of them,
    and a steady rate drawn over such windows wobbles by a twentieth; the
    last window counts only the frames there are."""
    span = int(last) - int(first) + 1
    width = max(int(np.ceil(span / max(int(windows), 1))), 1)
    n = int(np.ceil(span / width))
    edges = int(first) + width * np.arange(n + 1, dtype=float)
    covered = np.minimum(edges[1:], int(last) + 1) - edges[:-1]
    return edges, covered


def _which(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    k = np.searchsorted(edges, t, side="right") - 1
    return np.clip(k, 0, len(edges) - 2)


def rate(t: np.ndarray, edges: np.ndarray, widths: Optional[np.ndarray] = None
         ) -> np.ndarray:
    """Localizations per frame (or per unit of ``t``) in each window: the
    count over ``widths``, the frames each window covers (its width in the
    units of ``t`` if not given)."""
    counts = np.bincount(_which(np.asarray(t, float), edges), minlength=len(edges) - 1)
    return counts / (np.diff(edges) if widths is None else widths)


def window_statistic(t: np.ndarray, values: np.ndarray, edges: np.ndarray,
                     statistic: str = "median",
                     min_count: int = MIN_IN_WINDOW) -> np.ndarray:
    """The median (or mean) of ``values`` in each window; NaN in a window
    with fewer than ``min_count`` finite values."""
    t = np.asarray(t, float)
    values = np.asarray(values, float)
    ok = np.isfinite(values) & np.isfinite(t)
    k = _which(t[ok], edges)
    v = values[ok]
    out = np.full(len(edges) - 1, np.nan)
    order = np.argsort(k, kind="stable")
    k, v = k[order], v[order]
    bounds = np.searchsorted(k, np.arange(len(edges)))
    for w in range(len(edges) - 1):
        chunk = v[bounds[w]:bounds[w + 1]]
        if len(chunk) >= min_count:
            out[w] = np.median(chunk) if statistic == "median" else chunk.mean()
    return out


def tag_in_windows(frames: np.ndarray, values: np.ndarray, edges: np.ndarray
                   ) -> np.ndarray:
    """A numeric tag averaged over the frames of each window."""
    return window_statistic(frames, values, edges, "mean", min_count=1)


def change(series: np.ndarray) -> Tuple[float, float]:
    """The first and the last finite value: what the text reports."""
    finite = series[np.isfinite(series)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(finite[0]), float(finite[-1])


@dataclass
class Line:
    """One series of one layer (or of the acquisition, for a tag)."""
    series: str
    layer: str
    colour: str
    x: np.ndarray
    y: np.ndarray
    states: Optional[List[str]] = None         # a tag that is a state, not a number
    label: str = ""                            # what it is called, if not its series


# --------------------------------------------------------------- drawing

def _label(series: str) -> str:
    return series[len(TAG):] if series.startswith(TAG) else series


def _name(line: "Line") -> str:
    return line.label or _label(line.series)


#: a series' colour when there is one layer: a LUT's colour (hot's yellow)
#: is unreadable as a line on white, and one layer has nothing to tell apart
SERIES_COLOURS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


def draw_panels(figure, names: Sequence[str], lines: Sequence[Line], xlabel: str) -> None:
    """One panel per series, the time axis shared."""
    axes = figure.subplots(len(names), 1, sharex=True, squeeze=False).ravel()
    for ax, name in zip(axes, names):
        mine = [ln for ln in lines if ln.series == name]
        for ln in mine:
            if ln.states is not None:
                ax.step(ln.x, ln.y, where="post", color=ln.colour, lw=1)
                ax.set_yticks(range(len(ln.states)))
                ax.set_yticklabels(ln.states)
            else:
                ax.plot(ln.x, ln.y, color=ln.colour, lw=1.0, marker=".", ms=2,
                        label=ln.layer)
        ax.set_ylabel(_name(mine[0]) if mine else _label(name), fontsize=8)
        if len({ln.layer for ln in mine}) > 1:
            ax.legend(fontsize=7, frameon=False)
    axes[-1].set_xlabel(xlabel)


def draw_one(ax, lines: Sequence[Line], xlabel: str) -> None:
    """Every numeric series in one graph, each divided by its largest value:
    a colour per series for one layer, a colour per layer and a line style
    per series for several."""
    styles = ("-", "--", ":", "-.")
    names = list(dict.fromkeys(ln.series for ln in lines if ln.states is None))
    several = len({ln.layer for ln in lines if ln.layer}) > 1
    for ln in lines:
        if ln.states is not None:
            continue
        top = np.nanmax(np.abs(ln.y)) if np.isfinite(ln.y).any() else np.nan
        if not top:
            continue
        k = names.index(ln.series)
        colour = ln.colour if several or not ln.layer else SERIES_COLOURS[k % 8]
        style = styles[k % len(styles)] if several else "-"
        label = _name(ln) + (f" ({ln.layer})" if several and ln.layer else "")
        ax.plot(ln.x, ln.y / top, color=colour, linestyle=style, lw=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative to its maximum")
    ax.legend(fontsize=7, frameon=False)


# ---------------------------------------------------------------- the plugin

@dataclass
class TimeCourseSettings:
    series: list = param(default_factory=lambda: [COUNT, "photons"], label="series",
                         kind="columns",
                         help="what to draw against time: the localization rate, "
                              "columns (their median per window), image tags")
    statistic: str = param("median", label="statistic",
                           choices=(("median", "median"), ("mean", "mean")),
                           help="of a column in each window")
    x: str = param("frame", label="x",
                   choices=(("frame", "frame"), ("seconds", "time (s)"),
                            ("file", "file number")),
                   help="what the series are drawn against")
    windows: int = param(100, label="windows", min=2, max=10000,
                         help="how many equal parts the acquisition is cut into")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001,
                                           help="auto: the exposure the fit recorded")
    draw: str = param("panels", label="draw",
                      choices=(("panels", "a panel per series"),
                               ("one", "one graph, each to its maximum")),
                      help="one graph compares the shapes")


def _tags(locs) -> Dict[str, np.ndarray]:
    from ..frametags import KEY
    return (getattr(locs, "metadata", None) or {}).get(KEY) or {}


def _frame_time(locs, settings: TimeCourseSettings) -> Optional[float]:
    """The frame time in ms: the setting, or the camera's exposure."""
    if settings.frame_time_ms:
        return float(settings.frame_time_ms)
    camera = (locs.metadata or {}).get("camera")
    exposure = camera.get("exposure_ms") if isinstance(camera, dict) else None
    return float(exposure) if exposure else None


def _layers(ctx: Context):
    """``(name, colour, locs, merged)`` per visible localization layer, as
    shown; ``merged`` when it shows merged blinks."""
    from .line_profile import FALLBACK_COLOURS, layer_colour
    found = []
    for i, layer in enumerate(getattr(ctx.session, "layers", None) or []):
        if layer.is_image or not layer.visible:
            continue
        taken = [c for _, c, _, _ in found]
        table, selection = ctx.table(i)
        merged = table is not ctx.ungrouped()[0]
        found.append((layer.name, layer_colour(layer.get_display(), len(found), taken),
                      selection.apply(table), merged))
    if not found:
        found.append(("", FALLBACK_COLOURS[0], ctx.selection.apply(ctx.locs), False))
    if len(found) == 1:                   # nothing to tell apart: a dark line
        name, _, locs, merged = found[0]
        found[0] = (name, SERIES_COLOURS[0], locs, merged)
    return found


@register("Analysis/Measure/Time Course")
class TimeCourse(Plugin):
    """The localization rate, any column's median and the image tags
    against time: what changed during the acquisition."""

    Settings = TimeCourseSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx: Optional[Context], settings: TimeCourseSettings) -> FormState:
        """The series there are: the rate, the numeric columns, the tags."""
        state = FormState(active={"frame_time_ms": settings.x == "seconds",
                                  "windows": settings.x != "file"})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            locs = ctx.locs
            columns = [n for n in locs.keys()
                       if n not in ("frame", "filenumber")
                       and np.asarray(locs[n]).ndim == 1
                       and np.asarray(locs[n]).dtype.kind in "iuf"]
            tags = [TAG + n for n in _tags(locs) if n != "frame"]
            state.choices["series"] = [COUNT, *columns, *tags]
        return state

    def run(self, ctx: Context, settings: TimeCourseSettings) -> Result:
        groups = _layers(ctx)
        reference = ctx.locs
        tags = _tags(reference)
        wanted = list(settings.series) or [COUNT] + [
            n for n in reference.keys() if n not in ("frame", "filenumber")]
        missing = [s for s in wanted if s != COUNT and not s.startswith(TAG)
                   and s not in reference]
        missing += [s for s in wanted if s.startswith(TAG) and s[len(TAG):] not in tags]
        if missing:
            raise ValueError(f"not in the table: {', '.join(missing)}")

        # the clock the windows are cut on -- whole frames, MINFLUX's seconds,
        # or files -- and the factor that turns it into the axis drawn
        scale = 1.0
        if settings.x == "file":
            if "filenumber" not in reference:
                raise ValueError("one file only: there is no filenumber to draw against")
            clock, xlabel = "filenumber", "file number"
        elif settings.x == "seconds" and "time_s" in reference:
            clock, xlabel = "time_s", "time (s)"
        elif settings.x == "seconds":
            ms = _frame_time(reference, settings)
            if not ms:
                raise ValueError("no frame time: the fit recorded no exposure, "
                                 "so set the frame time")
            clock, xlabel, scale = "frame", "time (s)", ms / 1000.0
        else:
            clock, xlabel = "frame", "frame"

        present = [locs for _, _, locs, _ in groups if len(locs)]
        if not present:
            raise ValueError("no localizations to draw")
        all_t = np.concatenate([np.asarray(locs[clock], float) for locs in present])
        widths = None
        if clock == "filenumber":
            files = np.unique(all_t)
            edges = np.append(files - 0.5, files[-1] + 0.5)
        elif clock == "frame":
            edges, widths = frame_windows(int(all_t.min()), int(all_t.max()),
                                          settings.windows)
        else:
            edges = window_edges(float(all_t.min()), float(all_t.max()),
                                 settings.windows)
        centres = 0.5 * (edges[:-1] + edges[1:])
        if clock == "filenumber":
            widths = np.ones(len(centres))            # per file, not per frame

        lines: List[Line] = []
        for series in wanted:
            if series.startswith(TAG):
                line = self._tag_line(series, tags, edges, centres, clock)
                if line is not None:
                    line.x = line.x * scale
                    lines.append(line)
                continue
            for name, colour, locs, merged in groups:
                t = np.asarray(locs[clock], float)
                if series == COUNT:
                    y = rate(t, edges, widths)
                else:
                    y = window_statistic(t, np.asarray(locs[series], float), edges,
                                         settings.statistic)
                label = ""
                if series == COUNT:
                    what = "blinks" if merged else "localizations"
                    per = {"filenumber": "file", "time_s": "s"}.get(clock, "frame")
                    label = f"{what} per {per}"
                lines.append(Line(series, name, colour, centres * scale, y, label=label))
        edges = edges * scale

        unit = "s" if settings.x == "seconds" else "frames"
        text = [f"{len(edges) - 1} windows of {np.diff(edges)[0]:.4g} {unit}"
                if clock != "filenumber" else f"{len(edges) - 1} files"]
        if wanted and any(s.startswith(TAG) for s in wanted) and clock != "frame":
            text.append("image tags are per frame: not drawn against "
                        + ("files" if clock == "filenumber" else "the table's own clock"))
        for ln in lines:
            if ln.states is not None:
                text.append(f"{_name(ln)}: {' -> '.join(ln.states)}")
                continue
            first, last = change(ln.y)
            rel = (f" ({100 * (last - first) / first:+.0f} %)"
                   if np.isfinite(first) and first else "")
            who = f" ({ln.layer})" if ln.layer else ""
            text.append(f"{_name(ln)}{who}: {first:.4g} at the start, "
                        f"{last:.4g} at the end{rel}")
        names = list(dict.fromkeys(ln.series for ln in lines))
        if settings.draw == "one":
            def plot(ax):
                draw_one(ax, lines, xlabel)
            figure = Plot(plot, size=(6.0, 3.6))
        else:
            def plot(fig):
                draw_panels(fig, names, lines, xlabel)
            figure = Plot(plot, panels=len(names), size=(6.0, 1.3 + 1.4 * len(names)))
        return Result(text="\n".join(text), plot=figure,
                      data={"edges": edges, "lines": lines}, settings=settings,
                      title="time course")

    @staticmethod
    def _tag_line(series, tags, edges, centres, clock) -> Optional[Line]:
        """A tag in the windows: averaged if numeric, its states if not.  A
        tag is per frame, so it has no place on a clock of files or seconds
        that is not the frames'."""
        if clock != "frame":
            return None
        name = series[len(TAG):]
        frames = np.asarray(tags["frame"], float)
        values = np.asarray(tags[name])
        if values.dtype == object:
            states = list(dict.fromkeys(str(v) for v in values))
            index = np.array([states.index(str(v)) for v in values], float)
            return Line(series, "", "0.35", frames, index, states=states)
        return Line(series, "", "0.35", centres,
                    tag_in_windows(frames, values.astype(float), edges))
