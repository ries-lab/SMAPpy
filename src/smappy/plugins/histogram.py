"""The distribution of any column: a histogram per layer, with its numbers.

SMAP's ``Analyze/measure/Histograms``.  Localization Statistics draws the
four distributions that have a law to fit (photons, precision, on-time); this
is the one for everything else -- `z_nm`, `logl_rel`, a fitter's `sigma_nm`,
DECODE's `prob`, a column Math Parser made -- where the question is simply
what the values look like and where most of them lie.

**Per layer, overlaid.**  The layers are usually the thing compared: two
channels, or one filtered two ways.  Each visible layer gets its own
histogram in its own colour, read as the layer shows it (its filter, the ROI,
merged blinks or not), on common bins so the bars line up.  With *y* set to
*fraction* layers of different sizes can be compared by shape.

**The range** decides the bins and the numbers.  By default it runs from the
0.1th to the 99.9th percentile of all the layers' values together, SMAP's
default: one wild value (a failed fit at 10^6 nm) would otherwise make every
bin but one empty.  The mean, the standard deviation and the mode are of the
values inside the range, for the same reason; the median and the count are
said for those too, and the text says how many fell outside.

**The bins.**  A bin width set by hand wins.  Otherwise a column of whole
numbers -- `frame`, `n_in_group`, `cluster_id` -- gets bins one wide, so a
bar is a value and not two values smeared together, when the range is no
more than 500 values; anything else gets the Freedman-Diaconis width
(2 IQR n^(-1/3)), held between 20 and 200 bins.  SMAP always made 200,
which turns a few hundred values into a comb.

**The mode** is SMAP's: a parabola through the five bins around the tallest
and its vertex -- the peak of a smooth distribution read finer than a bin.

**Per region** (`per_region`): one run per ROI or cluster, giving the
count, mean, median, standard deviation and mode of the column in each,
as ``<column>_mean`` etc. for the first layer and with ``_L2``, ``_L3`` for
the others -- so *Sort ROIs* or *Filter ROIs* can work on, say, the median z
of every pore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register

#: SMAP's quantile: what is left off each end of the auto range
QUANTILE = 0.001
MIN_BINS, MAX_BINS = 20, 200
#: up to this many whole numbers across the range, one bin per value
MAX_UNIT_BINS = 500


# ------------------------------------------------------------- the algorithm

def value_range(arrays: Sequence[np.ndarray], how: str = "quantile",
                quantile: float = QUANTILE, lo: Optional[float] = None,
                hi: Optional[float] = None) -> Tuple[float, float]:
    """The range the histogram covers, over all the layers' values at once.

    ``how`` is "quantile" (``quantile`` off each end), "all" (smallest to
    largest) or "fixed" (``lo`` and ``hi``; one left None takes the data's).
    """
    values = np.concatenate([a for a in arrays if len(a)]) if any(len(a) for a in arrays) \
        else np.zeros(0)
    if not len(values) and how != "fixed":
        raise ValueError("no values to histogram")
    if how == "all":
        a, b = float(values.min()), float(values.max())
    elif how == "quantile":
        a, b = (float(v) for v in np.quantile(values, [quantile, 1.0 - quantile]))
    elif how == "fixed":
        a = lo if lo is not None else float(values.min())
        b = hi if hi is not None else float(values.max())
    else:
        raise ValueError(f"no range {how!r}")
    if not b > a:
        a, b = a - 0.5, b + 0.5          # one value only: a bin around it
    return float(a), float(b)


def is_whole(values: np.ndarray) -> bool:
    return values.dtype.kind in "iub" or bool(len(values)) and \
        bool(np.all(values == np.round(values)))


def bin_edges(arrays: Sequence[np.ndarray], lo: float, hi: float,
              width: Optional[float] = None) -> np.ndarray:
    """The bin edges over ``[lo, hi]``: ``width`` if given, else one per
    whole number or Freedman-Diaconis (see the module docstring)."""
    if width is not None and width > 0:
        n = int(np.ceil((hi - lo) / width))
        if n > 10 ** 5:
            raise ValueError(f"a bin width of {width:g} makes {n} bins over "
                             f"{lo:g} to {hi:g}: make it wider or the range smaller")
        return lo + width * np.arange(max(n, 1) + 1)
    inside = [a[(a >= lo) & (a <= hi)] for a in arrays]
    values = np.concatenate(inside) if inside else np.zeros(0)
    if len(values) and all(is_whole(a) for a in arrays if len(a)) \
            and hi - lo <= MAX_UNIT_BINS:
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.0)
    if len(values) > 1:
        q1, q3 = np.quantile(values, [0.25, 0.75])
        fd = 2.0 * (q3 - q1) * len(values) ** (-1.0 / 3.0)
        n = int(np.ceil((hi - lo) / fd)) if fd > 0 else MAX_BINS
    else:
        n = MIN_BINS
    return np.linspace(lo, hi, int(np.clip(n, MIN_BINS, MAX_BINS)) + 1)


def mode(counts: np.ndarray, edges: np.ndarray) -> float:
    """SMAP's modal value: the vertex of a parabola through the five bins
    around the tallest one, kept inside them."""
    if not len(counts) or counts.max() <= 0:
        return float("nan")
    centres = 0.5 * (edges[:-1] + edges[1:])
    top = int(np.argmax(counts))
    window = slice(max(0, top - 2), min(len(counts), top + 3))
    x, y = centres[window], counts[window].astype(float)
    if len(x) >= 3:
        a, b, _ = np.polyfit(x, y, 2)
        if a < 0:
            vertex = -b / (2 * a)
            if x[0] <= vertex <= x[-1]:
                return float(vertex)
    return float(centres[top])


@dataclass
class LayerHistogram:
    name: str
    colour: str
    counts: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)


def describe_values(values: np.ndarray, lo: float, hi: float,
                    edges: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """``(counts, stats)`` of one layer's values over the bins."""
    values = values[np.isfinite(values)]
    inside = values[(values >= lo) & (values <= hi)]
    counts, _ = np.histogram(inside, bins=edges)
    nan = float("nan")
    stats = {"n": int(len(inside)), "outside": int(len(values) - len(inside)),
             "mean": float(inside.mean()) if len(inside) else nan,
             "median": float(np.median(inside)) if len(inside) else nan,
             "std": float(inside.std(ddof=1)) if len(inside) > 1 else nan,
             "mode": mode(counts, edges)}
    return counts, stats


# --------------------------------------------------------------- drawing

def draw(ax, column: str, edges: np.ndarray, layers: Sequence[LayerHistogram],
         fraction: bool = False, log_y: bool = False, style: str = "bars") -> None:
    """The histograms, one per layer, overlaid on common bins."""
    width = np.diff(edges)
    several = len(layers) > 1
    for layer in layers:
        # one layer in Localization Statistics' grey: a LUT's colour (hot's
        # yellow) is a curve's colour, too pale for a filled bar on white
        colour = layer.colour if several else "0.55"
        total = max(int(layer.counts.sum()), 1)
        y = layer.counts / total if fraction else layer.counts
        label = f"{layer.name}: median {layer.stats['median']:.4g}"
        if style == "steps":
            ax.stairs(y, edges, color=colour, linewidth=1.2, label=label)
        else:
            ax.bar(edges[:-1], y, width=width, align="edge",
                   color=colour if several else "0.75", alpha=0.45 if several else 1.0,
                   edgecolor=colour, linewidth=0.4, label=label)
        if np.isfinite(layer.stats["mode"]):
            ax.axvline(layer.stats["mode"], color=colour if several else "#d62728",
                       linestyle="--", linewidth=0.8)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(column)
    ax.set_ylabel("fraction" if fraction else "count")
    ax.legend(fontsize=7, frameon=False)


# ---------------------------------------------------------------- the plugin

def _column_choices():
    return [("", "-")]


@dataclass
class HistogramSettings:
    column: str = param("", label="column", choices=_column_choices,
                        help="the column whose values are counted")
    range: str = param("quantile", label="range",
                       choices=(("quantile", "without the extremes"),
                                ("all", "smallest to largest"),
                                ("fixed", "from ... to ...")),
                       help="what the bins cover; the numbers are of the "
                            "values inside it")
    quantile: float = param(QUANTILE, label="leave off", min=0.0, max=0.25,
                            help="the fraction of the values left off each end")
    lo: Optional[float] = param(None, label="from", help="empty: the smallest value")
    hi: Optional[float] = param(None, label="to", help="empty: the largest value")
    bin_width: Optional[float] = param(None, label="bin width", min=0.0,
                                       help="auto: one per whole number, or "
                                            "Freedman-Diaconis")
    y: str = param("count", label="y", choices=(("count", "count"),
                                                ("fraction", "fraction of the layer")),
                   help="fraction compares layers of different sizes by shape")
    style: str = param("bars", label="draw as", choices=(("bars", "bars"),
                                                         ("steps", "outline")),
                       advanced=True, help="outlines read better for several layers")
    log_y: bool = param(False, label="log y", advanced=True,
                        help="for a long tail")


def _layers(ctx: Context) -> List[Tuple[str, str, object]]:
    """``(name, colour, (table, selection))`` per visible localization layer,
    or the context's own selection without a session."""
    from .line_profile import FALLBACK_COLOURS, layer_colour
    session = ctx.session
    found = []
    for i, layer in enumerate(getattr(session, "layers", None) or []):
        if layer.is_image or not layer.visible:
            continue
        taken = [colour for _, colour, _ in found]
        found.append((layer.name, layer_colour(layer.get_display(), len(found), taken),
                      ctx.table(i)))
    if not found:
        found.append((ctx.selection.name or "selection", FALLBACK_COLOURS[0],
                      (ctx.locs, ctx.selection)))
    return found


@register("Analysis/Measure/Histogram")
class Histogram(Plugin):
    """The histogram of any column, per layer, with its mean, median,
    standard deviation and mode."""

    Settings = HistogramSettings
    changes_locs = False
    per_region = True
    # not in a new list of per-region steps: it measures nothing until a
    # column is chosen.  `favorite` also decides the Analysis tab, so it is
    # not pinned there either; the tree has it
    favorite = False
    version = "1"

    def describe(self, ctx: Optional[Context], settings: HistogramSettings) -> FormState:
        """The table's numeric columns to choose from; the quantile or the
        bounds as the range needs."""
        state = FormState(shown={"quantile": settings.range == "quantile",
                                 "lo": settings.range == "fixed",
                                 "hi": settings.range == "fixed"})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            state.choices["column"] = [
                (n, n) for n in ctx.locs.keys()
                if np.asarray(ctx.locs[n]).ndim == 1
                and np.asarray(ctx.locs[n]).dtype.kind in "iufb"]
        return state

    def run(self, ctx: Context, settings: HistogramSettings) -> Result:
        column = settings.column
        if not column:
            raise ValueError("choose a column")
        groups = []
        for name, colour, (table, selection) in _layers(ctx):
            if column not in table:
                raise ValueError(f"layer {name} has no column {column!r}; it has "
                                 + ", ".join(sorted(table.keys())))
            values = np.asarray(selection.apply(table)[column], dtype=float)
            groups.append((name, colour, values[np.isfinite(values)]))
        arrays = [v for _, _, v in groups]
        if not any(len(a) for a in arrays):
            if ctx.per_region:
                return Result(values={f"{column}_n": 0}, settings=settings)
            raise ValueError(f"no values of {column} to histogram")
        lo, hi = value_range(arrays, settings.range, settings.quantile,
                             settings.lo, settings.hi)
        edges = bin_edges(arrays, lo, hi, settings.bin_width)
        layers = []
        for name, colour, values in groups:
            counts, stats = describe_values(values, lo, hi, edges)
            layers.append(LayerHistogram(name, colour, counts, stats))

        values = {}
        for i, layer in enumerate(layers):
            suffix = "" if i == 0 else f"_L{i + 1}"
            for key in ("n", "mean", "median", "std", "mode"):
                values[f"{column}_{key}{suffix}"] = layer.stats[key]
        lines = [f"{column}, {lo:.4g} to {hi:.4g} in {len(edges) - 1} bins"]
        for layer in layers:
            s = layer.stats
            lines.append(f"{layer.name}: n {s['n']}, mean {s['mean']:.4g}, median "
                         f"{s['median']:.4g}, std {s['std']:.4g}, mode {s['mode']:.4g}"
                         + (f"; {s['outside']} outside the range" if s["outside"] else ""))
        fraction = settings.y == "fraction"

        def plot(ax):
            draw(ax, column, edges, layers, fraction, settings.log_y, settings.style)

        return Result(text="\n".join(lines), plot=Plot(plot, size=(5.0, 3.4)),
                      values=values, data={"edges": edges, "layers": layers},
                      settings=settings, title=column)
