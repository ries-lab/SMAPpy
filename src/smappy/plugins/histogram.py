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
the others -- so *Sort regions* or *Filter ROIs* can work on, say, the median z
of every pore.

**Region results** (*source*): the same histogram of what the per-region
steps measured, one value per region -- the plugin that was
*Summarize/Histograms*, merged in (`plugins.MOVED` reads its old path and
settings), so a distribution is drawn one way whatever it is of.  The table
is read as every summary reads it (`roi.SiteAnalysisPlugin`: *results
from*, asked when several could be meant; on *auto* every evaluation on the
ROIs together).  With *attribute* empty every numeric measure is drawn, a
grid of panels, each on its own range and bins.  In place of layers it
overlays groups of regions -- by file, keyword, results column or
localization attribute, `region_groups`, the grouping *Compare groups* and
*Curves* share.  It has no per-region form: a summary of the regions is not
a measurement of one.  The old plugin's fixed 20 equal bins from the
smallest to the largest value are gone with it; a region table has the same
outliers as a localization table, and the same binning serves it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, FormState, Plot, Result, param, register
from .. import region_results
from ..region_groups import group_choices, group_labels, numeric_fields, ordered
from ..region_results import Evaluation
from .roi import SiteAnalysisPlugin, auto_only, choose_evaluation, grid_shape, measure_title

#: SMAP's quantile: what is left off each end of the auto range
QUANTILE = 0.001
MIN_BINS, MAX_BINS = 20, 200
#: up to this many whole numbers across the range, one bin per value
MAX_UNIT_BINS = 500


# ------------------------------------------------------------- the algorithm

def value_range(arrays: Sequence[np.ndarray], how: str = "quantile",
                quantile: float = QUANTILE, lo: Optional[float] = None,
                hi: Optional[float] = None, whole: bool = False) -> Tuple[float, float]:
    """The range the histogram covers, over all the layers' values at once.

    ``how`` is "quantile" (``quantile`` off each end), "all" (smallest to
    largest) or "fixed" (``lo`` and ``hi``; one left None takes the data's).
    With ``whole`` the quantiles are values of the data, never between two
    (numpy's "lower" and "higher"): what is left off is whole values, none
    while ``quantile`` is under one value's share.  A region table has tens
    of rows, and the interpolated 0.1th percentile of 40 regions would leave
    the smallest out.
    """
    values = np.concatenate([a for a in arrays if len(a)]) if any(len(a) for a in arrays) \
        else np.zeros(0)
    if not len(values) and how != "fixed":
        raise ValueError("no values to histogram")
    if how == "all":
        a, b = float(values.min()), float(values.max())
    elif how == "quantile":
        if whole:
            a = float(np.quantile(values, quantile, method="lower"))
            b = float(np.quantile(values, 1.0 - quantile, method="higher"))
        else:
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
         fraction: bool = False, log_y: bool = False, style: str = "bars",
         legend: bool = True) -> None:
    """The histograms, one per layer (or group of regions), overlaid on
    common bins."""
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
    if legend:
        ax.legend(fontsize=7, frameon=False)


def histograms_of(groups: Sequence[Tuple[str, str, np.ndarray]], how: str = "quantile",
                  quantile: float = QUANTILE, lo: Optional[float] = None,
                  hi: Optional[float] = None, width: Optional[float] = None,
                  whole: bool = False) -> Tuple[np.ndarray, List[LayerHistogram]]:
    """``(edges, histograms)`` of ``(name, colour, values)`` groups -- layers,
    or groups of regions -- on common bins: `value_range`, `bin_edges`,
    `describe_values` in turn.  Non-finite values are dropped."""
    arrays = [np.asarray(v, dtype=float) for _, _, v in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    a, b = value_range(arrays, how, quantile, lo, hi, whole)
    edges = bin_edges(arrays, a, b, width)
    return edges, [LayerHistogram(name, colour, *describe_values(v, a, b, edges))
                   for (name, colour, _), v in zip(groups, arrays)]


def region_histograms(rows: Sequence[Dict], fields: Sequence[str],
                      groups: Optional[Sequence[str]] = None, **binning
                      ) -> Dict[str, Dict]:
    """One histogram per column of a region table, a bar set per group.

    ``rows`` are `region_results.rows`' (one per region), ``groups`` each
    row's group (`region_groups.group_labels`; None: one, ``all``) and
    ``binning`` `histograms_of`'s keywords, applied to every column (the
    quantiles ``whole`` unless said otherwise).  Per
    column: ``edges``, ``groups`` (`LayerHistogram` each), ``values`` (every
    finite one), ``roi_ids`` (whose they are) and ``missing`` (regions
    without one).  A column no region has a value for has no ``edges``.
    """
    from ..region_groups import ordered
    labels = list(groups) if groups is not None else ["all"] * len(rows)
    names = ordered(labels)
    g = np.asarray(labels, dtype=object)
    out = {}
    for f in fields:
        values = np.asarray([_number(r.get(f)) for r in rows], dtype=float)
        finite = np.isfinite(values)
        entry = {"values": values[finite], "missing": int((~finite).sum()),
                 "roi_ids": [r.get("roi_id") for r, ok in zip(rows, finite) if ok]}
        if finite.any():
            entry["edges"], entry["groups"] = histograms_of(
                [(name, f"C{k}", values[(g == name) & finite])
                 for k, name in enumerate(names)], **{"whole": True, **binning})
        out[f] = entry
    return out


def _number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- the plugin

SOURCES = (("localizations", "localizations"), ("regions", "region results"))


def _column_choices():
    return [("", "-")]


def _no_groups():
    return [("", "-")]


@dataclass
class HistogramSettings:
    source: str = param("localizations", label="source", choices=SOURCES,
                        help="the localizations' attributes, per layer; or what "
                             "the per-region steps measured, one value per region")
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the measurement whose columns are drawn, on the "
                                 "ROIs or per region of an id attribute; auto: "
                                 "everything measured on the ROIs")
    column: str = param("", label="attribute", choices=_column_choices,
                        help="the attribute whose values are counted; for region "
                             "results empty draws every measure, or name several "
                             "separated by commas")
    group_by: str = param("", label="group by", choices=_no_groups,
                          help="a histogram per group of regions: file, an ROI "
                               "keyword, a results column or an attribute with few "
                               "values; empty: one")
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
                   help="fraction compares layers (or groups) of different sizes "
                        "by shape")
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


def _stats_line(h: LayerHistogram) -> str:
    s = h.stats
    return (f"{h.name}: n {s['n']}, mean {s['mean']:.4g}, median {s['median']:.4g}, "
            f"std {s['std']:.4g}, mode {s['mode']:.4g}"
            + (f"; {s['outside']} outside the range" if s["outside"] else ""))


@register("Analysis/Measure/Histogram")
class Histogram(SiteAnalysisPlugin):
    """The histogram of any attribute, per layer -- or of what was measured
    per region, per group of regions -- with its mean, median, standard
    deviation and mode."""

    Settings = HistogramSettings
    name = "Histogram of an attribute"
    changes_locs = False
    per_region = True
    version = "1"
    # what *results from* on auto reads, said in its hint
    auto_label = "all"

    # ------------------------------------------------------ the region source

    def evaluation(self, ctx: Context, settings) -> Optional[Evaluation]:
        """The evaluation the region source reads.  Named: that one.  On
        auto, every evaluation on the ROIs together (None) -- a histogram of
        each column is the overview, and nothing about it is ambiguous --
        unless a chain made its own per id column, or nothing was measured on
        the ROIs: then as every summary chooses (`choose_evaluation`), which
        asks when several id-column evaluations could be meant."""
        chosen = getattr(settings, "evaluation", "")
        if chosen:
            return choose_evaluation(ctx, "", chosen)
        own = [Evaluation.parse(name) for name, _ in ctx.evaluations]
        found = own or region_results.available(ctx)
        if not found or any(ev.on_rois for ev in found) and (
                not own or all(ev.on_rois for ev in own)):
            return None
        return choose_evaluation(ctx, "", "")

    def preflight(self, ctx: Context, settings: HistogramSettings):
        if settings.source != "regions":
            return None
        return super().preflight(ctx, settings)

    # ---------------------------------------------------------------- form

    def describe(self, ctx: Optional[Context], settings: HistogramSettings) -> FormState:
        """The fields of the chosen source only; the table's numeric columns
        to choose from (the localizations', or the results'); the quantile
        or the bounds as the range needs."""
        regions = settings.source == "regions"
        per_region = ctx is not None and ctx.per_region
        if regions and ctx is not None and not per_region:
            state = super().describe(ctx, settings)
        else:
            state = FormState()
        state.shown.update({"source": not per_region,
                            "evaluation": regions, "group_by": regions,
                            "quantile": settings.range == "quantile",
                            "lo": settings.range == "fixed",
                            "hi": settings.range == "fixed"})
        if ctx is None or ctx.session is None:
            return state
        if not regions:
            if len(ctx.locs):
                state.choices["column"] = [
                    (n, n) for n in ctx.locs.keys()
                    if np.asarray(ctx.locs[n]).ndim == 1
                    and np.asarray(ctx.locs[n]).dtype.kind in "iufb"]
            return state
        try:
            rows, ev = self.rows(ctx, settings)
        except Exception:                                  # noqa: BLE001
            return state
        state.choices["column"] = [("", "every measure")] + [
            (n, n) for n in numeric_fields(rows)]
        lists = list(getattr(ctx.rois, "annotation_lists", {}) or {}) \
            if ev is None or ev.on_rois else []
        state.choices["group_by"] = group_choices(ctx, rows, ev, extra=lists)
        return state

    # ----------------------------------------------------------------- run

    def run(self, ctx: Context, settings: HistogramSettings) -> Result:
        if settings.source == "regions":
            return self._regions(ctx, settings)
        return self._localizations(ctx, settings)

    def _binning(self, settings: HistogramSettings) -> Dict:
        return dict(how=settings.range, quantile=settings.quantile, lo=settings.lo,
                    hi=settings.hi, width=settings.bin_width)

    def _localizations(self, ctx: Context, settings: HistogramSettings) -> Result:
        column = settings.column
        if not column:
            raise ValueError("choose an attribute")
        groups = []
        for name, colour, (table, selection) in _layers(ctx):
            if column not in table:
                raise ValueError(f"layer {name} has no attribute {column!r}; it has "
                                 + ", ".join(sorted(table.keys())))
            values = np.asarray(selection.apply(table)[column], dtype=float)
            groups.append((name, colour, values[np.isfinite(values)]))
        arrays = [v for _, _, v in groups]
        if not any(len(a) for a in arrays):
            if ctx.per_region:
                return Result(values={f"{column}_n": 0}, settings=settings)
            raise ValueError(f"no values of {column} to histogram")
        edges, layers = histograms_of(groups, **self._binning(settings))
        lo, hi = value_range(arrays, settings.range, settings.quantile,
                             settings.lo, settings.hi)

        values = {}
        for i, layer in enumerate(layers):
            suffix = "" if i == 0 else f"_L{i + 1}"
            for key in ("n", "mean", "median", "std", "mode"):
                values[f"{column}_{key}{suffix}"] = layer.stats[key]
        lines = [f"{column}, {lo:.4g} to {hi:.4g} in {len(edges) - 1} bins"]
        lines += [_stats_line(layer) for layer in layers]
        fraction = settings.y == "fraction"

        def plot(ax):
            draw(ax, column, edges, layers, fraction, settings.log_y, settings.style)

        return Result(text="\n".join(lines), plot=Plot(plot, size=(5.0, 3.4)),
                      values=values, data={"edges": edges, "layers": layers},
                      settings=settings, title=column)

    def _regions(self, ctx: Context, settings: HistogramSettings) -> Result:
        """What the per-region steps measured, a histogram per measure and a
        bar set per group of regions."""
        if ctx.per_region:
            raise ValueError("region results are counted over the regions, not in "
                             "one: set *source* to localizations to run per region")
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("nothing measured in the regions yet: run the "
                             "per-region steps first")
        named = [f.strip() for f in settings.column.split(",") if f.strip()]
        fields = named or numeric_fields(rows, exclude=(settings.group_by,))
        missing = [f for f in named if not any(f in r for r in rows)]
        if missing:
            raise ValueError(f"no measure {', '.join(missing)}: the results have "
                             + ", ".join(sorted(rows[0])))
        labels = group_labels(ctx, rows, ev, settings.group_by)
        data = region_histograms(rows, fields, labels, **self._binning(settings))
        # a measure no region has a value for is said, not drawn as an empty
        # panel (NPC's labelling-efficiency columns before it has run, say)
        empty = [f for f, d in data.items() if "edges" not in d]
        shown = {f: d for f, d in data.items() if "edges" in d}
        if not shown:
            raise ValueError("no numeric columns with values to histogram"
                             + (f" ({', '.join(empty)} have none)" if empty else ""))
        what = "ROIs" if ev is None or ev.on_rois else "regions"
        names = ordered(labels)
        head = f"{len(rows)} {what}, {len(shown)} measure" + ("s" if len(shown) > 1 else "")
        if settings.group_by:
            head += f", {len(names)} groups by {settings.group_by}"
        if empty:
            head += f"; no values for {', '.join(empty)}"
        lines = [head]
        for f, d in shown.items():
            lines.append(f"{f}, {d['edges'][0]:.4g} to {d['edges'][-1]:.4g} in "
                         f"{len(d['edges']) - 1} bins")
            lines += ["  " + _stats_line(h) for h in d["groups"]]
        fraction = settings.y == "fraction"
        several = len(names) > 1

        def one(ax, f, d, first_column, legend) -> None:
            draw(ax, f, d["edges"], d["groups"], fraction, settings.log_y,
                 settings.style, legend=several and legend)
            ax.set_xlabel("")
            ax.set_title(measure_title(f), fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(("fraction of " if fraction else "") + what
                          if first_column else "", fontsize=8)
            if d["missing"]:
                ax.text(0.98, 0.95, f"{d['missing']} without a value",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="0.4")

        if len(shown) == 1:
            (f, d), = shown.items()
            plot = Plot(lambda ax: one(ax, f, d, True, True), size=(5.0, 3.4))
        else:
            n_rows, across = grid_shape(len(shown))

            def grid(figure) -> None:
                """A grid, up to four across, one histogram per measure:
                as many panels as the table has columns, which is why it
                declares them and is handed the figure."""
                axes = figure.subplots(n_rows, across, squeeze=False).ravel()
                for i, (axis, (f, d)) in enumerate(zip(axes, shown.items())):
                    one(axis, f, d, i % across == 0, i == 0)
                for axis in axes[len(shown):]:
                    axis.set_visible(False)

            plot = Plot(draw=grid, panels=len(shown), size=(3.2 * across, 2.6 * n_rows))
        return Result(text="\n".join(lines), plot=plot,
                      data={"histograms": data, "rows": rows, "groups": names},
                      settings=settings,
                      title=next(iter(shown)) if len(shown) == 1 else "region results")
