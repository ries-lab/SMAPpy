"""Curves: the mean curve of many regions, with its spread, per condition.

A per-region step that measures a curve -- Spatial statistics' g(r) and
L(r) - r in every ROI, later the MSD of every track -- leaves one curve per
region in the region table's arrays (`Result.arrays`).  One curve says little;
the question is what they are like together, and whether two conditions
differ.  So: every region's arrays of one evaluation, the mean at each x with
its standard deviation or standard error over the regions, a line per group.
`docs/smap_porting/wave4_design.md`, step 17b.

**Groups** are an ROI keyword list (`ROI.annotations`, *Annotate ROIs*): with
*group by* set to a list, the ROIs with one keyword are one curve, those with
another a second, those with none a third; so "control" and "treated" ROIs
of one data set are compared in one plot.  Regions of an id column (clusters)
carry no keywords and are one group.

**Pooled where the evaluator says how** (`Plugin.pool`).  A curve that is a
ratio -- pairs found over pairs expected -- is better summed than averaged:
the mean of the regions' g weighs an ROI with 20 pairs in a bin as much as one
with 20,000, and its noise with it.  An evaluator that hands back numerator
and denominator among its arrays implements `pool`, which makes the curves
from the arrays summed over a group's regions; for Spatial statistics that
is the g of all the group's regions taken together (exactly the g of their
union when their windows do not overlap), and K and L - r from it.  The
spread is still the regions' own: the standard deviation of their curves
around their mean, so a pooled curve is drawn with the band that says how
much the regions differ.

**One x grid, or none.**  The regions' x arrays must be the same (to a part
in 10^6); a region on another grid is refused, naming it, rather than
interpolated.  A different grid means the evaluator ran with other settings
(another bin width or range), and then the curves are not one measurement:
a pooled count summed across bins of different widths is wrong, and a mean
of curves with different smoothing is not a mean of anything.  Running the
evaluation again with one setting is the remedy, and the message says so.

The values (`Result.values` of each region) are summarised too: the mean of
each numeric column per group, in the text, beside the number of regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Result, param, register
from .. import region_results
from ..region_results import Evaluation
from .roi import (AmbiguousEvaluation, EVALUATION_HELP, SiteAnalysisPlugin, auto_only,
                  grid_shape, measure_title)

# the x arrays of two regions are one grid when they agree to this, relative
# to the grid's span (rounding in a saved table, not another setting)
GRID_TOLERANCE = 1e-6
# names an x axis is recognised by, first match wins (*x* on auto)
X_NAMES = ("r_nm", "x_nm", "distance_nm", "lag_s", "lag_frames", "t_s", "time_s",
           "frame")
NO_GROUP = "(none)"
# what a panel is called, for the curves of the shipped evaluators (a layer's
# suffix, `_L2`, kept after it); any other by its name and unit
TITLES = {"g": "g(r)", "K_nm2": "K(r) (nm²)", "lr_nm": "L(r) - r (nm)",
          "pairs": "pairs", "expected": "expected pairs"}
SPREADS = (("sd", "standard deviation"), ("sem", "standard error"))


# ------------------------------------------------------------- the algorithm

@dataclass
class Region:
    """One region's results: its id, its group, its arrays and values."""
    id: str
    arrays: Dict[str, np.ndarray]
    values: Dict[str, Any] = field(default_factory=dict)
    group: str = NO_GROUP


@dataclass
class GroupCurves:
    """What `summarise` gives per group."""
    name: str
    n: int
    ids: List[str]
    mean: Dict[str, np.ndarray]            # the centre: the mean, or the pooled curve
    spread: Dict[str, np.ndarray]          # SD or SEM over the regions
    count: Dict[str, np.ndarray]           # regions with a finite value at each x
    pooled: List[str]                      # the names whose centre is pooled
    values: Dict[str, float]               # the mean of each value column


def strictly_increasing(a) -> bool:
    a = np.asarray(a, dtype=float)
    return a.ndim == 1 and len(a) > 1 and bool(np.all(np.diff(a) > 0))


def x_axis(regions: Sequence[Region], chosen: str = "") -> str:
    """The array that is the x axis: ``chosen``, else the first of `X_NAMES`
    every region has, else the first array that increases in every region."""
    if not regions:
        raise ValueError("no regions with curves")
    common = [n for n in regions[0].arrays if all(n in r.arrays for r in regions)]
    if chosen:
        if chosen not in common:
            raise ValueError(f"not every region has an array {chosen!r}; they have "
                             + ", ".join(common))
        return chosen
    for name in X_NAMES:
        if name in common:
            return name
    for name in common:
        if all(strictly_increasing(r.arrays[name]) for r in regions):
            return name
    raise ValueError("no array increases in every region to be the x axis: "
                     "choose *x*; the arrays are " + ", ".join(common))


def common_grid(regions: Sequence[Region], x_name: str) -> np.ndarray:
    """The x grid every region shares, or a refusal naming the first region
    on another (see the module docstring for why there is no interpolation)."""
    x = np.asarray(regions[0].arrays[x_name], dtype=float)
    span = float(np.ptp(x)) if len(x) else 0.0
    for r in regions[1:]:
        other = np.asarray(r.arrays[x_name], dtype=float)
        if other.shape != x.shape or not np.allclose(
                other, x, rtol=0.0, atol=GRID_TOLERANCE * max(span, 1e-300)):
            def says(a):
                return (f"{len(a)} points from {a[0]:g} to {a[-1]:g}" if len(a)
                        else "no points")
            raise ValueError(
                f"region {r.id} has its {x_name} on another grid ({says(other)}) than "
                f"region {regions[0].id} ({says(x)}): the curves were measured with "
                "different settings; run the evaluation again with one setting")
    return x


def curve_names(regions: Sequence[Region], x_name: str, n: int,
                chosen: Sequence[str] = (), pooled: Sequence[str] = ()) -> List[str]:
    """The curves to summarise: ``chosen``, else the pooled ones when the
    evaluator pools, else every array of the x axis's length but the x axis
    and envelope edges."""
    common = [k for k in regions[0].arrays if all(k in r.arrays for r in regions)]
    if chosen:
        missing = [c for c in chosen if c not in common]
        if missing:
            raise ValueError(f"not every region has {', '.join(missing)}; the arrays "
                             "are " + ", ".join(common))
        return list(chosen)
    if pooled:
        return [k for k in common if k in pooled]
    return [k for k in common if k != x_name and not _is_band(k)
            and all(np.asarray(r.arrays[k]).shape == (n,) for r in regions)]


def _is_band(name: str) -> bool:
    """``g_lo``, ``lr_nm_hi_L2``: an envelope's edge, not a curve to average
    (auto *curves*)."""
    parts = name.split("_")
    return "lo" in parts[1:] or "hi" in parts[1:]


def spread_of(stack: np.ndarray, how: str = "sd") -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray]:
    """``(mean, spread, count)`` of curves stacked as rows, NaN ignored: the
    standard deviation (n - 1) or the standard error of the mean, NaN where
    fewer than two regions have a value."""
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        dev = np.where(finite, stack - mean, 0.0)
        var = np.where(count > 1, (dev ** 2).sum(axis=0) / np.maximum(count - 1, 1),
                       np.nan)
        sd = np.sqrt(var)
        spread = sd / np.sqrt(count) if how == "sem" else sd
    return mean, spread, count


def summarise(regions: Sequence[Region], x_name: str = "", names: Sequence[str] = (),
              spread: str = "sd",
              pool: Optional[Callable[[Dict[str, np.ndarray], np.ndarray],
                                      Dict[str, np.ndarray]]] = None
              ) -> Tuple[str, np.ndarray, List[str], List[GroupCurves]]:
    """Per group, the mean curve (or the pooled one) and its spread.

    ``regions`` with their groups; ``x_name`` the x axis (auto when empty);
    ``names`` the curves (auto when empty); ``pool`` the evaluator's
    `Plugin.pool`, given every array summed over a group's regions.
    Returns ``(x_name, x, names, groups)``, the groups in order of first
    appearance with the group-less last.
    """
    regions = [r for r in regions if r.arrays]
    if not regions:
        raise ValueError("no region has curves: run a per-region step that "
                         "measures one (Spatial statistics) first")
    x_name = x_axis(regions, x_name)
    x = common_grid(regions, x_name)
    n = len(x)
    pooled_names: List[str] = []
    if pool is not None:
        probe = pool({k: np.asarray(v, float) for k, v in regions[0].arrays.items()
                      if np.asarray(v).shape == (n,)}, x)
        pooled_names = list(probe)
    names = curve_names(regions, x_name, n, names, pooled_names)
    if not names:
        raise ValueError(f"no curves on the grid of {x_name}")
    order: List[str] = []
    for r in regions:
        if r.group not in order:
            order.append(r.group)
    if NO_GROUP in order:
        order.remove(NO_GROUP)
        order.append(NO_GROUP)
    groups = []
    for g in order:
        mine = [r for r in regions if r.group == g]
        mean, spr, count = {}, {}, {}
        for name in names:
            stack = np.vstack([np.asarray(r.arrays[name], dtype=float) for r in mine])
            mean[name], spr[name], count[name] = spread_of(stack, spread)
        used: List[str] = []
        if pool is not None:
            sums = {}
            for k in mine[0].arrays:
                if all(np.asarray(r.arrays.get(k, np.zeros(0))).shape == (n,) for r in mine):
                    sums[k] = np.sum([np.nan_to_num(np.asarray(r.arrays[k], float))
                                      for r in mine], axis=0)
            for k, v in pool(sums, x).items():
                if k in mean:
                    mean[k] = np.asarray(v, dtype=float)
                    used.append(k)
        groups.append(GroupCurves(g, len(mine), [r.id for r in mine], mean, spr, count,
                                  used, mean_values(mine)))
    return x_name, x, list(names), groups


def mean_values(regions: Sequence[Region]) -> Dict[str, float]:
    """The mean of each numeric value column over the regions that have it."""
    keys: List[str] = []
    for r in regions:
        for k, v in r.values.items():
            if k not in keys and isinstance(v, (int, float, np.integer, np.floating)) \
                    and not isinstance(v, bool):
                keys.append(k)
    out = {}
    for k in keys:
        v = np.asarray([r.values.get(k, np.nan) for r in regions], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    return out


# --------------------------------------------------------------- drawing

def curve_title(name: str) -> str:
    """``lr_nm_L2`` -> ``L(r) - r (nm), L2``."""
    for stem, title in TITLES.items():
        if name == stem:
            return title
        if name.startswith(stem + "_L"):
            return f"{title}, {name[len(stem) + 1:]}"
    return measure_title(name)


def draw(figure, x_name: str, x: np.ndarray, names: Sequence[str],
         groups: Sequence[GroupCurves], spread: str = "sd") -> None:
    """A panel per curve, a line per group with its spread shaded."""
    rows, across = grid_shape(len(names), columns=3)
    axes = figure.subplots(rows, across, squeeze=False).ravel()
    word = dict(SPREADS)[spread]
    for i, (ax, name) in enumerate(zip(axes, names)):
        for k, g in enumerate(groups):
            colour = "0.35" if g.name == NO_GROUP and len(groups) > 1 else f"C{k}"
            m, s = g.mean[name], g.spread[name]
            ax.fill_between(x, m - s, m + s, color=colour, alpha=0.2, linewidth=0)
            label = f"{g.name}: {g.n}" + (", pooled" if name in g.pooled else "")
            ax.plot(x, m, color=colour, linewidth=1.3, label=label)
        ax.set_xlabel(measure_title(x_name), fontsize=8)
        ax.set_title(curve_title(name), fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, title=f"± {word}", title_fontsize=7)
    for ax in axes[len(names):]:
        ax.set_visible(False)


# ---------------------------------------------------------------- the plugin

def _no_groups():
    return [("", "-")]


def _no_arrays():
    return [("", "auto")]


@dataclass
class CurvesSettings:
    evaluation: str = param("", label="results from", choices=auto_only,
                            help=EVALUATION_HELP)
    x: str = param("", label="x", choices=_no_arrays,
                   help="the array that is the x axis; auto: r_nm and the like")
    curves: str = param("", label="curves",
                        help="comma separated; empty: the pooled curves, or every "
                             "array on the x grid")
    spread: str = param("sd", label="spread", choices=SPREADS,
                        help="over the regions: how much they differ (SD), or how "
                             "well the mean is known (SEM)")
    group_by: str = param("", label="group by", choices=_no_groups,
                          help="an ROI keyword list: a curve per keyword")


def has_arrays(ctx, ev: Evaluation) -> bool:
    found = region_results.table(ctx, ev)
    return found is not None and bool(found.arrays)


@register("Analysis/Regions/Summarize/Curves")
class Curves(SiteAnalysisPlugin):
    """The mean curve of an evaluation over the regions, with its standard
    deviation or error, a line per ROI keyword: pooled where the evaluator
    gives counts (Spatial statistics' pairs over expected pairs)."""

    Settings = CurvesSettings
    changes_locs = False
    version = "1"

    def evaluation(self, ctx: Context, settings) -> Optional[Evaluation]:
        """As every analysis chooses (`choose_evaluation`), among the
        evaluations that kept curves: an ROI's Corners or Cluster shape has
        none to average, and asking about them would be a question with a
        wrong answer in it."""
        chosen = getattr(settings, "evaluation", "")
        if chosen:
            from .roi import choose_evaluation
            return choose_evaluation(ctx, "", chosen)
        own = [Evaluation.parse(name) for name, _ in ctx.evaluations]
        if own:
            return own[-1]
        found = [ev for ev in region_results.available(ctx) if has_arrays(ctx, ev)]
        if len(found) == 1:
            return found[0]
        if not found:
            return None
        raise AmbiguousEvaluation("", found)

    def describe(self, ctx, settings) -> FormState:
        state = super().describe(ctx, settings)
        if ctx is None:
            return state
        names = state.choices.get("evaluation", [])
        state.choices["evaluation"] = [
            (v, label) for v, label in names
            if not v or has_arrays(ctx, Evaluation.parse(v))]
        project = ctx.rois
        lists = list(getattr(project, "annotation_lists", {}) or {})
        state.choices["group_by"] = [("", "-")] + [(n, n) for n in lists]
        try:
            ev = self.evaluation(ctx, settings)
        except (AmbiguousEvaluation, ValueError):
            ev = None
        found = region_results.table(ctx, ev) if ev is not None else None
        if found is not None and found.arrays:
            state.choices["x"] = [("", "auto")] + [(n, n) for n in found.arrays]
            state.hints["curves"] = ", ".join(n for n in found.arrays if not _is_band(n))
        if ev is not None and not ev.on_rois and settings.group_by:
            state.notice = Notice("regions of an id column have no keywords: "
                                  "one group", kind="info")
        return state

    def regions(self, ctx: Context, ev: Evaluation, group_by: str = "") -> List[Region]:
        """Every current region of ``ev`` with its arrays, values and group."""
        out = []
        if ev.on_rois:
            project = ctx.rois
            if project is None:
                return out
            for roi_id in project.ordered_ids():
                roi = project.rois[roi_id]
                if not roi.use:
                    continue
                found = project.evaluations(roi_id).get(ev.label)
                if not found or "values" not in found[0]:
                    continue
                entry = found[0]
                group = (roi.annotations.get(group_by) or NO_GROUP) if group_by else NO_GROUP
                out.append(Region(str(roi.number), _arrays(entry), entry.get("values", {}),
                                  group))
            return out
        if not region_results.is_current(ctx, ev):
            raise ValueError(region_results.missing(ctx, ev))
        table = region_results.table(ctx, ev)
        for region_id in table.ids:
            entry = table.entry(region_id)
            if entry and "values" in entry:
                out.append(Region(str(region_id), _arrays(entry), entry["values"]))
        return out

    def _pool(self, ctx: Context, ev: Evaluation):
        """The evaluator's `pool`, when one plugin made every region."""
        from . import get
        made = region_results.steps(ctx, ev)
        paths = {m.get("plugin") for m in made}
        if len(paths) != 1:
            return None
        try:
            plugin = get(next(iter(paths)))()
        except Exception:                                  # noqa: BLE001
            return None
        return plugin.pool

    def run(self, ctx: Context, settings: CurvesSettings) -> Result:
        if ctx.site_table is not None and not ctx.evaluations:
            raise ValueError("Curves reads the curves kept with the regions, which a "
                             "table of values alone does not have")
        ev = self.evaluation(ctx, settings)
        if ev is None:
            raise ValueError("no results with curves in the regions yet: run a "
                             "per-region step that measures one (Spatial statistics)")
        regions = self.regions(ctx, ev, settings.group_by if ev.on_rois else "")
        named = [c.strip() for c in settings.curves.split(",") if c.strip()]
        x_name, x, names, groups = summarise(regions, settings.x, named, settings.spread,
                                             self._pool(ctx, ev))
        what = "ROIs" if ev.on_rois else "regions"
        lines = [f"{ev}: {sum(g.n for g in groups)} {what}, {len(names)} curves over "
                 f"{x_name} ({len(x)} points)"]
        if any(g.pooled for g in groups):
            lines[0] += "; pooled: " + ", ".join(groups[0].pooled)
        for g in groups:
            head = f"{g.name}: {g.n} {what}" if settings.group_by else f"{g.n} {what}"
            means = ", ".join(f"{k} {v:.4g}" for k, v in g.values.items())
            lines.append(head + (f"; mean {means}" if means else ""))
        data = {"evaluation": str(ev), "x_name": x_name, "x": x, "curves": names,
                "spread": settings.spread,
                "groups": {g.name: {"n": g.n, "ids": g.ids, "mean": g.mean,
                                    "spread": g.spread, "count": g.count,
                                    "pooled": g.pooled, "values": g.values}
                           for g in groups}}
        rows, across = grid_shape(len(names), columns=3)

        def plot(figure) -> None:
            draw(figure, x_name, x, names, groups, settings.spread)

        return Result(text="\n".join(lines), data=data, settings=settings,
                      plot=Plot(plot, panels=len(names),
                                size=(3.4 * across, 2.8 * rows)),
                      title="curves")


def _arrays(entry: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) for k, v in (entry.get("arrays") or {}).items()
            if np.asarray(v).dtype.kind in "iufb"}
