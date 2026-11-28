"""Region statistics: the localizations of every region at once.

The algorithm is `smappy.id_statistics` (its docstring has the decisions);
this is the wrapper.  A region is an ROI of the ROI manager or the rows that
share an id -- a MINFLUX trace (`tid`), a linked track, a cluster --
everything a block over the ROIs or that column would measure one region at
a time with *Localization statistics*, counted here in one pass, so 10^5
traces take a second instead of minutes.  An id column's regions are a
partition and are coded by a lookup table (`id_statistics`); the ROIs may
overlap, so their rows are joined into one index first and measured by the
same pass (`member_statistics`).  *shape* adds *Cluster shape*'s numbers
(`id_statistics.region_shapes`, which *Cluster shape* calls for its one
region) for every region at once.

What is summarised is one list, *attributes*, in groups -- position, time,
localization, MINFLUX, other -- whose headers tick their members (version
4, 2026-11; before, *common attributes* and *use z* were ticks beside a list
of extras, and the position and time were left out of it, which hid them).
A region's centre is measured whether or not its axis is ticked, because it
is where the region is; ticking an axis adds its spread and the statistics.
A ticked name the table lacks is passed over with a warning, not refused,
so one chain runs over MINFLUX and camera files alike -- silently for the
default names, which no file has all of.

It was *Id statistics* until 2026-11, when the ROIs were added beside the id
columns: what decides between it and the per-region steps is the task --
cheap numbers over many regions here, a measurement that needs each
region's own context there -- not where the regions come from.  The old
path is read as this one (`plugins.MOVED`, `moved_settings`).

What it hands back is a region table, not columns: the table of every
region, signed as a block's run would sign it -- an id column's by
`region_runs.column_table`, stored with the column (`Session.id_regions`),
"per tid: Region statistics"; the ROIs' row by row with each ROI's own
signature, stored with the ROIs as any per-region step's results are --
where *Histogram* on region results, *results from*, *Filter ROIs* and the
region filter (`docs/region-filter.md`) find it; *Write results* writes a
value onto the localizations or the ROIs where one is wanted there.

The localizations are the layer's own (its filter, not the drawn ROI, which
a block's regions ignore too), ungrouped: a trace's statistics are of its
localizations, and a merged blink of a trace would be the trace itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values

#: what *auto* takes the regions from, in order: MINFLUX's traces, linked
#: tracks, clusters, a simulation's tracks
AUTO_IDS = ("tid", "track_id", "cluster_id", "track_id_true")
#: the groups *attributes* shows its names in, a header each that ticks them
GROUPS = (
    ("position", ("x_nm", "y_nm", "z_nm")),
    ("time", ("time_s", "frame")),
    ("localization", ("photons", "background", "xy_err_nm", "z_err_nm", "sigma_nm",
                      "logl_rel")),
    ("MINFLUX", ("cfr", "efo", "dcr")),
)
#: what is ticked unless chosen otherwise: where the region is and when, and
#: the camera fit's brightness and precision -- not MINFLUX's quality, which
#: is asked for where it is wanted.  Passed over silently where a table
#: lacks one
DEFAULT_ATTRIBUTES = ("x_nm", "y_nm", "z_nm", "time_s", "frame", "photons", "xy_err_nm")
#: the names that are no attribute of their own: the position and the time
#: are measured their own way (`id_statistics`)
POSITIONS = {"x_nm": "x", "y_nm": "y", "z_nm": "z"}
TIMES = ("time_s", "frame")
#: the evaluation's name, what *results from* shows after "per <column>: "
LABEL = "Region statistics"
#: the *regions* value that names the ROI manager's ROIs (a block's own word)
ROIS = "rois"


def _id_choices():
    return [("", "auto")] + [(c, c) for c in AUTO_IDS]


def _region_choices():
    return [("", "auto"), (ROIS, "ROIs")] + [(c, c) for c in AUTO_IDS]


def _has_rois(ctx) -> bool:
    project = None if ctx is None else ctx.rois
    return bool(project is not None and any(r.use for r in project.rois.values()))


def regions_source(ctx, chosen: str = "") -> str:
    """What the regions are: `ROIS`, or an id column (`id_column`).  On
    *auto* an id column where the table has one of `AUTO_IDS`, else the ROIs
    where there are any -- an id column first, because a MINFLUX file's
    traces are what one measures there whatever ROIs were drawn."""
    chosen = (chosen or "").strip()
    if chosen == ROIS:
        if not _has_rois(ctx):
            raise ValueError("there are no ROIs to measure: find or draw some in "
                             "the ROI manager first, or choose an id attribute")
        return ROIS
    if not chosen and not any(c in ctx.locs for c in AUTO_IDS) and _has_rois(ctx):
        return ROIS
    return id_column(ctx.locs, chosen)


def moved_settings(values):
    """*Id statistics*' saved settings as this plugin's: *id attribute* is
    *regions*; its typed *attributes*, empty for the common ones (MINFLUX's
    and the camera fit's), the default list with those or with the typed
    names; its *use z* off, z left out."""
    out = dict(values)
    if "id_column" in out:
        out["regions"] = out.pop("id_column")
    typed = out.pop("columns", None)
    names = list(DEFAULT_ATTRIBUTES)
    if isinstance(typed, str):
        extra = [n for n in typed.replace(",", " ").split() if n]
        names += [n for n in (extra or ["cfr", "efo", "dcr"]) if n not in names]
    if out.pop("use_z", True) is False:
        names.remove("z_nm")
    out["attributes"] = names
    return out


def id_column(locs, chosen: str = "") -> str:
    """The column naming the regions: ``chosen``, or on *auto* the first of
    `AUTO_IDS` the table has.  ValueError naming the integer columns there
    are when there is none."""
    from ..tracking.columns import integer_columns
    column = (chosen or "").strip() or next((c for c in AUTO_IDS if c in locs), "")
    if column and column in locs:
        return column
    have = integer_columns(locs)
    start = (f"no attribute {chosen!r}" if chosen else
             "no regions: the table has none of " + ", ".join(AUTO_IDS))
    raise ValueError(start + "; choose the id attribute among "
                     + (", ".join(have) if have else "(no integer attribute)"))


def summarised_columns(locs, chosen=()) -> Tuple[List[str], List[str]]:
    """``(names, missing)``: the ticked names (a list, or a comma- or
    space-separated text) the table has, each once, and those it lacks that
    are worth a warning -- not the default ones, which no table has all of."""
    if isinstance(chosen, str):
        chosen = chosen.replace(",", " ").split()
    names: List[str] = []
    for n in (str(n).strip() for n in chosen):
        if n and n not in names:
            names.append(n)
    missing = [n for n in names if n not in locs and n not in DEFAULT_ATTRIBUTES]
    return [n for n in names if n in locs], missing


#: what *attributes* does not offer: what says where a row came from
_NOT_OFFERED = ("filenumber", "channel", "group_id", "n_in_group")


def offered_columns(locs) -> List[str]:
    """The numeric attributes *attributes* offers to tick: ids left out
    but the frame, which is a time."""
    from ..tracking.columns import integer_columns
    ids = set(integer_columns(locs)) - set(TIMES)
    return [c for c in locs.keys() if c not in _NOT_OFFERED and c not in ids
            and np.asarray(locs[c]).ndim == 1 and np.asarray(locs[c]).dtype.kind in "iuf"]


@dataclass
class StatisticsSettings:
    mean: bool = param(True, label="mean", help="of each attribute's finite values")
    median: bool = param(True, label="median",
                         help="robust to a few outliers; the one statistic that sorts")
    std: bool = param(False, label="standard deviation", help="ddof 1; NaN for one value")
    min: bool = param(False, label="minimum", help="the smallest finite value")
    max: bool = param(True, label="maximum", help="the largest finite value")
    percentiles: str = param("", label="percentiles",
                             help="typed, e.g. 5 95: photons_p5, photons_p95")

    def chosen(self) -> List[str]:
        from ..id_statistics import STATISTICS
        return [s for s in STATISTICS if getattr(self, s)]


@dataclass
class RegionStatisticsSettings:
    regions: str = param("", label="regions", choices=_region_choices,
                         help="the ROIs of the ROI manager, or an id attribute; "
                              "auto: tid, else track_id, else cluster_id, else the ROIs")
    attributes: list = param(default_factory=lambda: list(DEFAULT_ATTRIBUTES),
                             label="attributes", kind="columns", empty_is_all=False,
                             popup=True, groups=GROUPS,
                             help="what is summarised; the centre is measured "
                                  "whatever is ticked")
    shape: bool = param(False, label="shape",
                        help="Cluster shape's numbers too: hull area, axes, angle, "
                             "extents; the hull a loop")
    statistics: StatisticsSettings = param(default_factory=StatisticsSettings,
                                           label="statistics of the attributes")


@register("Analysis/Measure/Region statistics")
class RegionStatistics(Plugin):
    """Per ROI, trace, track or cluster at once: its localizations, the centre and
    spread of its positions, its duration, and statistics of chosen columns,
    kept as a region table."""

    Settings = RegionStatisticsSettings
    changes_locs = False
    # a trace's localizations, whatever the layer shows
    grouping = "ungrouped"
    version = "4"

    def describe(self, ctx: Optional[Context], settings: RegionStatisticsSettings
                 ) -> FormState:
        state = FormState()
        if ctx is None or ctx.session is None or not len(ctx.locs):
            return state
        from ..tracking.columns import integer_columns
        locs = ctx.locs
        try:
            auto = regions_source(ctx, "")
        except ValueError:
            auto = ""
        auto_text = "ROIs" if auto == ROIS else auto
        state.choices["regions"] = (
            [("", f"auto ({auto_text})" if auto_text else "auto")]
            + ([(ROIS, "ROIs")] if _has_rois(ctx) else [])
            + [(c, c) for c in integer_columns(locs)])
        state.choices["attributes"] = offered_columns(locs)
        return state

    def run(self, ctx: Context, settings: RegionStatisticsSettings) -> Result:
        from ..id_statistics import id_statistics, member_statistics
        locs = ctx.locs
        source = regions_source(ctx, settings.regions)
        for c in ("x_nm", "y_nm"):
            if c not in locs:
                raise ValueError(f"no attribute {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        from ..id_statistics import parse_percentiles
        ticked, missing = summarised_columns(locs, settings.attributes)
        if missing:
            ctx.warn("not in this table, passed over: " + ", ".join(missing))
        names = [n for n in ticked if n not in POSITIONS and n not in TIMES]
        stats = settings.statistics.chosen()
        percentiles = parse_percentiles(settings.statistics.percentiles)
        if names and not stats and not percentiles:
            raise ValueError("tick at least one statistic of the attributes")
        keep = self._kept(ctx)
        positions = {a: locs[c] for c, a in POSITIONS.items() if c in locs}
        where = "ROIs" if source == ROIS else source
        ctx.report(f"region statistics: {where}")
        options = dict(
            keep=keep, time_s=locs["time_s"] if "time_s" in ticked else None,
            frame=locs["frame"] if "frame" in ticked else None,
            columns={n: locs[n] for n in names}, stats=stats, percentiles=percentiles,
            axes=[POSITIONS[c] for c in ticked if c in POSITIONS], shape=settings.shape)
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        if source == ROIS:
            project = ctx.rois
            ids = [i for i in project.ordered_ids() if project.rois[i].use]
            n = len(locs)
            members = []
            for roi_id in ids:
                rows = np.asarray(project.session_rows(project.rois[roi_id]), np.int64)
                members.append((roi_id, rows[rows < n]))
            found = member_statistics(members, positions, **options)
            if not len(found):
                raise ValueError("no regions: no ROI holds a localization the "
                                 "layer keeps")
            table = roi_table(project, found, step)
        else:
            found = id_statistics(locs[source], positions, **options)
            if not len(found):
                raise ValueError(f"no regions: no localization the layer keeps has "
                                 f"an id in {source} (-1 is none)")
            from ..region_runs import column_table
            table = column_table(locs, source, LABEL, found.ids, found.values, step,
                                 layer=ctx.layer, session=ctx.session,
                                 groups=found.groups)
        v = found.values
        lines = [f"{len(found)} regions of {where}, {int(v['n_locs'].sum())} "
                 f"localizations: a median of {np.median(v['n_locs']):.0f} each"]
        if "std_xy_nm" in v and np.isfinite(v["std_xy_nm"]).any():
            lines.append(f"spread (std_xy_nm): median {np.nanmedian(v['std_xy_nm']):.3g} nm")
        if "elongation" in v and np.isfinite(v["std_long_nm"]).any():
            lines.append(f"shape: median {np.nanmedian(v['std_long_nm']):.3g} x "
                         f"{np.nanmedian(v['std_short_nm']):.3g} nm (std_long_nm x "
                         f"std_short_nm), area {np.nanmedian(v['area_nm2']):.3g} nm²")
        if "duration_s" in v:
            lines.append(f"duration: median {np.nanmedian(v['duration_s']):.3g} s")
        kept_as = LABEL if source == ROIS else f"per {source}: {LABEL}"
        lines.append(f"kept as '{kept_as}': {len(v)} values per region")
        panels = [(k, v[k]) for k in ("n_locs", "std_xy_nm", "duration_s", "frames",
                                       "elongation") if k in v]
        for name in names:
            group = found.groups.get(name, {})
            first = group.get("median") or group.get("mean") or next(iter(group.values()), None)
            if first:
                panels.append((first, v[first]))

        def draw(figure):
            axes = np.atleast_1d(figure.subplots(1, len(panels)))
            for ax, (name, values) in zip(axes, panels):
                draw_histogram(ax, name, values)

        return Result(text="\n".join(lines),
                      plot=Plot(draw, panels=len(panels), size=(3.0 * len(panels), 2.8)),
                      id_regions={source: {LABEL: table}},
                      data={"statistics": found, "column": source},
                      settings=settings, title="region statistics")

    @staticmethod
    def _kept(ctx: Context) -> np.ndarray:
        """The layer's own filter, without the drawn ROI: what a block's
        regions are cut by, which the table's signature says."""
        from ..region_runs import layer_mask
        return layer_mask(ctx)


def draw_histogram(ax, name: str, values) -> None:
    """The distribution of one value over the regions."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        hi = np.quantile(values, 0.99) if values.size > 100 else values.max()
        lo = values.min()
        bins = np.arange(lo - 0.5, hi + 1.5) if name in ("n_locs", "frames") and \
            hi - lo < 100 else min(60, max(10, values.size // 20))
        ax.hist(values[values <= hi], bins=bins, color="0.35")
    ax.set_xlabel(name)
    ax.set_ylabel("regions")


def roi_table(project, found, step):
    """The ROIs' table, each row signed as `ROIProject.evaluations` checks
    it: the ROI's own inputs with the step, so moving one ROI makes its row
    alone out of date, as a per-region step's would be."""
    import json
    from ..region_table import RegionTable
    from ..roi_manager.core import digest, json_text
    table = RegionTable.from_columns(LABEL, [str(i) for i in found.ids], found.values,
                                     step, "", "", found.groups)
    identity = {k: v for k, v in step.items() if k != "label"}
    for k, roi_id in enumerate(table.ids):
        inputs = json.loads(json_text(project.inputs(project.rois[roi_id])))
        table.data[k] = digest(inputs)
        table.signatures[k] = digest({"inputs": inputs, "step": identity})
    return table
