"""Id statistics: the localizations of every trace, track or cluster at once.

The algorithm is `smappy.id_statistics` (its docstring has the decisions);
this is the wrapper.  A region of an id column is a MINFLUX trace (`tid`), a
linked track, a cluster: everything a block over that column would measure
one region at a time with *Localization statistics*, counted here in one
pass, so 10^5 traces take a second instead of minutes.

What it hands back is a region table, not columns: the table of every
region (`region_runs.column_table`), signed as a block's run would sign it
and stored with the column (`Session.id_regions`) under the plugin's name --
"per tid: Id statistics" -- where *Summarize/Histograms*, *results from* and
the region filter (`docs/region-filter.md`) find it; *Regions to column*
writes a value onto the localizations where one is wanted there.

The localizations are the layer's own (its filter, not the drawn ROI, which
a block's regions ignore too), ungrouped: a trace's statistics are of its
localizations, and a merged blink of a trace would be the trace itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values

#: what *auto* takes the regions from, in order: MINFLUX's traces, linked
#: tracks, clusters, a simulation's tracks
AUTO_IDS = ("tid", "track_id", "cluster_id", "track_id_true")
#: what *auto* summarises, where the table has it: MINFLUX's quality and
#: colour, and the camera fit's
AUTO_COLUMNS = ("cfr", "efo", "dcr", "photons", "xy_err_nm")
#: the evaluation's name, what *results from* shows after "per <column>: "
LABEL = "Id statistics"


def _id_choices():
    return [("", "auto")] + [(c, c) for c in AUTO_IDS]


def id_column(locs, chosen: str = "") -> str:
    """The column naming the regions: ``chosen``, or on *auto* the first of
    `AUTO_IDS` the table has.  ValueError naming the integer columns there
    are when there is none."""
    from ..tracking.columns import integer_columns
    column = (chosen or "").strip() or next((c for c in AUTO_IDS if c in locs), "")
    if column and column in locs:
        return column
    have = integer_columns(locs)
    start = (f"no column {chosen!r}" if chosen else
             "no regions: the table has none of " + ", ".join(AUTO_IDS))
    raise ValueError(start + "; choose the id column among "
                     + (", ".join(have) if have else "(no integer column)"))


def summarised_columns(locs, text: str = "") -> List[str]:
    """The columns ``text`` names (comma- or space-separated), or on *auto*
    those of `AUTO_COLUMNS` the table has.  A name the table lacks is refused,
    naming what it has."""
    names = [n for n in text.replace(",", " ").split() if n]
    if not names:
        return [c for c in AUTO_COLUMNS if c in locs]
    missing = [n for n in names if n not in locs]
    if missing:
        raise ValueError(f"no column {', '.join(missing)}: the table has "
                         + ", ".join(sorted(locs.keys())))
    return names


@dataclass
class StatisticsSettings:
    mean: bool = param(True, label="mean", help="of each column's finite values")
    median: bool = param(True, label="median",
                         help="robust to a few outliers; the one statistic that sorts")
    std: bool = param(False, label="standard deviation", help="ddof 1; NaN for one value")
    min: bool = param(False, label="minimum", help="the smallest finite value")
    max: bool = param(True, label="maximum", help="the largest finite value")

    def chosen(self) -> List[str]:
        from ..id_statistics import STATISTICS
        return [s for s in STATISTICS if getattr(self, s)]


@dataclass
class IdStatisticsSettings:
    id_column: str = param("", label="id column", choices=_id_choices,
                           help="the regions; auto: tid, else track_id, else "
                                "cluster_id")
    columns: str = param("", label="columns",
                         help="columns to summarise, separated by commas; empty: "
                              "cfr, efo, dcr, photons, xy_err_nm where present")
    use_z: bool = param(True, label="use z", help="z_nm's centre and spread too")
    statistics: StatisticsSettings = param(default_factory=StatisticsSettings,
                                           label="statistics of the columns")


@register("Analysis/Measure/Id statistics")
class IdStatistics(Plugin):
    """Per trace, track or cluster at once: its localizations, the centre and
    spread of its positions, its duration, and statistics of chosen columns,
    kept as a region table."""

    Settings = IdStatisticsSettings
    changes_locs = False
    # a trace's localizations, whatever the layer shows
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: IdStatisticsSettings
                 ) -> FormState:
        state = FormState()
        if ctx is None or ctx.session is None or not len(ctx.locs):
            return state
        from ..tracking.columns import integer_columns
        locs = ctx.locs
        found = next((c for c in AUTO_IDS if c in locs), "")
        state.choices["id_column"] = ([("", f"auto ({found})" if found else "auto")]
                                      + [(c, c) for c in integer_columns(locs)])
        state.active["use_z"] = "z_nm" in locs
        if not settings.columns.strip():
            state.hints["columns"] = ", ".join(summarised_columns(locs)) or "none"
        return state

    def run(self, ctx: Context, settings: IdStatisticsSettings) -> Result:
        from ..id_statistics import id_statistics
        from ..region_runs import column_table
        locs = ctx.locs
        column = id_column(locs, settings.id_column)
        for c in ("x_nm", "y_nm"):
            if c not in locs:
                raise ValueError(f"no column {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        names = summarised_columns(locs, settings.columns)
        stats = settings.statistics.chosen()
        if names and not stats:
            raise ValueError("tick at least one statistic of the columns")
        keep = self._kept(ctx)
        positions = {"x": locs["x_nm"], "y": locs["y_nm"]}
        if settings.use_z and "z_nm" in locs:
            positions["z"] = locs["z_nm"]
        ctx.report(f"id statistics: {column}")
        found = id_statistics(
            locs[column], positions, keep=keep,
            time_s=locs["time_s"] if "time_s" in locs else None,
            frame=locs["frame"] if "time_s" not in locs and "frame" in locs else None,
            columns={n: locs[n] for n in names}, stats=stats)
        if not len(found):
            raise ValueError(f"no regions: no localization the layer keeps has "
                             f"{column} above 0")
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(locs, column, LABEL, found.ids, found.values, step,
                             layer=ctx.layer, session=ctx.session, groups=found.groups)
        v = found.values
        lines = [f"{len(found)} regions of {column}, {int(v['n_locs'].sum())} "
                 f"localizations: a median of {np.median(v['n_locs']):.0f} each"]
        if "std_xy_nm" in v and np.isfinite(v["std_xy_nm"]).any():
            lines.append(f"spread (std_xy_nm): median {np.nanmedian(v['std_xy_nm']):.3g} nm")
        if "duration_s" in v:
            lines.append(f"duration: median {np.nanmedian(v['duration_s']):.3g} s")
        lines.append(f"kept as 'per {column}: {LABEL}': {len(v)} values per region")
        panels = [(k, v[k]) for k in ("n_locs", "std_xy_nm", "duration_s", "frames")
                  if k in v]
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
                      id_regions={column: {LABEL: table}},
                      data={"statistics": found, "column": column},
                      settings=settings, title="id statistics")

    @staticmethod
    def _kept(ctx: Context) -> np.ndarray:
        """The layer's own filter over ``ctx.locs``, without the drawn ROI:
        what a block's regions are cut by, which the table's signature says."""
        session = ctx.session
        if session is not None:
            i = ctx.layer
            if session.layers[i].is_image:
                i = session.first_locs_layer()
            mask = np.asarray(session.layers[i].selection(i).mask, dtype=bool)
            if len(mask) == len(ctx.locs):
                return mask
        return np.asarray(ctx.selection.mask, dtype=bool)


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
