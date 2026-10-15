"""Between regions and columns: what was measured per region written onto
its localizations, and an ROI made of each region an id column names.

`docs/regions.md` ("Region values as columns") has the decision: rather than
a second table with a filter of its own, a region's results go onto its rows
as ordinary columns -- `n_corners`, a cluster's precision -- and every tool
that reads a column (the layer's filter, colour by a field, the statistics,
a block repeating over an id) works on them, per layer, with nothing new to
learn.  The cost is that a histogram of such a column counts localizations,
not regions (a large cluster weighs more): the distribution over regions is
*Summarize/Histograms*.

It is a snapshot, as any column written by a plugin.  A block writes the same
columns itself at the end of its run (*write values*), so a chain keeps them
current; this plugin is for results already there.

The ROIs' own numbers (`roi_id`, what *ROI to column* used to write) are one
of the values: of the ROIs, with or without an evaluation.

**Column to ROIs** -- now *Segment regions*' method *from an id column*
(`segment.py`), with `column_regions` here -- is the other direction: an ROI of the ROI manager for every
id of a column (a cluster, whoever found it) with enough localizations, to
look at, review and annotate.  It used to be part of DBSCAN; on its own, any
clusterer -- a plugin dropped in, or ids read from a file -- gets it.  By
default each ROI is the cluster's own outline, its convex hull; without it,
only the cluster's centre is placed and the ROI is the ROI manager's circle or
square, so clusters meant to be measured alike are measured in the same
window, and the shape can still be switched there afterwards.  A cluster with
no hull (its localizations on one line) gets the centre too, rather than no
ROI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import Context, FormState, Plugin, Result, param, register
from ..region_runs import VALUE_RULES
from .roi import auto_only

# what *results from* is when it is empty: the ROIs, without an evaluation
NUMBERS = "the ROIs' numbers"


@dataclass
class RegionColumnSettings:
    source: str = param("", label="results from", choices=auto_only,
                        help="what was measured on the ROIs, or per region of "
                             "an id column; empty: the ROIs, for their numbers")
    values: str = param("", label="values",
                        help="comma separated; empty: every number the "
                             "measurement has; roi_id: the ROI's number")
    prefix: str = param("", label="prefix", advanced=True,
                        help="put before each column's name")
    merge: str = param("last", label="overlaps", choices=VALUE_RULES, advanced=True,
                       help="a localization in two ROIs takes the later ROI's "
                            "value, the first's, or their mean")


def rows_values(rows: List[Dict[str, Any]], names: List[str]) -> Dict[str, Dict[str, float]]:
    """``{value: {region id: number}}`` from a site table's rows (``roi_id``
    and the values); ``names`` empty takes every numeric one.  A name no row
    has is an error naming the ones there are."""
    numeric = sorted({k for r in rows for k, v in r.items()
                      if k not in ("roi_id", "file_id") and isinstance(v, (int, float))})
    wanted = names or numeric
    missing = [k for k in wanted if k not in numeric]
    if missing:
        raise ValueError(f"no region has a number called {', '.join(missing)}; "
                         f"there is {', '.join(numeric) or 'none'}")
    return {k: {str(r["roi_id"]): r[k] for r in rows
                if isinstance(r.get(k), (int, float))} for k in wanted}


@register("Analysis/Regions/Regions to column")
class RegionsToColumn(Plugin):
    """Writes what was measured per region onto the localizations in it."""

    Settings = RegionColumnSettings
    changes_locs = True
    # the values go onto the localizations
    grouping = "ungrouped"
    version = "2"         # 2: roi_id is the ROI's own number, kept when others go

    def describe(self, ctx, settings) -> FormState:
        """*results from*: the evaluations the session has."""
        if ctx is None:
            return FormState()
        from ..region_results import available
        return FormState(choices={"source": [("", NUMBERS)]
                                  + [(str(ev), str(ev)) for ev in available(ctx)]})

    def run(self, ctx: Context, settings: RegionColumnSettings) -> Result:
        from .. import region_results
        from ..chain import BLOCK_ROIS
        from ..region_runs import REGION_NUMBER, write_values
        names = [v.strip() for v in settings.values.split(",") if v.strip()]
        source = settings.source
        ev = region_results.Evaluation.parse(source)
        regions = ev.source
        if source and not region_results.is_current(ctx, ev):
            raise ValueError(region_results.missing(ctx, ev))
        if not ev.on_rois:
            numbers = False
            if REGION_NUMBER in names:
                raise ValueError(f"the regions of {regions} have no ROI numbers: "
                                 f"{regions} itself is theirs")
            values = rows_values(region_results.rows(ctx, ev), names)
        else:
            if ctx.rois is None:
                raise ValueError("there are no ROIs: find or draw some first")
            numbers = not source or REGION_NUMBER in names
            measured = [n for n in names if n != REGION_NUMBER]
            values = {}
            if source:
                if measured or not names:
                    values = rows_values(region_results.rows(ctx, ev), measured)
            elif measured:
                raise ValueError(f"{', '.join(measured)}: name the measurement that "
                                 "measured them in *results from*")
        out, members = write_values(ctx, ctx.locs, regions, values, settings.merge,
                                    {"source": source or NUMBERS}, settings.prefix,
                                    numbers=numbers)
        n = len(ctx.locs)
        columns = [settings.prefix + k for k in values] + (
            [settings.prefix + REGION_NUMBER] if numbers else [])
        inside = np.zeros(n, dtype=bool)
        for _, rows in members:
            inside[rows] = True
        where = "ROIs" if regions == BLOCK_ROIS else f"{regions} regions"
        return Result(locs=out, settings=settings,
                      text=f"{', '.join(columns)}: {len(members)} {where} over "
                           f"{int(inside.sum())} localizations, "
                           f"{n - int(inside.sum())} in none",
                      values={"regions": len(members), "inside": int(inside.sum())})


# ------------------------------------------------------------ column to ROIs

@dataclass
class ColumnROIsSettings:
    column: str = param("cluster_id", label="column",
                        help="an id column of the table; 0 and below is in no region")
    min_size: int = param(10, label="at least", unit="locs", min=1,
                          help="the localizations an id needs to become an ROI")
    outline: bool = param(True, label="the cluster's own outline",
                          help="each ROI the convex hull of its cluster; off: only "
                               "the centre, with the ROI manager's circle or square")


def hull(xy: np.ndarray) -> Optional[List[List[float]]]:
    """The convex hull of a cluster's positions, as polygon vertices; None
    when there are fewer than three points or they lie on a line."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return None
    try:
        h = ConvexHull(xy)
    except (QhullError, ValueError):
        return None
    return xy[h.vertices].tolist()


def column_regions(locs, column: str, min_size: int = 1, outline: bool = True
                   ) -> List[Dict[str, Any]]:
    """One entry per id of ``column`` with at least ``min_size`` rows:
    ``{"id", "rows", "center", "polygon", "file"}`` -- the centre the mean
    position, the polygon the convex hull (None when ``outline`` is off, or
    the points are fewer than three or on a line), the file that of its
    first row."""
    from ..region_runs import split_ids
    if column not in locs:
        raise ValueError(f"the table has no {column!r}: it has "
                         f"{', '.join(sorted(locs.keys()))}")
    values = np.asarray(locs[column])
    if values.dtype.kind not in "iub" and not (
            values.dtype.kind == "f" and np.all(np.mod(values[np.isfinite(values)], 1) == 0)):
        raise ValueError(f"{column} is not an id column: its values are not whole numbers")
    x = np.asarray(locs["x_nm"], dtype=float)
    y = np.asarray(locs["y_nm"], dtype=float)
    files = (np.asarray(locs["filenumber"]) if "filenumber" in locs
             else np.zeros(len(locs), dtype=np.int64))
    out = []
    ids, rows = split_ids(values)
    for k, members in zip(ids, rows):
        if len(members) < min_size:
            continue
        xy = np.column_stack((x[members], y[members]))
        out.append({"id": int(k), "rows": members,
                    "center": [float(xy[:, 0].mean()), float(xy[:, 1].mean())],
                    "polygon": hull(xy) if outline else None,
                    "file": int(files[members[0]])})
    return out
