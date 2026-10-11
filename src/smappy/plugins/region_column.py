"""What was measured per region, written onto the localizations in it.

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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from . import Context, Plugin, Result, param, register
from .roi import PER, auto_only, id_table, table_rows

# what *results from* is when it is empty: the ROIs, without an evaluation
NUMBERS = "the ROIs' numbers"
OVERLAPS = ("last", "first", "mean")


@dataclass
class RegionColumnSettings:
    source: str = param("", label="results from", choices=auto_only,
                        help="an evaluation on the ROIs, or one per region of "
                             "an id column; empty: the ROIs, for their numbers")
    values: str = param("", label="values",
                        help="comma separated; empty: every number the "
                             "evaluation measured; roi_id: the ROI's number")
    prefix: str = param("", label="prefix", advanced=True,
                        help="put before each column's name")
    merge: str = param("last", label="overlaps", choices=OVERLAPS, advanced=True,
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
    version = "1"

    def choices(self, ctx: Context, settings):
        out = [("", NUMBERS)]
        if ctx.rois is not None:
            out += [(name, name) for name in sorted(ctx.rois.labels())]
        if ctx.session is not None:
            for column, tables in sorted(ctx.session.id_regions.items()):
                out += [(f"{PER}{column}: {label}",) * 2 for label in tables]
        return {"source": out}

    def run(self, ctx: Context, settings: RegionColumnSettings) -> Result:
        from ..chain import BLOCK_ROIS
        from ..region_runs import (REGION_NUMBER, region_members, values_onto,
                                   with_region_columns)
        names = [v.strip() for v in settings.values.split(",") if v.strip()]
        source = settings.source
        found = id_table(ctx, source)
        if found is not None:
            regions, table = found
            numbers = False
            if REGION_NUMBER in names:
                raise ValueError(f"the regions of {regions} have no ROI numbers: "
                                 f"{regions} itself is theirs")
            values = rows_values(table_rows(table), names)
        elif source.startswith(PER):
            raise ValueError(f"there are no results {source!r} (any more)")
        else:
            regions = BLOCK_ROIS
            if ctx.rois is None:
                raise ValueError("there are no ROIs: find or draw some first")
            numbers = not source or REGION_NUMBER in names
            measured = [n for n in names if n != REGION_NUMBER]
            values = {}
            if source:
                if source not in ctx.rois.labels():
                    raise ValueError(f"no evaluation on the ROIs is called {source!r}")
                if measured or not names:
                    values = rows_values(ctx.rois.results(labels=[source]), measured)
            elif measured:
                raise ValueError(f"{', '.join(measured)}: name the evaluation that "
                                 "measured them in *results from*")
        members = region_members(ctx, regions)
        n = len(ctx.locs)
        columns = {settings.prefix + k: v for k, v in
                   values_onto(n, members, values, settings.merge).items()}
        if regions == BLOCK_ROIS and numbers:
            columns[settings.prefix + REGION_NUMBER] = roi_numbers(
                n, members, ctx.rois.numbers(),
                "first" if settings.merge == "first" else "last")
        if not columns:
            raise ValueError("there is nothing to write")
        for name in columns:
            if not name.isidentifier():
                raise ValueError(f"{name!r} is not usable as a column name")
        out = with_region_columns(ctx.locs, columns,
                                  {"regions": regions, "source": source or NUMBERS})
        inside = np.zeros(n, dtype=bool)
        for _, rows in members:
            inside[rows] = True
        where = "ROIs" if regions == BLOCK_ROIS else f"{regions} regions"
        return Result(locs=out, settings=settings,
                      text=f"{', '.join(columns)}: {len(members)} {where} over "
                           f"{int(inside.sum())} localizations, "
                           f"{n - int(inside.sum())} in none",
                      values={"regions": len(members), "inside": int(inside.sum())})


def roi_numbers(n: int, members, numbers: Dict[Any, int], rule: str = "last") -> np.ndarray:
    """``n`` rows, each the number of the last ROI in ``members`` -- in the
    order they were made -- that holds it (the first, with ``rule``), 0 in
    none."""
    from ..region_runs import merge_pieces
    numbers = {str(k): v for k, v in numbers.items()}
    pieces = [(rows, np.full(len(rows), numbers[r], dtype=np.int64))
              for r, rows in members if len(rows)]
    return merge_pieces(n, pieces, rule) if pieces else np.zeros(n, dtype=np.int64)
