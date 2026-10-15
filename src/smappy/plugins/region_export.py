"""What was measured per region, written as a table for a spreadsheet.

SMAP's ``ExportEvaluationsTable``, for every evaluation at once: the results
table -- a row per ROI (or per region of an id column), a column per number
the steps measured -- as a csv, for the statistics and figures that are
made elsewhere.  What it reads is what *Summarize/Histograms* reads
(`region_results`, through `SiteAnalysisPlugin.rows`): on *auto* every
evaluation on the ROIs side by side, or one named evaluation, on the ROIs
or per cluster.

**The columns are chosen.**  A table of twenty measures and three
annotation lists is more than a figure wants; the list is ticked, and all
ticked writes every column there is when it runs (`[]`), also one a later
step adds.

**Where an ROI is.**  For the ROIs, *ROI position* adds its centre
(`x_nm`, `y_nm`), its file's name and its comment, which the results table
does not carry: what is needed to go back to an ROI from a row of the
spreadsheet, or to put the sites on a map.  A region of an id column has no
place of its own (its centre is a measurement, Localization Statistics'
mean position) and gets nothing added.

Rows come in the ROIs' order -- as *Sort ROIs* left it -- and only the
included ROIs (*use* ticked) whose results still describe them, as
everywhere a result is read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import Context, FormState, Result, param, register
from .roi import SiteAnalysisPlugin, auto_only

#: what *ROI position* adds, after `roi_id` and `file_id`
POSITION_COLUMNS = ("x_nm", "y_nm", "file", "comment")


def table_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Every column of the rows, in the order they first appear."""
    return list(dict.fromkeys(key for row in rows for key in row))


def with_positions(rows: Sequence[Dict[str, Any]], project) -> List[Dict[str, Any]]:
    """Each ROI's row with its centre, its file's name and its comment."""
    out = []
    for row in rows:
        roi = project.roi_numbered(row.get("roi_id"))
        if roi is None:
            out.append(dict(row))
            continue
        source = project.sources.get(roi.file_id)
        extra = {"x_nm": float(roi.center[0]), "y_nm": float(roi.center[1]),
                 "file": source.name if source is not None else "",
                 "comment": roi.comment}
        first = {k: row[k] for k in ("roi_id", "file_id") if k in row}
        out.append({**first, **extra,
                    **{k: v for k, v in row.items() if k not in first}})
    return out


def write_table(rows: Sequence[Dict[str, Any]], path: Path,
                columns: Optional[Sequence[str]] = None) -> Path:
    """The rows as delimited text: a comma for .csv, a tab for .tsv and .txt.

    A value a row does not have is an empty cell; a number is written as
    Python writes it (17 significant digits at most, exact), an array as
    its values separated by spaces in one cell.
    """
    columns = list(columns) if columns else table_columns(rows)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(value):
        if value is None:
            return ""
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
        if isinstance(value, float) and value != value:          # NaN
            return ""
        return value

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in columns])
    return path


@dataclass
class ExportTableSettings:
    path: str = param("", label="file", kind="save_file",
                      file_filter="Text (*.csv *.tsv *.txt);;All files (*)",
                      help=".csv: commas; .tsv or .txt: tabs")
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the measurement whose table is written, on the "
                                 "ROIs or per region of an id column; auto: "
                                 "everything measured on the ROIs")
    positions: bool = param(True, label="ROI position",
                            help="add each ROI's centre, its file and its comment")
    columns: list = param(default_factory=list, label="columns", kind="columns",
                          help="the columns to write; all ticked writes every "
                               "column there is when it runs")


@register("Analysis/Regions/Summarize/Export table")
class ExportTable(SiteAnalysisPlugin):
    """Write the results table -- a row per ROI or region, a column per
    number measured -- as a csv."""

    Settings = ExportTableSettings
    changes_locs = False
    logged = True          # where a file went is not visible in the file
    version = "1"
    auto_label = "all on the ROIs"

    def evaluation(self, ctx: Context, settings):
        # every evaluation on the ROIs unless one is named, as Histograms
        from .roi import choose_evaluation
        if settings.evaluation:
            return choose_evaluation(ctx, "", settings.evaluation)
        return None

    def _rows(self, ctx: Context, settings) -> List[Dict[str, Any]]:
        rows, ev = self.rows(ctx, settings)
        on_rois = ev is None or ev.on_rois
        if settings.positions and on_rois and ctx.rois is not None:
            rows = with_positions(rows, ctx.rois)
        return rows

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        """*results from* as every summary has it, and the columns of the
        table it would write to tick."""
        state = super().describe(ctx, settings)
        if ctx is not None:
            try:
                state.choices["columns"] = table_columns(self._rows(ctx, settings))
            except ValueError:
                pass
        return state

    def run(self, ctx: Context, settings: ExportTableSettings) -> Result:
        if not settings.path:
            raise ValueError("choose where to write")
        rows = self._rows(ctx, settings)
        if not rows:
            raise ValueError("nothing measured in the regions yet: run the "
                             "per-region steps first")
        there = table_columns(rows)
        columns = [c for c in settings.columns if c in there] or there
        if settings.columns and not [c for c in settings.columns if c in there]:
            raise ValueError("none of the chosen columns is in the table; it has "
                             + ", ".join(there))
        written = write_table(rows, Path(settings.path), columns)
        return Result(text=f"{len(rows)} rows of {len(columns)} columns to {written}",
                      written=str(written), settings=settings)
