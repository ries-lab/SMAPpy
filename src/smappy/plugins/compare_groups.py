"""*Compare groups*: what was measured per region, compared between groups
of regions -- files, conditions, ROI keywords -- as a table of medians and
a grid of plots, one panel per measure and a box (or violin) per group.

The summaries of a region table (*Histograms of ROI results*, *Curves*)
pool every region; the MFX scripts' plot grids split traces by a condition
-- a laser power, a sample, an acquisition -- and draw the same measure for
each side by side.  This is that, for any region table: of ROIs or of an
id column such as ``tid``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import Context, FormState, Plot, Result, param, register
from .roi import SiteAnalysisPlugin, auto_only, grid_shape, measure_title

#: a column with at most this many distinct values can name groups
MAX_GROUPS = 12
#: what *group by* says for an attribute of the regions' localizations
ATTR = "attribute:"


@dataclass
class CompareGroupsSettings:
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the measurement compared, on the ROIs or per region of "
                                 "an id attribute")
    group_by: str = param("file", label="group by",
                          help="file; a column of the results with few values (an ROI "
                               "keyword); or, for an id attribute's regions, an attribute "
                               "of their localizations with few values (a file condition, "
                               "thi)")
    fields: str = param("", label="measures",
                        help="comma separated; empty: every numeric column")
    style: str = param("box", label="draw as", choices=(("box", "boxes"),
                                                        ("violin", "violins")),
                       help="the distribution per group")


@register("Analysis/Regions/Summarize/Compare groups")
class CompareGroups(SiteAnalysisPlugin):
    """The measures of a region table compared between groups of regions --
    files, conditions, keywords: a table of medians and a box per group."""

    Settings = CompareGroupsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        state = super().describe(ctx, settings)
        try:
            rows, ev = self.rows(ctx, settings) if ctx is not None else ([], None)
        except Exception:                                  # noqa: BLE001
            return state
        if rows:
            state.choices["group_by"] = [("file", "file")] + [
                (k, k) for k in group_columns(rows)] + [
                (ATTR + c, f"{c} (of the localizations)")
                for c in attribute_columns(ctx, ev)]
        return state

    def run(self, ctx: Context, settings: CompareGroupsSettings) -> Result:
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("nothing measured in the regions yet: run a per-region "
                             "measurement first")
        groups = group_labels(ctx, rows, ev, settings.group_by)
        fields = [f.strip() for f in settings.fields.split(",") if f.strip()] or \
            numeric_fields(rows, exclude=(settings.group_by,))
        missing = [f for f in fields if f not in rows[0]]
        if missing:
            raise ValueError(f"no measure {', '.join(missing)}: the results have "
                             + ", ".join(sorted(rows[0])))
        names = sorted(set(groups), key=_natural)
        table = compare(rows, groups, fields, names)
        lines = [f"{len(rows)} regions in {len(names)} groups by {settings.group_by}: "
                 + ", ".join(f"{g} ({table['n'][g]})" for g in names)]
        width = max(len(f) for f in fields)
        lines.append(" " * width + "  " + "  ".join(f"{g:>10.10}" for g in names))
        for f in fields:
            lines.append(f"{f:<{width}}  " + "  ".join(
                f"{table['median'][f][g]:>10.4g}" for g in names))
        n_rows, across = grid_shape(len(fields))

        def draw(figure):
            axes = figure.subplots(n_rows, across, squeeze=False).ravel()
            for ax, f in zip(axes, fields):
                data = [table["values"][f][g] for g in names]
                ok = [len(d) > 0 for d in data]
                pos = [i for i, k in enumerate(ok) if k]
                if not pos:
                    ax.set_visible(False)
                    continue
                vals = [data[i] for i in pos]
                if settings.style == "violin" and all(len(v) > 1 for v in vals):
                    ax.violinplot(vals, positions=pos, showmedians=True)
                else:
                    ax.boxplot(vals, positions=pos, widths=0.6, showfliers=False)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([str(g) for g in names], rotation=30, fontsize=7)
                ax.set_title(measure_title(f), fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axes[len(fields):]:
                ax.set_visible(False)

        return Result(text="\n".join(lines),
                      plot=Plot(draw, panels=len(fields), size=(3.2 * across, 2.8 * n_rows)),
                      data={"table": table, "groups": names}, settings=settings,
                      title="compare groups")


def _natural(g):
    return (0, float(g)) if _is_number(g) else (1, str(g))


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def numeric_fields(rows: Sequence[Dict[str, Any]], exclude=()) -> List[str]:
    """Every numeric column of the results but the ids, the counts behind
    means and ``exclude``."""
    return [k for k in rows[0]
            if k not in ("roi_id", "file_id") and k not in exclude and not k.endswith("_n")
            and all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                    for r in rows)]


def group_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The columns of the results that can name groups: few distinct values."""
    out = []
    for k in rows[0]:
        if k in ("roi_id", "file_id"):
            continue
        values = {str(r.get(k)) for r in rows}
        if 1 < len(values) <= MAX_GROUPS:
            out.append(k)
    return out


def group_labels(ctx: Context, rows, ev, group_by: str) -> List[str]:
    """Each row's group: its file's name (an ROI's ``file_id``; for a region
    of an id column, the file its localizations are in), or the value of
    the column ``group_by``."""
    if group_by.startswith(ATTR):
        return attribute_labels(ctx, rows, ev, group_by[len(ATTR):])
    if group_by != "file":
        if group_by not in rows[0]:
            raise ValueError(f"no column {group_by!r} to group by: the results have "
                             + ", ".join(sorted(rows[0])))
        return [str(r.get(group_by)) for r in rows]
    names = {}
    session = ctx.session
    if session is not None:
        names = {str(i): f.name for i, f in enumerate(session.files)}
    if ev is None or ev.on_rois:
        out = []
        for r in rows:
            fid = str(r.get("file_id", ""))
            out.append(fid if not names else _file_of_roi(ctx, r, names) or fid)
        return out
    if session is None or ev.source not in session.locs or "filenumber" not in session.locs:
        return ["all"] * len(rows)
    ids = np.asarray(session.locs[ev.source])
    files = np.asarray(session.locs["filenumber"])
    order = np.argsort(ids, kind="stable")
    found = np.asarray([float(r["roi_id"]) for r in rows])
    at = np.clip(np.searchsorted(ids[order], found), 0, len(ids) - 1)
    hit = ids[order][at] == found
    return [names.get(str(int(files[order][a])), str(int(files[order][a]))) if h else "none"
            for a, h in zip(at, hit)]


def attribute_columns(ctx, ev) -> List[str]:
    """The localizations' attributes with few distinct values that can name
    groups of an id column's regions (a file condition, ``thi``)."""
    session = ctx.session if ctx is not None else None
    if session is None or ev is None or ev.on_rois:
        return []
    from ..ids import is_id
    locs = session.locs
    out = []
    for c in locs.keys():
        v = np.asarray(locs[c])
        if v.ndim != 1 or is_id(c) or c in ("frame", "filenumber") or not len(v):
            continue
        sample = v[:: max(1, len(v) // 20000)]
        u = np.unique(sample[np.isfinite(sample)] if v.dtype.kind == "f" else sample)
        if 1 < len(u) <= MAX_GROUPS:
            out.append(c)
    return out


def attribute_labels(ctx, rows, ev, column: str) -> List[str]:
    """Each region's value of ``column`` -- its first localization's --
    named by the table's words where it is a coded condition
    (*File conditions*)."""
    session = ctx.session
    if session is None or ev is None or ev.on_rois or column not in session.locs:
        raise ValueError(f"grouping by {column} needs the regions of an id attribute "
                         "and the localizations")
    ids = np.asarray(session.locs[ev.source])
    values = np.asarray(session.locs[column])
    words = (session.locs.metadata.get("conditions") or {}).get(column)
    order = np.argsort(ids, kind="stable")
    found = np.asarray([float(r["roi_id"]) for r in rows])
    at = np.clip(np.searchsorted(ids[order], found), 0, len(ids) - 1)
    hit = ids[order][at] == found
    out = []
    for a, h in zip(at, hit):
        if not h:
            out.append("none")
            continue
        v = values[order][a]
        if words is not None and 0 <= int(v) < len(words):
            out.append(str(words[int(v)]))
        else:
            out.append(f"{v:g}" if isinstance(v, (float, np.floating)) else str(v))
    return out


def _file_of_roi(ctx, row, names) -> Optional[str]:
    project = ctx.rois
    if project is None:
        return None
    try:
        roi = project.roi_numbered(row.get("roi_id"))
        src = project.sources.get(roi.file_id) if roi is not None else None
        return getattr(src, "name", None) or (roi.file_id if roi is not None else None)
    except Exception:                                      # noqa: BLE001
        return None


def compare(rows, groups, fields, names) -> Dict[str, Any]:
    """Per measure and group: the values, their median, mean and count."""
    g = np.asarray(groups, dtype=object)
    out = {"values": {}, "median": {}, "mean": {}, "n": {}}
    for name in names:
        out["n"][name] = int((g == name).sum())
    for f in fields:
        v = np.asarray([r.get(f, np.nan) for r in rows], dtype=float)
        out["values"][f], out["median"][f], out["mean"][f] = {}, {}, {}
        for name in names:
            x = v[(g == name) & np.isfinite(v)]
            out["values"][f][name] = x
            out["median"][f][name] = float(np.median(x)) if len(x) else np.nan
            out["mean"][f][name] = float(x.mean()) if len(x) else np.nan
    return out
