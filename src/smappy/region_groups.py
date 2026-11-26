"""Groups of regions: which group each row of a region table is in.

One grouping for every summary of a region table -- *Compare groups*,
*Histogram* on region results, *Curves* -- so that "by file", "by keyword"
and "by condition" mean the same thing wherever they are offered.  A group
is named by:

* ``file`` -- the file an ROI was drawn on, or, for a region of an id
  column (a trace, a cluster), the file its localizations are in;
* a column of the results with few distinct values -- an ROI keyword list
  (*Annotate ROIs*), or a number such as a laser power a step measured;
* ``attribute:<name>`` -- for an id column's regions, an attribute of their
  localizations with few values (a *File conditions* column, ``thi``),
  read from each region's first localization.

The rows are `region_results.rows`' (``roi_id``, ``file_id`` and the
values); nothing here imports Qt or a plugin, so a script groups the same
way.  Moved out of `plugins/compare_groups.py`, which re-exports it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: a column with at most this many distinct values can name groups
MAX_GROUPS = 12
#: what *group by* says for an attribute of the regions' localizations
ATTR = "attribute:"
#: the group of a row that has no value of the grouping column (an ROI
#: without a keyword in the list)
NO_GROUP = "(none)"


def numeric_fields(rows: Sequence[Dict[str, Any]], exclude=()) -> List[str]:
    """Every numeric column of the results but the ids, the counts behind
    means (``_n``, nearly always the localization count again) and
    ``exclude``."""
    if not rows:
        return []
    return [k for k in rows[0]
            if k not in ("roi_id", "file_id") and k not in exclude and not k.endswith("_n")
            and all(isinstance(r.get(k), (int, float, np.integer, np.floating))
                    and not isinstance(r.get(k), (bool, np.bool_)) for r in rows)]


def _keys(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Every column any row has, in order of first appearance: an ROI's
    keyword is a column of its row only when it has one."""
    out: Dict[str, None] = {}
    for r in rows:
        out.update(dict.fromkeys(r))
    return list(out)


def group_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The columns of the results that can name groups: few distinct values."""
    out = []
    for k in _keys(rows):
        if k in ("roi_id", "file_id"):
            continue
        values = {str(r.get(k)) for r in rows}
        if 1 < len(values) <= MAX_GROUPS:
            out.append(k)
    return out


def group_choices(ctx, rows, ev, extra: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """*group by*'s list: none, ``file``, the results' columns with few values
    (and ``extra``, the ROI keyword lists, which a column may not show while
    one keyword is all there is), then the localizations' attributes for an
    id column's regions."""
    columns = group_columns(rows) if rows else []
    columns += [k for k in extra if k not in columns]
    return [("", "-"), ("file", "file")] + [(k, k) for k in columns] + [
        (ATTR + c, f"{c} (of the localizations)") for c in attribute_columns(ctx, ev)]


def group_labels(ctx, rows, ev, group_by: str) -> List[str]:
    """Each row's group: its file's name (an ROI's ``file_id``; for a region
    of an id column, the file its localizations are in), the value of the
    column ``group_by`` (`NO_GROUP` where a row has none), or of an
    attribute of its localizations (``attribute:<name>``).  Empty
    ``group_by``: one group, ``all``."""
    if not group_by:
        return ["all"] * len(rows)
    if group_by.startswith(ATTR):
        return attribute_labels(ctx, rows, ev, group_by[len(ATTR):])
    if group_by != "file":
        if rows and not any(group_by in r for r in rows):
            raise ValueError(f"no column {group_by!r} to group by: the results have "
                             + ", ".join(sorted(_keys(rows))))
        return [NO_GROUP if r.get(group_by) in (None, "") else str(r.get(group_by))
                for r in rows]
    names = {}
    session = ctx.session if ctx is not None else None
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
    from .ids import is_id
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
    session = ctx.session if ctx is not None else None
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


def natural(g):
    """Sort key: numbers by value before words, so ``10 uW`` follows ``2 uW``."""
    try:
        return (0, float(g), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(g))


def ordered(groups: Sequence[str]) -> List[str]:
    """The distinct groups in natural order, `NO_GROUP` last."""
    names = sorted(set(groups), key=natural)
    if NO_GROUP in names:
        names.remove(NO_GROUP)
        names.append(NO_GROUP)
    return names


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
