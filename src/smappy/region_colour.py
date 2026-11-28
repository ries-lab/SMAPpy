"""What was measured per region, as a colour of its localizations.

The Render tab colours a layer by an attribute; this makes a region's value
one of them -- a cluster's circularity, a pore's number of corners --
without writing it onto the localizations.  A layer's ``color_field`` names
the value as the region filter does, ``ROIs › NPC Corners › n_corners`` or
``per cluster_id › Id statistics › circularity`` (`name`, `parse`), and the
column is made when the layer is drawn: `ViewState.snapshot` asks the
session (`Tables.column_source`, `Session.region_column`), and the render
threads, the 3D view, the colour bars and *Save image* see an ordinary column
of the snapshot.  The stored table never has it, so nothing is saved,
exported or handed to a plugin; *Write results* is the way to put the values
onto the localizations for good.

**Why not a column.**  Written onto the table (*Write results*) the values
are a copy that goes stale when the regions are measured again, and are two
steps away from where they were measured.  Made here they follow the region
table: the cache is keyed by what they are made from (`_data_key`, the
region filter's key) and made again when that moves.

**The rules.**

* A localization in **no region** has no value, NaN, which the LUT draws grey
  (`lut.NO_VALUE`); *hide no value* (`RenderSettings.hide_missing`) leaves
  them out instead.
* Where **ROIs overlap** a localization takes the **mean** of their values:
  it belongs to all of them, and the mean says so where "the last one"
  would pick one by the order they were drawn.
* An ROI switched off (*use*) gives no value: its results are not part of
  the analysis.
* On a **merged** layer a blink takes the mean over its localizations
  (`group_id`); an id column the grouped table has is looked up directly.
* Only numbers colour: a keyword is no place on a LUT, so `numeric_values`
  leaves it out; a flag (*use*) is 0 / 1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .chain import BLOCK_ROIS as ROIS

#: between the parts of a name, as the region filter's lists show them
SEP = " › "
_ROIS_WORD = "ROIs"
_PER = "per "


def name(source: str, label: str, value: str) -> str:
    """The colour field of one value: what the Render tab and the colour bar
    show, and what a layer keeps."""
    where = _ROIS_WORD if source == ROIS else f"{_PER}{source}"
    return f"{where}{SEP}{label}{SEP}{value}"


def parse(field: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """``(source, label, value)`` of a region colour field, None for any
    other: a column of the table never has the separator."""
    if not field or SEP not in field:
        return None
    parts = field.split(SEP)
    if len(parts) != 3:
        return None
    where, label, value = parts
    if where == _ROIS_WORD:
        return ROIS, label, value
    if where.startswith(_PER):
        return where[len(_PER):], label, value
    return None


def _is_number(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return True
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    return False


def numeric_values(session) -> Dict[str, Dict[str, List[str]]]:
    """``{source: {label: [value names]}}`` of every value some region has a
    number for: what a layer can be coloured by."""
    from .region_filter import region_values, sources
    out: Dict[str, Dict[str, List[str]]] = {}
    for source, labels in sources(session).items():
        for label in labels:
            _ids, values = region_values(session, source, label)
            found = [v for v in sorted(values)
                     if any(_is_number(x) for x in values[v] if x is not None)]
            if found:
                out.setdefault(source, {})[label] = found
    return out


def _floats(values) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        if _is_number(v):
            out[i] = float(v)
    return out


def _onto_ids(ids_column: np.ndarray, region_ids, values: np.ndarray) -> np.ndarray:
    """Each row's region's value, by its id; NaN for a row in none (-1, NaN,
    an id the table does not have)."""
    out = np.full(len(ids_column), np.nan, dtype=np.float32)
    keys, vals = [], []
    for r, v in zip(region_ids, values):
        try:
            keys.append(int(float(r)))
        except (TypeError, ValueError):
            continue
        vals.append(v)
    if not keys:
        return out
    keys = np.asarray(keys, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    ids = np.asarray(ids_column)
    good = np.isfinite(ids) if ids.dtype.kind == "f" else np.ones(len(ids), bool)
    as_int = np.where(good, ids, -1).astype(np.int64)
    at = np.clip(np.searchsorted(keys, as_int), 0, len(keys) - 1)
    hit = good & (keys[at] == as_int)
    out[hit] = vals[at[hit]]
    return out


def region_values_onto(session, source: str, label: str, value: str) -> np.ndarray:
    """One value per localization of the session's table: its region's, the
    mean of its ROIs' where they overlap, NaN where it is in none."""
    from .region_filter import region_values
    locs = session.locs
    ids, columns = region_values(session, source, label)
    values = _floats(columns.get(value, [None] * len(ids)))
    if source != ROIS:
        if source not in locs:
            return np.full(len(locs), np.nan, dtype=np.float32)
        return _onto_ids(locs[source], ids, values)
    project = session.rois
    sums = np.zeros(len(locs))
    counts = np.zeros(len(locs))
    for roi_id, v in zip(ids, values):
        roi = project.rois.get(roi_id)
        if (roi is None or not roi.use or not np.isfinite(v)
                or roi.file_id not in project.sources):
            continue
        rows = project.session_rows(roi)     # each row once: += is safe
        sums[rows] += v
        counts[rows] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan).astype(np.float32)


def _per_blink(session, table, values: np.ndarray) -> Optional[np.ndarray]:
    """The localizations' values as the grouped ``table``'s rows: each
    blink's mean over its localizations (`group_id`), NaN where none has one."""
    locs = session.locs
    if "group_id" not in locs or "group_id" not in table:
        return None
    gid = np.asarray(locs["group_id"], dtype=np.int64)
    rows = np.asarray(table["group_id"], dtype=np.int64)
    top = int(max(gid.max(initial=0), rows.max(initial=0)))
    ok = (gid >= 0) & np.isfinite(values)
    sums = np.bincount(gid[ok], weights=values[ok], minlength=top + 1)
    counts = np.bincount(gid[ok], minlength=top + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out = np.full(len(rows), np.nan, dtype=np.float32)
    inside = (rows >= 0) & (rows <= top)
    out[inside] = per[rows[inside]]
    return out


def _data_key(session, source: str, label: str) -> tuple:
    """What the values depend on, cheaply -- the region filter's key: the
    table, the region table and, for the ROIs, each one's place and use."""
    locs = session.locs
    parts = [id(locs), len(locs)]
    if source == ROIS:
        project = session.rois
        table = project.tables.get(label)
        parts.append((id(table), None if table is None else tuple(table.signatures),
                      tuple((r.id, r.use, tuple(r.center), id(r.polygon), id(r.ellipse),
                             r.file_id, repr(sorted(r.annotations.items())))
                            for r in project.rois.values())))
    else:
        table = (session.id_regions.get(source) or {}).get(label)
        parts.append((id(table), None if table is None else (len(table), tuple(table.signatures)),
                      id(locs[source]) if source in locs else None))
    return tuple(parts)


def onto_table(session, table, field: str, cache: dict) -> Optional[np.ndarray]:
    """``field``'s values as rows of ``table`` -- the session's table or its
    grouped form -- or None when ``field`` is no region value or cannot be
    put onto this table.  ``cache`` keeps the answers between draws."""
    found = parse(field)
    if found is None:
        return None
    source, label, value = found
    data = _data_key(session, source, label)
    key = (id(table), len(table), field)
    hit = cache.get(key)
    if hit is not None and hit[0] == data:
        return hit[1]
    if source != ROIS and source in table:
        from .region_filter import region_values
        ids, columns = region_values(session, source, label)
        values = _onto_ids(table[source], ids, _floats(columns.get(value, [None] * len(ids))))
    else:
        values = region_values_onto(session, source, label, value)
        if table is not session.locs:
            values = _per_blink(session, table, values)
    if values is None or len(values) != len(table):
        return None
    if len(cache) > 16:
        cache.clear()
    # the table kept with its answer, so that its id is not reused meanwhile
    cache[key] = (data, values, table)
    return values
