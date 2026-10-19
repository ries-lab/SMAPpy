"""The region filter: keeping the localizations of the regions whose results pass.

`docs/region-filter.md` has the design.  What was measured per region -- an
ROI's corners, a trace's spread, a cluster's size -- stays in its region
table (`region_results`); the filter is a list of **conditions** on those
values, and its answer reaches the localizations as one column,
`region_filter` (1 in, 0 out), which a bound on every layer honours like any
other: views, plugins, exports, batches.  Nothing else is written onto the
localizations.

**How conditions combine** (the note's examples):

* conditions on one source (the ROIs, or the regions of one id column) are
  judged **per region**: a region passes if it passes all of them -- of two
  overlapping pores, one with the corners and the other with the diameter,
  neither passes;
* within a source a localization is **in if it is in any passing region**
  (OR): a localization in the overlap of a good and a bad pore stays;
* different sources combine by **AND**;
* a localization in **no region** of a filtered source is out (DBSCAN's
  noise once clusters are filtered);
* a missing value (not measured, NaN) fails its condition, as an unmeasured
  ROI fails a comparison in *Filter ROIs*.

**What is stored.**  The conditions travel with the table, in its metadata
(`KEY`), as the derived columns' recipes do: a table handed back by a plugin
keeps them, an undo brings back the table's own, and a saved file opens
filtered.  The flag is recomputed from them, not kept as a snapshot, whenever
the table, the ROIs or the results change (`Session.refresh_region_filter`).

**Out of date.**  After a table change that keeps the ids (a drift
correction) the region tables are out of date as a whole; the filter keeps
applying them, with a warning, as long as the id column is the one they were
applied to (its hash, remembered per source).  A changed id column -- a new
clustering -- leaves nothing to apply: that source's conditions are
suspended, and the warning says so.  An ROI's values are its newest stored
ones, current or not, as *Filter ROIs* reads them.

A region's measurement never sees the flag (`region_runs.unmeasured`): a
filter on a result, applied while the result is measured again, would
rewrite it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chain import BLOCK_ROIS as ROIS
from .region_runs import REGION_FILTER, _hash

#: where the conditions are kept in a table's metadata
KEY = "region_filter"
#: the evaluation that is an ROI's annotations: *use*, keywords, lines
ANNOTATIONS = "annotations"
#: in, out
IN, OUT = 1.0, 0.0


@dataclass
class Condition:
    """One value of one evaluation of one source, and what it must be: a
    range (``lo`` / ``hi``, None unbounded) for a number, or the accepted
    texts (``accept``) for a keyword or a flag."""
    source: str                       # ROIS, or the id column
    label: str                        # the evaluation, or ANNOTATIONS
    value: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    accept: Optional[List[str]] = None

    def path(self) -> str:
        """What the list shows: ``ROIs › NPC Corners › n_corners``."""
        where = "ROIs" if self.source == ROIS else f"per {self.source}"
        return f"{where} › {self.label} › {self.value}"

    def bounds_text(self) -> str:
        if self.accept is not None:
            return " or ".join(self.accept) if self.accept else "(nothing)"
        lo = "-∞" if self.lo is None else f"{self.lo:g}"
        hi = "∞" if self.hi is None else f"{self.hi:g}"
        return f"{lo} … {hi}"

    def text(self) -> str:
        return f"{self.path()}  {self.bounds_text()}"

    def passes(self, value) -> bool:
        """Whether ``value`` meets the condition; a missing one never does."""
        if value is None:
            return False
        if self.accept is not None:
            return _as_text(value) in self.accept
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(v):
            return False
        return (self.lo is None or v >= self.lo) and (self.hi is None or v <= self.hi)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Condition":
        accept = doc.get("accept")
        return cls(str(doc["source"]), str(doc["label"]), str(doc["value"]),
                   None if doc.get("lo") is None else float(doc["lo"]),
                   None if doc.get("hi") is None else float(doc["hi"]),
                   None if accept is None else [str(a) for a in accept])


def _as_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ------------------------------------------------------- the stored state

def conditions(locs) -> List[Condition]:
    """The conditions the table carries, in the order they were added."""
    doc = (locs.metadata.get(KEY) or {}) if locs is not None else {}
    return [Condition.from_dict(c) for c in doc.get("conditions") or []]


def _state(locs) -> Dict[str, Any]:
    doc = dict(locs.metadata.get(KEY) or {})
    doc.setdefault("conditions", [])
    doc.setdefault("ids", {})
    return doc


def store(locs, found: Sequence[Condition]) -> None:
    """Keep ``found`` as the table's conditions; the remembered id hashes of
    sources no condition names any more go with them."""
    doc = _state(locs)
    doc["conditions"] = [c.to_dict() for c in found]
    sources = {c.source for c in found}
    doc["ids"] = {s: h for s, h in doc["ids"].items() if s in sources}
    if found:
        locs.metadata[KEY] = doc
    else:
        locs.metadata.pop(KEY, None)


# --------------------------------------------------- what can be filtered

def region_values(session, source: str, label: str) -> Tuple[List[str], Dict[str, list]]:
    """``(region ids, {value: a value per region})`` of one evaluation, as
    the filter reads it.  An id column's regions are its table's rows; the
    ROIs are every ROI, by `ROI.id`, with the newest value its table has
    (None where it has none), or with its annotations (*use*, keywords,
    lines)."""
    if source != ROIS:
        table = (session.id_regions.get(source) or {}).get(label)
        if table is None:
            return [], {}
        return list(table.ids), {k: list(v) for k, v in table.values.items()}
    project = session.rois
    ids = list(project.rois)
    if label == ANNOTATIONS:
        from .roi_manager.core import annotation_columns
        rows = [{"use": bool(project.rois[i].use),
                 **annotation_columns(project.rois[i], {})} for i in ids]
    else:
        table = project.tables.get(label)
        if table is None:
            return [], {}
        rows = []
        for i in ids:
            entry = table.entry(i) if i in table else None
            rows.append(dict(entry.get("values") or {}) if entry else {})
    names: List[str] = []
    for row in rows:
        names += [n for n in row if n not in names]
    return ids, {n: [row.get(n) for row in rows] for n in names}


def sources(session) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Everything that can be filtered: ``{source: {label: {"values":
    [names], "groups": {group: {member: name}}}}}`` -- the tree the panel
    shows, as deep as the groups make it."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    project = session.rois
    if project.rois:
        found = {ANNOTATIONS: {"values": sorted(region_values(session, ROIS, ANNOTATIONS)[1]),
                               "groups": {}}}
        for label, table in project.tables.items():
            found[label] = {"values": sorted(table.values), "groups": dict(table.groups)}
        out[ROIS] = found
    for column, tables in sorted(session.id_regions.items()):
        found = {label: {"values": sorted(t.values), "groups": dict(t.groups)}
                 for label, t in sorted(tables.items()) if len(t)}
        if found:
            out[column] = found
    return out


# ------------------------------------------------------------ the answer

@dataclass
class Outcome:
    """What the conditions give: the flag over the session's table (None
    when nothing is filtered), the regions each source keeps, how many each
    condition keeps of how many, and what to warn of."""
    flag: Optional[np.ndarray] = None
    passing: Dict[str, set] = field(default_factory=dict)
    counts: List[Tuple[int, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suspended: set = field(default_factory=set)


def evaluate(session, found: Optional[Sequence[Condition]] = None,
             members: bool = True) -> Outcome:
    """The flag and the regions that pass (see the module docstring), without
    writing anything; ``found`` defaults to the table's conditions.  Without
    ``members`` only the regions and the counts, not the flag: what a form
    shows as it is typed, without a spatial query per ROI."""
    from .region_results import Evaluation, is_current
    locs = session.locs
    found = conditions(locs) if found is None else list(found)
    out = Outcome()
    if not found:
        return out
    remembered = _state(locs)["ids"]
    ctx = session.context()
    by_source: Dict[str, List[int]] = {}
    for k, c in enumerate(found):
        by_source.setdefault(c.source, []).append(k)
    out.counts = [(0, 0)] * len(found)
    flag = np.ones(len(locs), dtype=bool)
    for source, ks in by_source.items():
        if source != ROIS:
            if source not in locs:
                out.warnings.append(f"the table has no {source}: its conditions are "
                                    "suspended")
                out.suspended.add(source)
                continue
            stale = [found[k].label for k in ks
                     if not is_current(ctx, Evaluation(source, found[k].label))]
            if stale and remembered.get(source) not in (None, _hash(_ids(locs, source))):
                out.warnings.append(
                    f"{source} has changed since {', '.join(sorted(set(stale)))} was "
                    f"measured: its conditions are suspended until it is measured again")
                out.suspended.add(source)
                continue
            if stale:
                out.warnings.append(f"{', '.join(sorted(set(stale)))} (per {source}) is "
                                    "out of date: filtering on what was measured before")
        passing = None
        for k in ks:
            c = found[k]
            ids, values = region_values(session, source, c.label)
            column = values.get(c.value)
            if column is None:
                out.warnings.append(f"{c.path()}: no such value (any more); nothing "
                                    "passes it")
                ok = set()
            else:
                ok = {r for r, v in zip(ids, column) if c.passes(v)}
            out.counts[k] = (len(ok), len(ids))
            passing = ok if passing is None else passing & ok
        passing = passing or set()
        out.passing[source] = passing
        if members:
            flag &= _members(session, source, passing)
    if len(out.suspended) == len(by_source) or not members:
        return out                      # nothing applies: no flag at all
    out.flag = flag
    return out


def _ids(locs, column: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(locs[column]))


def _members(session, source: str, passing: set) -> np.ndarray:
    """Which localizations of the session's table are in a passing region:
    of an id column by their id, of the ROIs through each ROI's rows (OR
    where ROIs overlap)."""
    locs = session.locs
    inside = np.zeros(len(locs), dtype=bool)
    if not passing:
        return inside
    if source != ROIS:
        ids = np.asarray(locs[source])
        wanted = np.array(sorted(int(float(r)) for r in passing), dtype=np.int64)
        good = np.isfinite(ids) if ids.dtype.kind == "f" else np.ones(len(ids), bool)
        inside[good] = np.isin(ids[good].astype(np.int64), wanted)
        return inside
    project = session.rois
    for roi_id in passing:
        roi = project.rois.get(roi_id)
        if roi is None or roi.file_id not in project.sources:
            continue
        inside[project.session_rows(roi)] = True
    return inside


def passing_rows(ctx, source: str) -> Optional[set]:
    """The regions of ``source`` the filter keeps, as a site table names them
    in ``roi_id`` (an ROI by its number, a region of an id column by its id),
    or None when the filter says nothing of ``source``: what the summaries
    honour (`region_results.rows`)."""
    session = getattr(ctx, "session", None)
    if session is None or not conditions(session.locs):
        return None
    if not any(c.source == source for c in conditions(session.locs)):
        return None
    outcome = session.region_filter_outcome()
    if source in outcome.suspended or source not in outcome.passing:
        return None
    kept = outcome.passing[source]
    if source == ROIS:
        rois = session.rois.rois
        return {str(rois[i].number) for i in kept if i in rois}
    return {str(r) for r in kept}


# ------------------------------------------------------------ writing it

def write(session, outcome: Outcome) -> None:
    """Put the flag into the session's table in place, and into the grouped
    table through `group_id` (a blink is in if any of its localizations
    is), every layer's masks following; with no flag, take it out."""
    from .mathparse import remember
    locs = session.locs
    if outcome.flag is None:
        for table in _tables(session):
            table.columns.pop(REGION_FILTER, None)
            table._buffers.pop(REGION_FILTER, None)
        return
    values = np.where(outcome.flag, IN, OUT).astype(np.float32)
    _put(locs, values)
    remember(locs, REGION_FILTER, "", grouped="any")
    grouped = session.tables.peek_grouped()
    if grouped is not None and "group_id" in locs and "group_id" in grouped.locs:
        gid = np.asarray(locs["group_id"], dtype=np.int64)
        top = int(max(gid.max(initial=0), np.asarray(grouped.locs["group_id"]).max(initial=0)))
        per = np.zeros(top + 1, dtype=np.float32)
        np.maximum.at(per, np.clip(gid, 0, top), values)
        _put(grouped.locs, per[np.clip(np.asarray(grouped.locs["group_id"], np.int64), 0, top)])
    for layer in session.layers:
        if not layer.is_image:
            layer.state.columns_changed([REGION_FILTER])


def _put(table, values: np.ndarray) -> None:
    table.columns[REGION_FILTER] = values
    table._buffers[REGION_FILTER] = values


def _tables(session):
    out = [session.locs]
    grouped = session.tables.peek_grouped()
    if grouped is not None:
        out.append(grouped.locs)
    return out


def remember_ids(session, outcome: Outcome) -> None:
    """For each id column whose conditions applied to current results, the
    hash of the column they applied to: what tells a drift correction (the
    same ids) from a new clustering (`evaluate`)."""
    from .region_results import Evaluation, is_current
    locs = session.locs
    doc = _state(locs)
    if not doc["conditions"]:
        return
    ctx = session.context()
    found = conditions(locs)
    for source in {c.source for c in found}:
        if source == ROIS or source in outcome.suspended or source not in locs:
            continue
        if all(is_current(ctx, Evaluation(c.source, c.label))
               for c in found if c.source == source):
            doc["ids"][source] = _hash(_ids(locs, source))
    locs.metadata[KEY] = doc
