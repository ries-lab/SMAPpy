"""Running plugins once per region: regions named by an id column, and the
progress of a run over many.

`docs/regions.md` has the decisions.  An ROI of the ROI manager is one kind of
region (`roi_manager.ROIProject`); this module adds the other -- the rows that
share a value of an integer column, a cluster's ``cluster_id`` or any id a
plugin wrote -- and what both share: the status line of a long run and the
way it is stopped.

**Membership is the id alone.**  A region's rows are the rows with its id;
each layer's own filter applies inside them, as for an ROI.  Id 0 and below
is "in no region" (DBSCAN's noise) and is never evaluated, and a region left
with no row is not either.

**Freshness is one signature per step, not per region.**  An ROI can be moved
alone, so its results are checked one by one; a column's regions change
together -- a new table, a new clustering -- so a step's results are current
or not as a whole.  The signature is the table (its length, columns, a sample
of positions, the id column itself), every layer's filters and grouping, and
the step, which is why checking 10^5 regions costs one digest instead of
10^5.

**The grouped table** of a layer, where it is current, is cut through
``group_id``: the blinks a region's localizations were linked into.  A blink
whose localizations fall in two regions is in both.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .locs import Localizations
from .region_table import RegionTable

# how often the status line is written, in seconds: often enough to look
# alive, rarely enough that writing it costs nothing next to the regions
REPORT_EVERY = 0.25


# ------------------------------------------------------------- progress

def progress_text(done: int, total: int, elapsed: float, what: str = "regions") -> str:
    """"37% (370 of 1000 regions), 120 a second, about 5 s left" -- the rate
    and the time left from the regions so far, once there are any."""
    if total <= 0:
        return f"no {what}"
    text = f"{100 * done // total}% ({done} of {total} {what})"
    if done and elapsed > 0:
        rate = done / elapsed
        left = (total - done) / rate
        text += f", {rate:.3g} a second"
        if done < total:
            text += f", about {_duration(left)} left"
    return text


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


class Progress:
    """The status of a run over ``total`` regions: call it with the number
    done.  ``report(text)`` gets the line, at most every `REPORT_EVERY`
    seconds and once at the end; ``tick(done, total)`` every time, for a
    progress bar."""

    def __init__(self, total: int, report: Optional[Callable[[str], None]] = None,
                 tick: Optional[Callable[[int, int], None]] = None,
                 what: str = "regions"):
        self.total, self.report, self.tick, self.what = total, report, tick, what
        self.started = time.perf_counter()
        self._last = -np.inf

    def __call__(self, done: int) -> None:
        if self.tick:
            self.tick(done, self.total)
        now = time.perf_counter()
        if self.report and (done >= self.total or now - self._last >= REPORT_EVERY):
            self._last = now
            self.report(progress_text(done, self.total, now - self.started, self.what))


def stopped(stop) -> bool:
    """Whether a run was asked to stop: ``stop`` is None, a
    `threading.Event`, or a callable (a progress dialog's ``wasCanceled``)."""
    if stop is None:
        return False
    if isinstance(stop, threading.Event):
        return stop.is_set()
    return bool(stop())


# ------------------------------------------------- regions of an id column

def split_ids(ids) -> Tuple[np.ndarray, List[np.ndarray]]:
    """The regions an id column names: ``(ids, rows)`` with the rows of each,
    sorted, ids above 0 only -- one sort, whatever the number of regions."""
    ids = np.asarray(ids)
    if ids.dtype.kind == "f":
        good = np.isfinite(ids)
        ids = np.where(good, ids, 0)
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    start = np.searchsorted(ordered, 0, side="right")     # skip 0 and below
    order, ordered = order[start:], ordered[start:]
    if not len(order):
        return np.zeros(0, ids.dtype), []
    cuts = np.flatnonzero(np.diff(ordered)) + 1
    groups = np.split(order, cuts)
    return ordered[np.r_[0, cuts]], [np.sort(g) for g in groups]


class ColumnRegions:
    """The regions an id column of the session's table names.

    ``column`` is the id (``cluster_id``); ``layer`` the one a plugin's
    ``ctx.selection`` is, as for an ROI the layer the ROI manager follows.
    Without a session, ``locs`` is the table and there is one layer, all of
    it selected.
    """

    def __init__(self, session=None, column: str = "cluster_id", layer: int = 0,
                 locs: Optional[Localizations] = None):
        self.session = session
        self.locs = locs if locs is not None else session.locs
        if column not in self.locs:
            raise ValueError(f"the table has no {column!r}: it has "
                             f"{', '.join(sorted(self.locs.keys()))}")
        self.column = column
        self.layer = layer
        self.ids, self.rows = split_ids(self.locs[column])
        self._grouped: Dict[int, Tuple[Localizations, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def region_ids(self) -> List[str]:
        return [str(int(i)) for i in self.ids]

    def inputs(self) -> Dict[str, Any]:
        """What every region's results depend on (see the module docstring)."""
        locs = self.locs
        n = len(locs)
        sample = np.arange(n)[:: max(1, n // 512)][:512]
        positions = [np.round(np.asarray(locs[c])[sample].astype(float), 3).tolist()
                     for c in ("x_nm", "y_nm") if c in locs]
        ids = np.ascontiguousarray(np.asarray(locs[self.column]))
        out = {"rows": n, "columns": sorted(locs.keys()), "positions": positions,
               "column": self.column, "ids": _hash(ids), "layer": self.layer}
        if self.session is not None:
            out["layers"] = [_layer_inputs(lay) for lay in self.session.layers
                             if not lay.is_image]
        return out

    def tables(self, k: int):
        """``(pinned, rows, layer, session)`` for region ``k``: every layer cut
        to its rows, as `ROIProject.region_context` cuts them to an ROI."""
        from .plugins import Selection
        rows = self.rows[k]
        name = f"{self.column} {int(self.ids[k])}"
        locs = self.locs[rows]
        if self.session is None:
            return ({0: {"ungrouped": (locs, Selection.all(len(rows))),
                         "shows": "ungrouped"}}, rows, 0, None)
        pinned = {}
        for i, lay in enumerate(self.session.layers):
            if lay.is_image:
                continue
            mask = np.asarray(lay.selection(i).mask, dtype=bool)
            entry = {"ungrouped": (locs, Selection(mask[rows], layer=i,
                                                   name=f"{lay.name}, {name}"))}
            grouped = self._grouped_rows(i, lay, rows)
            if grouped is not None:
                table, sel_mask, g = grouped
                entry["grouped"] = (table[g], Selection(sel_mask[g], layer=i,
                                                        name=f"{lay.name}, grouped, {name}"))
            pinned[i] = entry
        layer = min(self.layer, len(self.session.layers) - 1)
        if self.session.layers[layer].is_image:
            layer = self.session.first_locs_layer()
        return pinned, rows, layer, self.session

    def _grouped_rows(self, i, lay, rows):
        """The grouped table's rows for these localizations, via ``group_id``."""
        locset = lay.state.sets.get("grouped")
        if locset is None or lay.state.grouped_stale or "group_id" not in self.locs \
                or "group_id" not in locset.locs:
            return None
        if i not in self._grouped:
            gid = np.asarray(locset.locs["group_id"])
            order = np.argsort(gid, kind="stable")
            self._grouped[i] = (locset.locs, np.asarray(locset.filter.mask, dtype=bool),
                                (gid[order], order))
        table, mask, (sorted_ids, order) = self._grouped[i]
        wanted = np.unique(np.asarray(self.locs["group_id"])[rows])
        at = np.searchsorted(sorted_ids, wanted)
        found = (at < len(sorted_ids))
        found[found] = sorted_ids[at[found]] == wanted[found]
        return table, mask, np.sort(order[at[found]])


def _hash(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def _layer_inputs(lay) -> Dict[str, Any]:
    grouped = lay.state.sets.get("grouped") if lay.grouped else None
    return {"filters": lay.state.sets["ungrouped"].filter.ranges,
            "grouped_filters": None if grouped is None else grouped.filter.ranges,
            "shows": "grouped" if lay.grouped else "ungrouped"}


# --------------------------------------------------------------- the runner

def evaluate_regions(source: ColumnRegions, steps: Sequence, tables: Optional[
        Dict[str, RegionTable]] = None, report: Optional[Callable[[str], None]] = None,
        tick: Optional[Callable[[int, int], None]] = None, stop=None,
        reuse: bool = True) -> Dict[str, RegionTable]:
    """Run ``steps`` (`roi_manager.pipeline.Step`s) once in each region of
    ``source``, a table per step by its label.

    ``tables`` holds what earlier runs left: with ``reuse`` a step whose
    table is current for every region (same table, filters and settings) is
    kept as it is, and only the others run.  A step that raises costs its
    own entry in that region, as for an ROI.  ``stop`` ends the run between
    regions, keeping what was measured; the regions not reached have no row.
    """
    import json
    from .roi_manager.core import digest, json_text
    tables = dict(tables or {})
    inputs = json.loads(json_text(source.inputs()))
    data = digest(inputs)
    ids = source.region_ids()
    todo = []
    for step in steps:
        signature = digest({"inputs": inputs, "step": step.identity()})
        table = tables.get(step.label)
        if reuse and table is not None and len(table) == len(ids) and all(
                r in table and table.entry(r).get("signature") == signature for r in ids):
            continue
        tables[step.label] = RegionTable(step.label)
        todo.append((step, signature))
    if not todo:
        return tables
    progress = Progress(len(ids), report, tick)
    for k, region_id in enumerate(ids):
        if stopped(stop):
            break
        cut = source.tables(k)
        for step, signature in todo:
            entry = _run_one(step, cut, region_id)
            entry["signature"] = signature
            tables[step.label].set(region_id, entry, step.as_record(), data)
        progress(k + 1)
    return tables


def _run_one(step, cut, region_id) -> Dict[str, Any]:
    from .plugins import Context
    from .roi_manager.core import region_entry
    pinned, rows, layer, session = cut
    locs, selection = pinned[layer]["ungrouped"]
    try:
        ctx = Context(session=session, locs=locs, selection=selection, layer=layer,
                      pinned=pinned, rows=rows, region_id=region_id)
        return region_entry(step.plugin.run(ctx, step.settings))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
