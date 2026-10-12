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
    if seconds < 1:
        return "under a second"
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
        return column_inputs(self.locs, self.column, self.layer, self.session)

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


class ContextRegions(ColumnRegions):
    """The regions of an id column of a context's own table -- a block
    inside another block's region: the region narrows, and every layer's
    selection is cut on from the region's."""

    def __init__(self, ctx, column: str):
        if column not in ctx.locs:
            raise ValueError(f"the region's table has no {column!r}: a step "
                             "before the block has to write it")
        self.ctx = ctx
        self.session = None
        self.locs = ctx.locs
        self.column = column
        self.layer = ctx.layer
        self.ids, self.rows = split_ids(self.locs[column])
        self._grouped = {}

    def inputs(self) -> Dict[str, Any]:
        return {"rows": len(self.locs), "column": self.column,
                "ids": _hash(np.ascontiguousarray(np.asarray(self.locs[self.column])))}

    def tables(self, k: int):
        from .plugins import Selection
        rows = self.rows[k]
        locs = self.locs[rows]
        pinned = {}
        for i, entry in (self.ctx._pinned or {}).items():
            table, selection = entry["ungrouped"]
            if len(table) != len(self.locs):
                continue                     # a layer that is not this table's
            pinned[i] = {"ungrouped": (locs, Selection(np.asarray(selection.mask)[rows],
                                                       layer=i, name=selection.name)),
                         "shows": "ungrouped"}
        if self.layer not in pinned:
            pinned[self.layer] = {"ungrouped": (locs, Selection.all(len(rows))),
                                  "shows": "ungrouped"}
        return pinned, rows, self.layer, self.ctx.session


def column_inputs(locs: Localizations, column: str, layer: int = 0,
                  session=None) -> Dict[str, Any]:
    """What the results of ``column``'s regions depend on: the table (its
    length, columns, a sample of positions, the id column itself), every
    layer's filters and grouping, and the layer a run followed.

    A function rather than only `ColumnRegions.inputs` so that a reader can
    check a table for staleness (`region_results.is_current`) without the
    sort that splits the regions.  The columns written from regions' own
    results (`REGION_COLUMNS`) are left out of the list: writing a cluster's
    number onto its localizations does not change what was measured in it,
    and counting it would make the results it was written from out of date.
    """
    n = len(locs)
    sample = np.arange(n)[:: max(1, n // 512)][:512]
    positions = [np.round(np.asarray(locs[c])[sample].astype(float), 3).tolist()
                 for c in ("x_nm", "y_nm") if c in locs]
    ids = np.ascontiguousarray(np.asarray(locs[column]))
    written = set(locs.metadata.get(REGION_COLUMNS) or {})
    out = {"rows": n, "columns": sorted(c for c in locs.keys() if c not in written),
           "positions": positions, "column": column, "ids": _hash(ids), "layer": layer}
    if session is not None:
        out["layers"] = [_layer_inputs(lay) for lay in session.layers
                         if not lay.is_image]
    return out


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
        reuse: bool = True, keep_figures: bool = False, after=None,
        tag: bool = False) -> Dict[str, RegionTable]:
    """Run ``steps`` (`roi_manager.pipeline.Step`s) once in each region of
    ``source``, a table per step by its label.

    ``tables`` holds what earlier runs left: with ``reuse`` a step whose
    table is current for every region (same table, filters and settings) is
    kept as it is, and only the others run.  A step that raises costs its
    own entry in that region, as for an ROI.  ``stop`` ends the run between
    regions, keeping what was measured; the regions not reached have no row.
    ``keep_figures`` keeps each region's figure data (`Plugin.keep`) -- off
    by default, since to look at clusters one makes ROIs of them.

    The steps of a region run in one `RegionScratch`, a new table handed on;
    so when one step must run, they all do -- a stored result of a step
    before it cannot hand on the table it measured.  ``after(region_id,
    scratch, original)`` is told each region's tables at the end, and
    ``tag`` gives them their rows (a block's write-back).
    """
    import json
    from .roi_manager.core import changes_table, digest, json_text
    tables = dict(tables or {})
    inputs = json.loads(json_text(source.inputs()))
    data = digest(inputs)
    ids = source.region_ids()
    signatures, before = {}, []
    for step in steps:
        signatures[step.label] = digest({"inputs": inputs, "step": step.identity(),
                                         **({"after": list(before)} if before else {})})
        if changes_table(step):
            before.append(step.identity())
    current = reuse and after is None and all(
        step.label in tables and len(tables[step.label]) == len(ids) and all(
            r in tables[step.label]
            and tables[step.label].entry(r).get("signature") == signatures[step.label]
            for r in ids) for step in steps)
    if current:
        return tables
    for step in steps:
        tables[step.label] = RegionTable(step.label)
    progress = Progress(len(ids), report, tick)
    for k, region_id in enumerate(ids):
        if stopped(stop):
            break
        scratch = RegionScratch(source.tables(k), None, region_id, tag=tag)
        original = scratch.locs
        for step in steps:
            entry = _run_one(step, scratch, keep_figures)
            entry["signature"] = signatures[step.label]
            tables[step.label].set(region_id, entry, step.as_record(), data)
        if after is not None:
            after(region_id, scratch, original)
        progress(k + 1)
    return tables


def _run_one(step, scratch, keep: bool = False) -> Dict[str, Any]:
    from .roi_manager.core import region_entry
    try:
        result = step.plugin.run(scratch.context(), step.settings)
        scratch.apply(result, step.label)
        return region_entry(result, step.plugin, keep)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


# ------------------------------------------- a region's tables, step by step

ROW = "_row"        # the hidden column: which row of the real table a row is


class RegionScratch:
    """One region's tables while the steps of a block run in it.

    Starts as what `ROIProject._region_tables` or `ColumnRegions.tables` cut;
    a step that hands back a new table (`Result.locs`) replaces the region's
    table for the steps after it -- each layer's filter applied to it again,
    the grouped table dropped, since a new table has not been linked.  With
    ``tag``, the table carries `ROW`, the row of the real table each row came
    from, which is what a block writes columns back by; a plugin that builds
    its table from the columns it was given keeps it.
    """

    def __init__(self, cut, region=None, region_id: Optional[str] = None,
                 tag: bool = False):
        pinned, rows, layer, session = cut
        self.region, self.region_id = region, region_id
        self.layer, self.session = layer, session
        self.start_rows = np.asarray(rows)
        self.rows = self.start_rows              # None once the table has changed
        self.changed = False
        # tables a step made of regions inside this one (a block in a block),
        # by the step's label: kept beside its numbers by the outer block
        self.children: Dict[str, Dict[str, RegionTable]] = {}
        if tag:
            pinned = {i: dict(entry) for i, entry in pinned.items()}
            first = next(iter(pinned.values()))["ungrouped"][0]
            tagged = Localizations({**first.columns, ROW: self.start_rows.astype(np.int64)},
                                   dict(first.metadata))
            for entry in pinned.values():
                entry["ungrouped"] = (tagged, entry["ungrouped"][1])
        self.pinned = pinned

    @property
    def locs(self) -> Localizations:
        return self.pinned[self.layer]["ungrouped"][0]

    def context(self, **extra):
        from .plugins import Context
        locs, selection = self.pinned[self.layer]["ungrouped"]
        return Context(session=self.session, locs=locs, selection=selection,
                       layer=self.layer, pinned=self.pinned, rows=self.rows,
                       region=self.region, region_id=self.region_id, **extra)

    def apply(self, result, label: str = "") -> None:
        """What a step handed back: a new table becomes the region's, and the
        tables of regions inside this one are kept as its children."""
        if result is None:
            return
        inner = (result.data or {}).get("region_tables")
        if inner:
            self.children[label] = dict(inner)
        if result.locs is None:
            return
        from .filter import LocFilter
        from .plugins import Selection
        new = result.locs
        for i, entry in self.pinned.items():
            mask = np.ones(len(new), dtype=bool)
            if self.session is not None and i < len(self.session.layers):
                ranges = self.session.layers[i].state.sets["ungrouped"].filter.ranges
                f = LocFilter(new)
                for name, (lo, hi) in ranges.items():
                    if name in new:
                        f.set(name, lo, hi)
                mask = np.asarray(f.mask, dtype=bool)
            old = entry["ungrouped"][1]
            entry.clear()
            entry["ungrouped"] = (new, Selection(mask, layer=i, roi=self.region,
                                                 name=old.name))
            entry["shows"] = "ungrouped"
        self.rows = None
        self.changed = True


class WriteBack:
    """The columns a block writes back into the real table, and how.

    ``columns`` are merged by `ROW` after every region has run: a row in no
    region keeps its value (NaN, or 0 for an id or a flag, in a new column);
    a row in several overlapping regions takes the value the ``rule`` says
    -- ``last`` (the default) or ``first`` region's, the ``mean``, or
    ``any`` for a flag.  ``use`` among them also flags the rows a region's
    steps removed (False), rather than deleting them: a deletion merged over
    overlapping regions would mean nothing.
    """

    RULES = ("last", "first", "mean", "any")

    def __init__(self, columns: Sequence[str] = (), rule: str = "last"):
        if rule not in self.RULES:
            raise ValueError(f"merge rule {rule!r} is not one of {', '.join(self.RULES)}")
        self.columns = [c for c in columns if c and c != ROW]
        self.rule = rule
        self.parts: List[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = []
        self.missing: Dict[str, int] = {}

    def __bool__(self) -> bool:
        return bool(self.columns)

    def add(self, scratch: RegionScratch) -> None:
        """One region's final table, as far as the written columns go."""
        locs = scratch.locs
        if ROW not in locs:
            raise ValueError("a step built a table without the rows it came from "
                             f"({ROW}): nothing of it can be written back")
        values = {}
        for c in self.columns:
            if c in locs:
                values[c] = np.asarray(locs[c])
            elif c != "use":
                self.missing[c] = self.missing.get(c, 0) + 1
        self.parts.append((scratch.start_rows, np.asarray(locs[ROW]), values))

    def merge(self, locs: Localizations) -> Localizations:
        """The real table with the written columns merged in."""
        n = len(locs)
        columns = dict(locs.columns)
        for c in self.columns:
            pieces = [(rows, values[c]) for _, rows, values in self.parts if c in values]
            if c == "use":
                for start, rows, values in self.parts:
                    gone = np.setdiff1d(start, rows)
                    pieces.append((gone, np.zeros(len(gone), bool)))
            if not pieces:
                continue
            columns[c] = merge_pieces(n, pieces, self.rule, existing=columns.get(c),
                                      fill=True if c == "use" else None)
        return Localizations(columns, dict(locs.metadata))


def merge_pieces(n: int, pieces: Sequence[Tuple[np.ndarray, np.ndarray]],
                 rule: str = "last", existing: Optional[np.ndarray] = None,
                 fill: Any = None) -> np.ndarray:
    """One column of ``n`` rows from values written over some of them.

    ``pieces`` are ``(rows, values)`` in the order the regions ran; a row in
    several takes the ``rule``'s value (`WriteBack.RULES`).  A row in none
    keeps ``existing`` where the column was there before, else ``fill`` --
    by default NaN for numbers with a fraction and 0 for an id or a flag.
    """
    dtype = np.result_type(*[np.asarray(v).dtype for _, v in pieces])
    if existing is not None:
        out = np.array(existing, copy=True)
        out = out.astype(np.result_type(out.dtype, dtype), copy=False)
    elif fill is not None:
        out = np.full(n, fill, dtype=np.result_type(dtype, np.asarray(fill).dtype))
    elif dtype.kind == "f":
        out = np.full(n, np.nan, dtype=dtype)
    else:
        out = np.zeros(n, dtype=dtype)
    if rule == "mean" and out.dtype.kind in "fiu":
        total = np.zeros(n)
        count = np.zeros(n)
        for rows, v in pieces:
            np.add.at(total, rows, np.asarray(v, dtype=float))
            np.add.at(count, rows, 1)
        out = out.astype(float)
        hit = count > 0
        out[hit] = total[hit] / count[hit]
    elif rule == "any":
        out = out.astype(bool) if existing is None else out
        hit = np.zeros(n, dtype=bool)
        touched = np.zeros(n, dtype=bool)
        for rows, v in pieces:
            np.logical_or.at(hit, rows, np.asarray(v).astype(bool))
            touched[rows] = True
        out[touched] = hit[touched]
    else:
        for rows, v in (pieces if rule == "last" else pieces[::-1]):
            out[rows] = v
    return out


# ----------------------------------------------- region values onto rows

# the metadata entry naming the columns written from regions' results
# (`values_onto`): only those may be written over again, so a value called
# like a localization's own column (``photons``) never replaces it
REGION_COLUMNS = "region_columns"
# what a region's own number is asked for by, beside its measured values
REGION_NUMBER = "roi_id"


def region_members(ctx, regions: str) -> List[Tuple[str, np.ndarray]]:
    """``(region id, rows of ctx.locs)`` of every region: the reviewed,
    included ROIs,
    in the order they were made, or the regions of the id column
    ``regions``.  Membership is the region alone, as when a block runs."""
    from .chain import BLOCK_ROIS
    if regions == BLOCK_ROIS:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs: find or draw some first")
        out = []
        for roi_id, roi in project.rois.items():
            if roi.reviewed and roi.use:
                _, rows, _, _ = project._region_tables(roi, project.region(roi))
                out.append((str(roi_id), np.asarray(rows, dtype=np.int64)))
        return out
    if regions not in ctx.locs:
        raise ValueError(f"the table has no {regions!r}: it has "
                         f"{', '.join(sorted(ctx.locs.keys()))}")
    ids, rows = split_ids(ctx.locs[regions])
    return [(str(int(i)), r) for i, r in zip(ids, rows)]


def values_onto(n: int, members: Sequence[Tuple[str, np.ndarray]],
                values: Dict[str, Dict[str, float]], rule: str = "last"
                ) -> Dict[str, np.ndarray]:
    """Each region's value written onto its rows: ``values`` is ``{column:
    {region id: value}}``, and a row takes its region's -- the ``rule``'s
    where regions overlap -- or NaN in none, or in a region with no value
    (one the step failed in, or that came after the evaluation)."""
    out = {}
    for name, per in values.items():
        pieces = []
        for region_id, rows in members:
            v = per.get(region_id)
            if v is None or not len(rows):
                continue
            v = float(v)
            if np.isfinite(v):
                pieces.append((rows, np.full(len(rows), v)))
        out[name] = (merge_pieces(n, pieces, rule, fill=np.nan) if pieces
                     else np.full(n, np.nan))
    return out


def table_values(table: RegionTable, names: Sequence[str] = ()) -> Dict[str, Dict[str, float]]:
    """``{column: {region id: value}}`` of a region table's numeric values --
    ``names`` of them, or every one -- the rows where the step failed left
    out.  A name it does not have is an error naming the ones it has."""
    numeric = {k: v for k, v in table.values.items()
               if all(x is None or (isinstance(x, (int, float, np.number))
                                    and not isinstance(x, str)) for x in v)}
    wanted = list(names) or list(numeric)
    missing = [k for k in wanted if k not in numeric]
    if missing:
        raise ValueError(f"{table.label or 'the evaluation'} has no numeric "
                         f"{', '.join(missing)}: it has {', '.join(sorted(numeric))}")
    return {k: {r: numeric[k][i] for i, r in enumerate(table.ids)
                if not table.errors[i] and numeric[k][i] is not None}
            for k in wanted}


def check_writable(locs: Localizations, names: Sequence[str]) -> None:
    """Refuse to write over a column that was not written from regions."""
    own = set(locs.metadata.get(REGION_COLUMNS) or {})
    taken = [c for c in names if c in locs and c not in own]
    if taken:
        raise ValueError(f"the table already has {', '.join(taken)}, which was not "
                         "written from regions: give the columns a prefix")


def with_region_columns(locs: Localizations, columns: Dict[str, np.ndarray],
                        origin: Dict[str, Any]) -> Localizations:
    """The table with ``columns`` added, each remembered as written from
    regions (`REGION_COLUMNS`: ``origin`` says from what) and with a rule for
    grouping: a blink takes the mean of its localizations' values, the
    highest of a region's number."""
    from .mathparse import remember
    check_writable(locs, list(columns))
    out = Localizations({**locs.columns, **columns}, dict(locs.metadata))
    own = dict(out.metadata.get(REGION_COLUMNS) or {})
    for name in columns:
        own[name] = dict(origin)
        remember(out, name, "", grouped="max" if name.endswith(REGION_NUMBER) else "mean")
    out.metadata[REGION_COLUMNS] = own
    return out


def changed_columns(before: Localizations, scratch: RegionScratch) -> List[str]:
    """The columns a region's steps added or changed: what a block warns
    about when it is not told to write them back."""
    after = scratch.locs
    if not scratch.changed:
        return []
    out = [c for c in after.keys() if c != ROW and c not in before]
    if ROW in after:
        rows = np.searchsorted(scratch.start_rows, np.asarray(after[ROW]))
        for c in after.keys():
            if c == ROW or c not in before:
                continue
            old = np.asarray(before[c])[rows]
            new = np.asarray(after[c])
            same = (old == new) | (_isnan(old) & _isnan(new))
            if not np.all(same):
                out.append(c)
    return out


def _isnan(a: np.ndarray) -> np.ndarray:
    return np.isnan(a) if a.dtype.kind == "f" else np.zeros(a.shape, dtype=bool)
