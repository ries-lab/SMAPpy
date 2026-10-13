"""File-associated analysis ROIs, independent of the interactive interface.

All geometry is in nm. Source tables are immutable snapshots; selection always
uses the same ViewState filter as rendering, never the visible field of view.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional
from uuid import uuid4

import h5py
import numpy as np

from ..group import GroupSettings
from ..io.hdf5 import load_localizations
from ..locs import Localizations, to_nm
from ..region_table import RegionTable
from ..regions import Region
from ..layer_state import ViewState


def json_text(value):
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot save {type(obj).__name__}")
    return json.dumps(value, default=default, sort_keys=True)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def point(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError("A position must contain two finite coordinates in nm")
    return p.tolist()


def polygon_vertices(value):
    v = np.asarray(value, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("A polygon needs at least three finite xy vertices")
    # Use local coordinates to avoid cancellation far from the file origin.
    w = v - v[0]
    area = np.sum(w[:, 0] * np.roll(w[:, 1], -1) -
                  w[:, 1] * np.roll(w[:, 0], -1))
    if abs(area) < 1e-10:
        raise ValueError("A polygon must enclose nonzero area")
    return v.tolist()


def rows_in(locs, index, region, keep=None):
    """The rows of ``locs`` inside ``region``, sorted, through its spatial
    index: only the rows in the region's bounding box are tested.  ``keep``,
    over the whole table, narrows them first (a file of several)."""
    x0, y0, x1, y1 = region.bounds
    candidates = index.query(x0, x1, y0, y1)
    if keep is not None:
        candidates = candidates[keep[candidates]]
    x = np.asarray(locs["x_nm"][candidates], dtype=float)
    y = np.asarray(locs["y_nm"][candidates], dtype=float)
    return np.sort(candidates[region.mask(x, y)])


def changes_table(step) -> bool:
    """Does this step hand back a new table (`Plugin.changes_locs_for`)?  In
    a region's run its table is what the steps after it measure."""
    try:
        return bool(step.plugin.changes_locs_for(step.settings))
    except Exception:
        return bool(getattr(step.plugin, "changes_locs", False))


def region_entry(result, plugin=None, keep: bool = False) -> Dict:
    """What one run in one region stores: its `Result.values`, a number
    each, and its `Result.arrays`.

    A value that is not a number (or a string, a flag, None) is refused
    rather than stored: the values are the region table's columns, and a
    list in a column is a table nobody can histogram.  An array belongs in
    `arrays`, kept here as JSON lists beside the values; `RegionTable`
    writes them to the file end to end.

    With ``keep``, what ``plugin.keep(result)`` returns -- the data its
    figure is drawn from -- goes with the row as ``kept``, so the figure can
    come back (`Plugin.restore`) without measuring again.
    """
    values = {}
    for name, value in (result.values or {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"value {name!r} is a {type(value).__name__}, not a "
                            "number: put it in Result.arrays")
        values[name] = value
    entry = {"values": values}
    if result.arrays:
        entry["arrays"] = {name: np.asarray(a).tolist()
                           for name, a in result.arrays.items()}
    if keep and plugin is not None:
        try:
            kept = plugin.keep(result)
        except Exception:
            kept = None             # the numbers stand; only the replot is lost
        if kept:
            entry["kept"] = kept
    return json.loads(json_text(entry))


@dataclass
class ROI:
    file_id: str
    center: list
    id: str = field(default_factory=lambda: uuid4().hex)
    # what the user sees and a column carries (`roi_id`): given once, from the
    # project's counter, and never reused, so removing ROI 3 leaves ROI 5 as
    # 5 -- in the list, on the localizations and in the results alike.  The
    # uuid stays the key the results are stored under.
    number: int = 0
    polygon: Optional[list] = None
    direction: Optional[list] = None  # two absolute xy endpoints
    use: bool = True
    comment: str = ""
    origin: dict = field(default_factory=lambda: {"method": "manual"})

    def region(self, shape: str, size_nm: float) -> Region:
        """This ROI's shape: its own polygon, or the project's ``shape`` of
        ``size_nm`` around its centre -- which the ROI does not know, since
        changing it is one edit for every ROI of the project.  So the shape
        is worked out when asked for rather than held."""
        return Region.site(self.center, shape, size_nm, self.polygon)

    @classmethod
    def from_dict(cls, saved: dict) -> "ROI":
        """An ROI as a file kept it.  Files from before the review flag went
        carry ``reviewed``: it was never shown in the GUI, where every ROI
        counted as reviewed, so it is dropped and ``use`` alone decides."""
        names = {f.name for f in fields(cls)}
        roi = cls(**{k: v for k, v in saved.items() if k in names})
        point(roi.center)
        if roi.polygon is not None:
            polygon_vertices(roi.polygon)
        return roi


# Past this nobody is walking back through a list of edits.
MAX_ROI_STEPS = 20


@dataclass
class ROIStep:
    """One edit of the ROIs, and the ROIs as they were before it."""
    label: str
    rois: Dict[str, "ROI"]


def copy_rois(rois: Dict[str, "ROI"]) -> Dict[str, "ROI"]:
    """A copy that later edits cannot reach.  Shallow per ROI, which is
    enough because an ROI is only ever edited by assigning its fields --
    ``roi.use = ...``, ``roi.polygon = [...]``, never ``roi.center[0] = ...``
    -- so the lists and the origin dict can be shared."""
    return {roi_id: replace(roi) for roi_id, roi in rois.items()}


class ROIHistory:
    """Undo and redo for the set of ROIs, apart from the session's undo.

    The session's undo is for tables (`smappy.undo` says why only tables);
    this one is for the ROI manager's edits: a segmenter's run, an ROI
    added, removed, excluded or reshaped.  Not for evaluations, which can be
    run again, and whose results stay with the ROI's id -- so an ROI brought
    back by an undo brings its numbers back with it.  Whole copies of the
    ROIs rather than diffs: an ROI is a few small fields, and a step of a
    few thousand costs less than a millisecond's worth of thinking about
    what a diff would have to cover.
    """

    def __init__(self, max_steps: int = MAX_ROI_STEPS):
        self.max_steps = max_steps
        self._undo: List[ROIStep] = []
        self._redo: List[ROIStep] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1].label if self._redo else None

    def push(self, step: ROIStep) -> None:
        self._undo.append(step)
        del self._undo[:-self.max_steps]
        self._redo.clear()

    def undo(self, rois: Dict[str, "ROI"]) -> Optional[ROIStep]:
        """Put the ROIs before the last edit into `rois`; the edit undone."""
        return self._travel(rois, self._undo, self._redo)

    def redo(self, rois: Dict[str, "ROI"]) -> Optional[ROIStep]:
        return self._travel(rois, self._redo, self._undo)

    @staticmethod
    def _travel(rois, source, target) -> Optional[ROIStep]:
        if not source:
            return None
        step = source.pop()
        target.append(ROIStep(step.label, copy_rois(rois)))
        # in place: whoever holds the project's dict keeps holding the ROIs
        rois.clear()
        rois.update(copy_rois(step.rois))
        return step

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class Source:
    def __init__(self, locs, path=None, name=None, source_id=None):
        if "x_nm" not in locs or "y_nm" not in locs:
            pixelsize = locs.metadata.get("pixelsize_nm")
            if pixelsize is None:
                raise ValueError("ROI analysis needs nm coordinates or pixelsize_nm metadata")
            locs = to_nm(locs, positive(pixelsize, "Pixel size"))
        if "x_nm" not in locs or "y_nm" not in locs:
            raise ValueError("Source has no x_nm/y_nm coordinates")
        columns = {}
        h = hashlib.sha256()
        for key in sorted(locs.columns):
            a = np.array(locs[key], copy=True, order="C")
            if a.ndim != 1 or len(a) != len(locs) or a.dtype.kind not in "biuf":
                raise ValueError(f"{key}: expected a numeric column with {len(locs)} rows")
            h.update(key.encode())
            h.update(a.dtype.str.encode())
            h.update(memoryview(a).cast('B'))
            a.flags.writeable = False
            columns[key] = a
        h.update(json_text(locs.metadata).encode())
        self.fingerprint = h.hexdigest()
        self.id = source_id or uuid4().hex
        self.path = str(Path(path).resolve()) if path else None
        self.name = name or (Path(path).name if path else "Untitled")
        self.state = ViewState(Localizations(columns, dict(locs.metadata)))


class ROIProject:
    def __init__(self):
        self.sources: Dict[str, Source] = {}
        self.rois: Dict[str, ROI] = {}
        self.size_nm = 300.0  # circle diameter or square side length
        self.shape = "circle"
        self.tile_nm = 0.0    # a grid over each file for a systematic walk; 0 = none
        # the bounds a layer opens with, so a scripted project filters as the GUI does
        from ..session import DEFAULT_BOUNDS
        self.filters = dict(DEFAULT_BOUNDS)
        self.grouped = False
        self.group_settings = GroupSettings()
        # The results, one table per evaluation by its name, a row per ROI
        # (`RegionTable`): the newest result of each evaluation on each ROI,
        # with the step that made it.  Kept in the order they were last
        # written, so the newest is last.  Rows of ROIs that were removed
        # stay until the project is saved, so an undo brings the ROI back
        # with its numbers.
        self.tables: Dict[str, RegionTable] = {}
        # What was run, newest last: the pipeline, when, and whether a chain
        # or one site's re-evaluation did it -- a log, without the results,
        # which are in `tables`
        self.runs = []
        self.navigation = {}  # GUI state only; never used for extraction
        # The per-region steps, a `chain.Block`: what the steps window edits
        # and runs, and what the ROI manager measures an ROI with.  It
        # travels with the project because the columns of a results table
        # mean nothing without knowing which steps produced them, with what
        # parameters.  No steps: the newest run's, or the default ones
        # (`resolved`).
        from ..chain import Block
        self.block = Block()
        # What a segmenter's Preview proposed and nobody has added yet:
        # ``{"file_id", "method", "sites": [{"center", "use"}], "replaces":
        # [roi ids]}``, or None.  Drawn by the manager and never saved -- it
        # exists so that a finder's settings can be tried without each try
        # leaving ROIs behind to be removed by hand.
        self.segment_preview = None
        self.history = ROIHistory()
        # the number the next ROI is given (`ROI.number`): it only counts up
        # and is saved with the ROIs, so an undone ROI's number is not handed
        # to the next one, and its redo brings it back unchanged
        self.next_number = 1

    def add_file(self, path):
        return self.add_source(load_localizations(path), path=path)

    def add_source(self, locs, path=None, name=None, source_id=None):
        if path and any(s.path == str(Path(path).resolve()) for s in self.sources.values()):
            raise ValueError(f"Source already belongs to this project: {path}")
        source = Source(locs, path, name, source_id)
        if source.id in self.sources:
            raise ValueError("Duplicate source ID")
        self.sources[source.id] = source
        return source

    def set_geometry(self, size_nm, shape=None):
        size = positive(size_nm, "ROI size")
        shape = shape or self.shape
        if shape not in ("circle", "square"):
            raise ValueError("Shape must be circle or square")
        self.size_nm, self.shape = size, shape

    def set_tiles(self, size_nm):
        """Cover each file with a grid of squares this wide; 0 turns it off."""
        self.tile_nm = 0.0 if not size_nm else positive(size_nm, "Tile size")

    def tiles(self, file_id):
        """The grid over one file, row by row: ``(x0, y0, x1, y1)`` in nm.

        Tiles are not stored: they follow the file's extent and the one size,
        so they never go stale.  What they are for is a systematic walk --
        every part of the data looked at once, in a fixed order, rather than
        whichever bright spot the eye landed on.
        """
        if not self.tile_nm or file_id not in self.sources:
            return []
        x0, y0, x1, y1 = self.state(file_id).index.bounds
        size = self.tile_nm
        nx = max(1, int(np.ceil((x1 - x0) / size)))
        ny = max(1, int(np.ceil((y1 - y0) / size)))
        return [(x0 + i * size, y0 + j * size, x0 + (i + 1) * size, y0 + (j + 1) * size)
                for j in range(ny) for i in range(nx)]

    def tile_at(self, file_id, x, y):
        """The index of the tile a point falls in, or None outside the grid.

        Tiles share their edges, so the ranges are half open: a point on a
        boundary belongs to the tile it opens, and only the far edge of the
        last row and column falls back to the tile it closes.
        """
        tiles = self.tiles(file_id)
        for i, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            if tx0 <= x < tx1 and ty0 <= y < ty1:
                return i
        for i, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                return i
        return None

    def set_filters(self, ranges):
        normalized = {}
        for name, bounds in ranges.items():
            if name in ("filenumber", "file_id"):
                continue
            lo, hi = (None if v is None else float(v) for v in bounds)
            if any(v is not None and not np.isfinite(v) for v in (lo, hi)):
                raise ValueError(f"{name}: bounds must be finite or blank")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"{name}: lower bound exceeds upper bound")
            if lo is not None or hi is not None:
                normalized[name] = (lo, hi)
        self.filters = normalized

    def state(self, file_id):
        """The file's view under the project's bounds and grouping.  The
        grouped table is its `Tables`' cache, keyed by the link settings, so
        a change of them links again when it is next asked for."""
        source = self.sources[file_id]
        state = source.state
        state.tables.settings = self.group_settings
        state.show_grouped(self.grouped)
        state.set_ranges({k: v for k, v in self.filters.items()
                          if k not in ("filenumber", "file_id")})
        return state

    @contextmanager
    def edit(self, label: str):
        """An undoable edit of the ROIs: ``with project.edit("Remove ROI 3"):``.

        Yields the step, whose label may be finished once the edit knows
        what it did; nothing is recorded when the ROIs come out unchanged,
        so a run that found nothing leaves no step to undo.
        """
        step = ROIStep(label, copy_rois(self.rois))
        yield step
        if step.rois != self.rois:
            self.history.push(step)

    def undo(self) -> Optional[str]:
        """Undo the last edit of the ROIs; what it was, or None."""
        step = self.history.undo(self.rois)
        return step.label if step else None

    def redo(self) -> Optional[str]:
        step = self.history.redo(self.rois)
        return step.label if step else None

    def add_roi(self, file_id, center, polygon=None, origin=None):
        if file_id not in self.sources:
            raise KeyError(file_id)
        roi = ROI(file_id, point(center), polygon=(None if polygon is None else
                  polygon_vertices(polygon)))
        if origin is not None:
            roi.origin = json.loads(json_text(origin))
        roi.number = self.next_number
        self.next_number += 1
        self.rois[roi.id] = roi
        return roi

    def numbers(self) -> Dict[str, int]:
        """Each ROI's number (`ROI.number`), by its id."""
        return {roi_id: roi.number for roi_id, roi in self.rois.items()}

    def roi_numbered(self, number) -> Optional[ROI]:
        """The ROI with this number -- a result row's ``roi_id`` -- or None."""
        try:
            number = int(number)
        except (TypeError, ValueError):
            return None
        return next((r for r in self.rois.values() if r.number == number), None)

    def number_rois(self, next_number=None) -> None:
        """Numbers for the ROIs that have none (a file from before they were
        kept), in the order the ROIs were made and after the highest in use,
        and a counter past all of them."""
        start = max([int(next_number or 1)]
                    + [r.number + 1 for r in self.rois.values() if r.number > 0])
        seen = set()
        for roi in self.rois.values():
            if roi.number <= 0 or roi.number in seen:
                roi.number = start
                start += 1
            seen.add(roi.number)
        self.next_number = start

    def move_roi(self, roi_id, center):
        """Translate the entire ROI, including its polygon and direction line."""
        roi = self.rois[roi_id]
        center = point(center)
        delta = np.asarray(center) - roi.center
        for name in ("polygon", "direction"):
            if getattr(roi, name) is not None:
                setattr(roi, name, (np.asarray(getattr(roi, name)) + delta).tolist())
        roi.center = center

    def geometry(self, roi):
        if roi.polygon is not None:
            return {"center": roi.center, "polygon": roi.polygon,
                    "direction": roi.direction}
        return {"center": roi.center, "shape": self.shape, "size_nm": self.size_nm,
                "direction": roi.direction}

    def region(self, roi) -> Region:
        """The ROI's shape, with this project's shared shape and size."""
        return roi.region(self.shape, self.size_nm)

    def indices(self, roi):
        state = self.state(roi.file_id)
        return rows_in(state.locs, state.index, self.region(roi),
                       keep=np.asarray(state.filter.mask))

    def extract(self, roi):
        """What the ROI shows: the shown table's filtered rows inside it."""
        return self.state(roi.file_id).locs[self.indices(roi)]

    # ------------------------------------------------ measuring in an ROI
    def region_context(self, roi, **extra):
        """The context a plugin measuring this ROI is given.

        The ordinary contract, confined to the ROI (`docs/regions.md`):
        ``ctx.locs`` is the ROI's rows of its file, unfiltered, of the table
        the layer shows (the grouped one cut to the ROI where there is one);
        ``ctx.ungrouped()`` the localizations among them, and
        ``ctx.layer_selection(i)`` each layer's filter over those;
        ``ctx.table(i)`` hands out each layer's table cut the same way;
        ``ctx.region`` is the ROI's shape.  Which rows belong is the ROI's shape alone -- a
        filter is applied inside it, never to decide membership -- and no
        drawn ROI or 3D slab narrows it further: the region replaces them.
        """
        region = self.region(roi)
        return self._context(region, self._region_tables(roi, region),
                             region_id=roi.id, **extra)

    def _context(self, region, tables, **extra):
        """A context over what `_region_tables` cut: one per run, since a
        context keeps what it was told, over tables the steps share."""
        from ..plugins import Context
        pinned, rows, layer, session = tables
        locs, selection = pinned[layer]["ungrouped"]
        return Context(session=session, locs=locs, selection=selection, layer=layer,
                       pinned=pinned, rows=rows, region=region, rois=self, **extra)

    def _region_tables(self, roi, region):
        """``(tables, rows, layer, session)`` for `region_context`.

        A project of its own files has one layer: the file, under the
        project's bounds and grouping.  `SessionROIs` has the session's.
        """
        from ..plugins import Selection
        state = self.state(roi.file_id)
        entry = {"shows": "grouped" if state.use_grouped else "ungrouped"}
        rows = None
        sets = {"ungrouped": state.tables.ungrouped}
        if state.tables.is_current:
            sets["grouped"] = state.tables.peek_grouped()
        for kind, locset in sets.items():
            found = rows_in(locset.locs, locset.index, region)
            if kind == "ungrouped":
                rows = found
            entry[kind] = (locset.locs[found],
                           Selection(np.asarray(state.filter_for(locset).mask)[found],
                                     layer=0, roi=region, name=str(region)))
        return {0: entry}, rows, 0, None

    def inputs(self, roi):
        state = self.state(roi.file_id)
        return {"source": self.sources[roi.file_id].fingerprint,
                "geometry": self.geometry(roi), "filters": state.filter.ranges,
                "grouped": self.grouped,
                "group_settings": asdict(self.group_settings) if self.grouped else None}

    # ------------------------------------------------- what is still current
    def resolved(self, steps=None, block=None):
        """The steps that would run: resolved once, for callers that share them.

        ``block`` -- a `chain.Block`, or a list of its steps -- or the
        project's own, or -- when that has none -- the steps of the newest
        run, which is what a script that handed its steps straight to
        `evaluate` leaves behind.  Falling back to the installed plugins
        instead would report steps nobody ran.
        """
        from . import pipeline as pipeline_module
        from ..chain import Block
        if steps is not None:
            return list(steps)
        if block is None:
            block = self.block
        elif not isinstance(block, Block):
            block = Block(steps=list(block))
        if not block.steps:
            for run in reversed(self.runs):
                if run.get("chain"):
                    continue        # a chain's steps are its own, not the window's
                block = pipeline_module.block_from_run(run.get("pipeline") or [])
                if block.steps:
                    break
        if not block.steps:
            block = pipeline_module.default_block()
        return pipeline_module.block_steps(block)

    def block_steps(self):
        """The project's own block's steps, with no fallback: what the ROI
        manager measures the ROI being looked at with."""
        from . import pipeline as pipeline_module
        return pipeline_module.block_steps(self.block)

    def step_signature(self, step, inputs, before=()):
        """What this step's result depends on: the data *and* the parameters.

        The record used to carry one signature over the ROI's inputs alone,
        so editing an evaluator's parameter left every stored number looking
        perfectly current -- the one case where "out of date" matters most.
        ``before`` are the identities of the steps before it that change the
        region's table (`changes_table`): in a block what a step measures is
        what they left, so editing one of them is editing it.
        """
        if not before:
            return digest({"inputs": inputs, "step": step.identity()})
        return digest({"inputs": inputs, "step": step.identity(),
                       "after": list(before)})

    def signatures(self, steps, inputs):
        """`step_signature` for each step, by label, in order."""
        out, before = {}, []
        for step in steps:
            out[step.label] = self.step_signature(step, inputs, before)
            if changes_table(step):
                before.append(step.identity())
        return out

    def entries(self, roi_id, steps=None):
        """The newest stored entry for each step of this ROI, and its state.

        ``{label: (entry or None, state)}`` where state is "current" (same
        data and same parameters), "stale" (something changed) or
        "missing".  Only the steps that would run are reported: a step taken
        out of the pipeline stops contributing columns.
        """
        steps = self.resolved(steps)
        inputs = json.loads(json_text(self.inputs(self.rois[roi_id])))
        wanted = self.signatures(steps, inputs)
        # by signature as well as by label: a step that was renamed, or moved
        # up the pipeline, measured the same thing and its numbers stand.  The
        # label is what the columns are called, and calling them something
        # else is not a reason to run anything again.
        by_signature: Dict[str, Dict] = {}
        by_label: Dict[str, Dict] = {}
        for label, table in self._newest_first():
            entry = table.entry(roi_id)
            if entry is None:
                continue
            by_label[label] = entry
            if entry.get("signature"):
                by_signature.setdefault(entry["signature"], entry)
        found = {}
        for label, signature in wanted.items():
            if signature in by_signature:
                found[label] = (by_signature[signature], "current")
            elif label in by_label:
                found[label] = (by_label[label], "stale")
            else:
                found[label] = (None, "missing")
        return found

    def evaluations(self, roi_id):
        """Every evaluation this ROI has had that still describes it.

        ``{label: (entry, step)}``: under each name, the newest stored result
        that was computed from the ROI's data as it is now (place, file,
        filter, grouping), and the step that made it -- plugin, version,
        parameters -- as its run recorded it.  The name is what makes two
        evaluations different: the same evaluator renamed and run with other
        settings is a second evaluation beside the first, so the two can be
        compared, and running it again under its old name replaces the old
        numbers.  Whether the GUI's pipeline, a chain or a script ran it does
        not matter: a result belongs to the ROI, and says itself how it was
        made.
        """
        stored = str(roi_id)
        if not any(stored in table for table in self.tables.values()):
            return {}          # its data is not cut out for nothing: 60 us an ROI
        inputs = json.loads(json_text(self.inputs(self.rois[roi_id])))
        data = digest(inputs)
        found = {}
        for label, table in self._newest_first():
            entry = table.entry(roi_id)
            if entry is None:
                continue
            step = table.step(roi_id)
            if entry.get("signature") and step:
                identity = {k: v for k, v in step.items() if k != "label"}
                current = entry["signature"] == digest({"inputs": inputs,
                                                        "step": identity})
            else:                         # stored without a step to check against
                current = table.data_signature(roi_id) == data
            if current:
                found[label] = (entry, step)
        return found

    def evaluation_steps(self, label, roi_ids=None):
        """The distinct steps that made the evaluation called ``label`` on these
        ROIs (all included ones by default): one, unless some ROIs were
        evaluated again with other settings and some not."""
        ids = self._included() if roi_ids is None else list(roi_ids)
        out = []
        for roi_id in ids:
            entry, step = self.evaluations(roi_id).get(label, (None, None))
            if step and step not in out:
                out.append(step)
        return out

    def labels(self, plugin=None, roi_ids=None):
        """The names of the current evaluations on these ROIs, newest
        evaluator first, optionally only those made by ``plugin``."""
        ids = self._included() if roi_ids is None else list(roi_ids)
        out = []
        for roi_id in ids:
            for label, (entry, step) in self.evaluations(roi_id).items():
                if label not in out and (plugin is None or step.get("plugin") == plugin):
                    out.append(label)
        return out

    def _included(self):
        return [i for i, roi in self.rois.items() if roi.use]

    def stale_steps(self, roi_id, steps=None):
        """The steps of this ROI that would have to run to be sure."""
        steps = self.resolved(steps)
        states = self.entries(roi_id, steps)
        return [step for step in steps
                if states[step.label][1] != "current"]

    def needs_evaluation(self, steps=None, roi_ids=None):
        """Which included ROIs have a step that is not current."""
        steps = self.resolved(steps)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        return [i for i in ids if self.rois[i].use
                and self.stale_steps(i, steps)]

    # -------------------------------------------------------------- running
    def evaluate(self, block=None, roi_ids=None, progress=None, steps=None,
                 reuse=False, chain="", report=None, stop=None, after=None,
                 tag=False):
        """Run a block's steps over every included ROI.

        ``block`` is a `chain.Block`, or a list of `chain.Step`s, resolved
        here into steps; by default the project's own (`resolved`).  Each step contributes columns to a
        site's row, and a step that raises costs its own columns and not the
        ROI, nor the ROIs after it.

        With `reuse`, a step whose stored result is still current is copied
        forward instead of run again -- which is what "re-evaluate what
        changed" means: editing one evaluator's parameter costs that
        evaluator over the sites, not the whole pipeline over all of them.
        ``chain`` marks the run as a chain's (`chain.evaluate_sites`).
        ``progress(done, total)`` is told after every ROI; ``report(text)``
        gets the status line -- percentage, rate, time left
        (`region_runs.Progress`); ``stop`` (an Event, or a callable such as a
        dialog's ``wasCanceled``) ends the run between ROIs, keeping the ones
        measured.  ``after`` and ``tag`` are a block's (`_evaluate_record`).
        """
        steps = self.resolved(steps, block)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        ids = [i for i in ids if self.rois[i].use]
        from ..region_runs import Progress, stopped
        records = {}
        status = Progress(len(ids), report, progress, what="ROIs")
        for n, roi_id in enumerate(ids):
            if stopped(stop):
                break
            record, _, _ = self._evaluate_record(roi_id, steps, reuse=reuse,
                                                 after=after, tag=tag)
            records[roi_id] = record
            status(n + 1)
        run = self._log(steps, len(records), chain=chain,
                        stopped=len(records) < len(ids))
        self._store(records, steps)
        # the records go back to the caller, who counts what failed; what
        # is kept of them is in `tables`
        return {**run, "records": records}

    def _log(self, steps, regions, **marks):
        """A run in the log: its pipeline, when, how many ROIs, and the
        marks given (``chain``, ``scope``) when they say something."""
        run = {"id": uuid4().hex, "time": datetime.now(timezone.utc).isoformat(),
               "pipeline": [step.as_record() for step in steps], "regions": regions}
        run.update({k: v for k, v in marks.items() if v})
        self.runs.append(run)
        return run

    def _store(self, records, steps):
        """Records into the tables: each step's entry the ROI's row of the
        table named after the step, with the step's record beside it."""
        made = {step.label: step.as_record() for step in steps}
        for roi_id, record in records.items():
            for label, entry in (record.get("steps") or {}).items():
                table = self.tables.pop(label, None) or RegionTable(label)
                table.set(roi_id, entry, made.get(label), record.get("signature", ""))
                self.tables[label] = table          # newest last

    def _newest_first(self):
        return reversed(list(self.tables.items()))

    def _evaluate_record(self, roi_id, steps, reuse=False, force=(), run="stale",
                         after=None, tag=False):
        """One ROI: a complete record, and the results of what actually ran.

        `force` names steps to run even when their stored result is current,
        which is how a figure is got back: the numbers are in the record but
        a plot is a closure over the data it drew, and nothing in a file can
        bring that back -- unless the row kept what its figure is drawn from
        (`Plugin.keep`), which is then restored instead of measuring again.

        `run="forced"` runs nothing else -- a stored result that is out of
        date is carried forward as it is, still signed with what made it, so
        it goes on reading as out of date.  That is the ROI manager with
        re-evaluation turned off: scrolling a list stays free, and what is
        shown is what was measured, labelled for what it is.

        The steps run in one `RegionScratch`: a step that hands back a new
        table hands it to the steps after it.  So once a step has to run,
        every step before it that changes the table runs as well, whatever
        is stored.  ``after(roi_id, scratch, original)`` is told the region's
        tables at the end (a block's write-back); ``tag`` gives them the
        rows they came from.
        """
        from ..region_runs import RegionScratch
        roi = self.rois[roi_id]
        inputs = json.loads(json_text(self.inputs(roi)))
        wanted = self.signatures(steps, inputs)
        known = self.entries(roi_id, steps) if reuse else {}
        plan = []
        for step in steps:
            entry, state = known.get(step.label, (None, "missing"))
            keep = entry is not None and (state == "current" or run == "forced")
            plan.append([step, entry, state, keep])
        last = max((k for k, (step, _, _, keep) in enumerate(plan)
                    if not keep or step.label in force), default=-1)
        for k in range(last):
            if changes_table(plan[k][0]):
                plan[k][3] = False                   # its table is needed
        scratch = None
        record = {"inputs": inputs, "signature": digest(inputs),
                  "file_id": roi.file_id, "steps": {}}
        results = {}
        updated = []
        for step, entry, state, keep in plan:
            if keep and step.label not in force:
                record["steps"][step.label] = dict(entry)
                results[step.label] = None          # kept, so not drawn
                continue
            if keep and entry.get("kept") and not changes_table(step):
                # the figure from what the row kept: no measuring again
                try:
                    restored = step.plugin.restore(entry["kept"])
                except Exception:
                    restored = None
                if restored is not None:
                    record["steps"][step.label] = dict(entry)
                    results[step.label] = restored
                    continue
            if scratch is None:                     # only if something runs
                region = self.region(roi)
                scratch = RegionScratch(self._region_tables(roi, region), region,
                                        roi_id, tag=tag)
                original = scratch.locs
            if state != "current":
                # a step re-run only for its figure tells nobody anything
                # new, and storing it again would grow the file per click
                updated.append(step.label)
            entry, result = self._one_step(step, scratch)
            entry["signature"] = wanted[step.label]
            record["steps"][step.label] = entry
            results[step.label] = result
        if after is not None and scratch is not None:
            after(roi_id, scratch, original)
        return record, results, updated

    def _one_step(self, step, scratch):
        """One evaluator on one ROI, as (what is recorded, what it returned).

        The record is data and outlives the session; the result is the live
        object, with the figures the evaluator drew, and is kept by nobody
        unless the caller wants it.  A failure is recorded, not raised.
        A new table the step hands back becomes the ROI's for the steps after
        it (`RegionScratch.apply`).
        """
        try:
            result = step.plugin.run(scratch.context(
                rois=self, grouping=step.plugin.grouping), step.settings)
            scratch.apply(result, step.label)
            # an ROI is looked at one by one: it keeps what its figures are
            # drawn from (an id column's regions do not, `evaluate_regions`)
            return region_entry(result, step.plugin, keep=True), result
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}, None

    def evaluate_one(self, roi_id, block=None, steps=None, reuse=False,
                     force=(), store=False, run="stale"):
        """The pipeline on a single ROI, figures and all.

        What `evaluate` does per site, for the one site being looked at, and
        handing back each step's whole `Result` rather than only the numbers
        it recorded, and the labels of the steps whose stored result it
        replaced -- which is what lets the ROI manager draw what the
        evaluators drew while the list is walked through.

        `reuse` takes the stored numbers for the steps that are still current
        and runs only the rest, so that a list can be scrolled through
        without recomputing what has not changed; `force` runs a step anyway,
        for the figure of the tab being looked at.  With `store` the record
        joins the runs as a one-site run, which is what makes a re-evaluation
        on selection worth anything: without it the same stale step would be
        re-run on every visit and the site table would never catch up.  A
        step re-run only for its figure has nothing new to say and is not
        stored again.
        """
        steps = self.resolved(steps, block)
        record, results, updated = self._evaluate_record(roi_id, steps, reuse=reuse,
                                                         force=force, run=run)
        if store and updated:
            self._log(steps, 1, scope="site")
            self._store({roi_id: record}, steps)
        return record, results, updated

    def latest(self, roi_id, steps=None):
        """The newest result for this ROI, step by step, and what it is worth.

        ``(record, states)``: a record assembled from the newest entry of
        each step the pipeline would run, and ``{label: state}`` saying which
        of them is still current.  `None` when the ROI has nothing stored for
        any of those steps.

        Per step rather than per record, because the two things that go out
        of date do not go out of date together: moving an ROI invalidates all
        of its steps, editing one evaluator's parameter invalidates that
        evaluator over every ROI.
        """
        steps = self.resolved(steps)
        if not steps:
            # nothing resolves -- the evaluators that made these numbers are
            # not installed here.  Show what was stored rather than nothing,
            # and say that only the data behind it could be checked.
            data = digest(json.loads(json_text(self.inputs(self.rois[roi_id]))))
            record = {"file_id": self.rois[roi_id].file_id, "steps": {}}
            states = {}
            for label, table in self._newest_first():
                entry = table.entry(roi_id)
                if entry is not None:
                    record["steps"][label] = entry
                    states[label] = ("unverified" if table.data_signature(roi_id) == data
                                     else "stale")
            return (record, states) if states else (None, {})
        found = self.entries(roi_id, steps)
        # and what else has measured this ROI: a chain's evaluators, or a
        # renamed one that is no longer in the pipeline
        for label, (entry, _) in self.evaluations(roi_id).items():
            found.setdefault(label, (entry, "current"))
        if all(entry is None for entry, _ in found.values()):
            return None, {label: state for label, (_, state) in found.items()}
        record = {"file_id": self.rois[roi_id].file_id,
                  "steps": {label: entry for label, (entry, _) in found.items()
                            if entry is not None}}
        return record, {label: state for label, (_, state) in found.items()}

    def results(self, steps=None, labels=None):
        """The site table: a row per included ROI that has results.

        A row holds every evaluation the ROI has that still describes it
        (`evaluations`), whoever ran it -- the evaluation window's pipeline,
        a chain, a script -- the pipeline's first, then the rest by name.  A
        column keeps its plain name while one evaluation writes it, and is
        qualified with the evaluation's name when two do
        (`pipeline.merged_values`), so two settings of one evaluator run
        under two names read side by side.

        With ``labels`` only those evaluations are in it, so a column is
        called as its evaluator calls it: what an analysis of one evaluation
        reads (`plugins.roi.SiteAnalysisPlugin`).

        With ``steps`` the row is instead those steps only, and is left out
        while any of them is out of date or missing: what a pipeline would
        report, for a caller that wants exactly that.
        """
        from . import pipeline as pipeline_module
        if steps is not None:
            return self._pipeline_results(steps)
        order = {}
        try:
            order = {step.label: n for n, step in enumerate(self.resolved())}
        except Exception:
            pass
        rows = []
        for roi_id in self._included():
            found = self.evaluations(roi_id)
            names = sorted((l for l in found if labels is None or l in labels),
                           key=lambda l: (order.get(l, len(order)), l))
            values = pipeline_module.merged_values(
                {"steps": {label: found[label][0] for label in names}})
            if values:
                roi = self.rois[roi_id]
                rows.append({"roi_id": roi.number, "file_id": roi.file_id, **values})
        return rows

    def _pipeline_results(self, steps):
        from . import pipeline as pipeline_module
        steps = self.resolved(steps)
        rows = []
        for roi in self.rois.values():
            if not roi.use:
                continue
            record, states = self.latest(roi.id, steps)
            if record is None or any(states.get(step.label) in ("stale", "missing", None)
                                     for step in steps):
                continue
            record = {**record, "steps": {step.label: record["steps"][step.label]
                                          for step in steps}}
            values = pipeline_module.merged_values(record)
            if values:
                rows.append({"roi_id": roi.number, "file_id": roi.file_id, **values})
        return rows

    def find(self, file_id, plugin=None, settings=None, parameters=None):
        """Add the ROIs a segmentation plugin finds on one file.

        Each is included as the finder judged it (``use``); `evaluate` and
        `results` take every included ROI, as the GUI does.
        """
        origin, proposals = self.propose(file_id, plugin, settings, parameters)
        found = []
        for proposed in proposals:
            roi = self.add_roi(file_id, proposed["center"],
                               origin={**origin, **proposed["origin"]})
            roi.use = proposed["use"]
            found.append(roi)
        return found

    def propose(self, file_id, plugin=None, settings=None, parameters=None,
                exclude=()):
        """``(origin, proposals)``: what `find` would add, adding nothing.

        Each proposal is ``{"center", "use", "origin"}``, the last what the
        finder measured, to go into the ROI's origin beside the provenance.
        The ROIs in `exclude` are left out of the ones the finder keeps its
        distance from -- the ones a run would remove first, so that a
        preview of a replacing run proposes what the run will add.
        """
        from ..plugins import settings_from, settings_values
        from ..plugins.roi import DensityPeaks
        if plugin is None:
            plugin = DensityPeaks()
        if settings is None:
            settings = settings_from(plugin.Settings, parameters or {})
        state = self.state(file_id)
        origin = {"method": getattr(plugin, "path", "") or getattr(plugin, "name", ""),
                  "version": str(getattr(plugin, "version", "1")),
                  "parameters": settings_values(settings),
                  "filters": state.filter.ranges, "grouped": self.grouped,
                  "group_settings": asdict(self.group_settings) if self.grouped else None}
        exclude = set(exclude)
        existing = [roi.center for roi in self.rois.values()
                    if roi.file_id == file_id and roi.id not in exclude]
        proposals = []
        for proposed in plugin.propose(state.locs[state.filter.indices], settings,
                                       existing):
            # a centre, or a dict for a finder that also judges what it found:
            # its "use", and what it measured, kept with the ROI's origin
            if isinstance(proposed, dict):
                proposals.append({"center": point(proposed["center"]),
                                  "use": bool(proposed.get("use", True)),
                                  "origin": dict(proposed.get("origin", {}))})
            else:
                proposals.append({"center": point(proposed), "use": True, "origin": {}})
        return origin, proposals

    def absorb_run(self, run):
        """A run as an older file kept it -- its records inside -- into the
        log and the tables; a run without records is only logged."""
        records = run.get("records") or {}
        header = {k: v for k, v in run.items() if k != "records"}
        header.setdefault("regions", len(records))
        self.runs.append(header)
        made = {s.get("label"): s for s in run.get("pipeline") or []}
        for roi_id, record in records.items():
            for label, entry in (record.get("steps") or {}).items():
                table = self.tables.pop(label, None) or RegionTable(label)
                table.set(roi_id, entry, made.get(label), record.get("signature", ""))
                self.tables[label] = table

    def tables_dict(self, roi_ids=None):
        """The tables as plain data, keeping only the rows of ``roi_ids``
        (all when None): what the project's JSON carries."""
        out = []
        for table in self.tables.values():
            doc = table.to_dict(roi_ids)
            if doc["ids"]:
                out.append(doc)
        return out

    def save(self, path):
        """Atomic sidecar save. In-memory sources must first be saved as localizations."""
        path = Path(path).resolve()
        for source in self.sources.values():
            if source.path is None:
                raise ValueError(f"Save localization source {source.name!r} before saving the project")
            if path == Path(source.path):
                raise ValueError("The project must not overwrite a localization source")
        doc = {"size_nm": self.size_nm, "shape": self.shape, "tile_nm": self.tile_nm,
               "filters": self.filters,
               "grouped": self.grouped, "group_settings": asdict(self.group_settings),
               "navigation": self.navigation, "rois": [asdict(r) for r in self.rois.values()],
               "next_number": self.next_number,
               "runs": self.runs, "block": self.block.to_dict(),
               "sources": [{"id": s.id, "name": s.name,
                            "path": os.path.relpath(s.path, path.parent),
                            "fingerprint": s.fingerprint} for s in self.sources.values()]}
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            with h5py.File(temporary, 'w') as f:
                f.attrs['format'] = 'smappy-roi-project'
                f.attrs['format_version'] = 1
                f.create_dataset('project', data=json_text(doc), dtype=h5py.string_dtype('utf-8'))
                tables = f.create_group('tables')
                n = 0
                for table in self.tables.values():
                    table.keep_only(self.rois)
                    if len(table):                    # by number: a label may hold "/"
                        table.to_hdf5(tables.create_group(str(n)))
                        n += 1
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @classmethod
    def load(cls, path, source_paths=None):
        """Reload sources and verify contents; source_paths can relink moved files by ID."""
        path = Path(path).resolve()
        with h5py.File(path, 'r') as f:
            if f.attrs.get('format') != 'smappy-roi-project' or f.attrs.get('format_version') != 1:
                raise ValueError("Unsupported ROI project format")
            doc = json.loads(f['project'][()])
            tables = [RegionTable.from_hdf5(f['tables'][k])
                      for k in sorted(f.get('tables', {}), key=int)]
        project = cls()
        project.set_geometry(doc['size_nm'], doc['shape'])
        project.set_tiles(doc.get('tile_nm', 0.0))
        project.set_filters(doc['filters'])
        project.grouped = doc['grouped']
        project.group_settings = GroupSettings(**doc['group_settings'])
        for saved in doc['sources']:
            source_path = (source_paths or {}).get(saved['id'], path.parent / saved['path'])
            source = project.add_source(load_localizations(source_path), path=source_path,
                                        name=saved['name'], source_id=saved['id'])
            if source.fingerprint != saved['fingerprint']:
                raise ValueError(f"Source contents changed: {source_path}. Restore the original source.")
        for saved in doc['rois']:
            roi = ROI.from_dict(saved)
            if roi.file_id not in project.sources or roi.id in project.rois:
                raise ValueError("Invalid ROI source or duplicate ROI ID")
            project.rois[roi.id] = roi
        project.number_rois(doc.get("next_number"))
        project.runs = []
        for run in doc['runs']:
            project.absorb_run(run)
        for table in tables:
            project.tables[table.label] = table
        project.navigation = doc['navigation']
        from .link import block_from
        project.block = block_from(doc)
        return project
