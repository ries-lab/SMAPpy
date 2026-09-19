"""File-associated analysis ROIs, independent of the interactive interface.

All geometry is in nm. Source tables are immutable snapshots; selection always
uses the same ViewState filter as rendering, never the visible field of view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional
from uuid import uuid4

import h5py
import numpy as np

from ..group import GroupSettings
from ..io.hdf5 import load_localizations
from ..locs import Localizations, to_nm
from ..viewer import DEFAULT_BOUNDS, ViewState


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


def inside_polygon(x, y, vertices):
    """Even-odd fill with boundary points included; no GUI dependency."""
    v = np.asarray(vertices, dtype=float)
    inside = np.zeros(len(x), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        dx, dy = b - a
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        cross = (x - a[0]) * dy - (y - a[1]) * dx
        dot = (x - a[0]) * dx + (y - a[1]) * dy
        boundary |= ((np.abs(cross) <= 1e-7 * length) &
                     (dot >= -1e-7 * length) & (dot <= length**2 + 1e-7 * length))
        if dy != 0:
            inside ^= (((a[1] > y) != (b[1] > y)) &
                       (x < a[0] + (y - a[1]) * dx / dy))
    return inside | boundary


@dataclass
class ROI:
    file_id: str
    center: list
    id: str = field(default_factory=lambda: uuid4().hex)
    polygon: Optional[list] = None
    direction: Optional[list] = None  # two absolute xy endpoints
    reviewed: bool = False
    use: bool = True
    comment: str = ""
    origin: dict = field(default_factory=lambda: {"method": "manual"})


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
        self.group_key = None


class ROIProject:
    def __init__(self):
        self.sources: Dict[str, Source] = {}
        self.rois: Dict[str, ROI] = {}
        self.size_nm = 300.0  # circle diameter or square side length
        self.shape = "circle"
        self.tile_nm = 0.0    # a grid over each file for a systematic walk; 0 = none
        self.filters = dict(DEFAULT_BOUNDS)
        self.grouped = False
        self.group_settings = GroupSettings()
        self.runs = []
        self.navigation = {}  # GUI state only; never used for extraction
        # The evaluation pipeline, as `workspace.Instance`s.  It travels with
        # the project because the columns of a site table mean nothing without
        # knowing which evaluators produced them, with what parameters.
        self.pipeline = []
        self._filter_keys = {}

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
        source = self.sources[file_id]
        state = source.state
        group_key = digest(asdict(self.group_settings))
        if self.grouped and source.group_key != group_key:
            state.sets.pop("grouped", None)
            state.group(self.group_settings)
            source.group_key = group_key
            self._filter_keys.pop((file_id, True), None)
        state.use_grouped = self.grouped
        effective = {k: v for k, v in self.filters.items()
                     if k in state.locs and k not in ("filenumber", "file_id")}
        key = (file_id, self.grouped)
        signature = digest(effective)
        if self._filter_keys.get(key) != signature:
            state.filter.clear()
            for k, bounds in effective.items():
                state.filter.set(k, *bounds)
            self._filter_keys[key] = signature
        return state

    def add_roi(self, file_id, center, polygon=None, reviewed=True, origin=None):
        if file_id not in self.sources:
            raise KeyError(file_id)
        roi = ROI(file_id, point(center), polygon=(None if polygon is None else
                  polygon_vertices(polygon)), reviewed=reviewed)
        if origin is not None:
            roi.origin = json.loads(json_text(origin))
        self.rois[roi.id] = roi
        return roi

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

    def indices(self, roi):
        state = self.state(roi.file_id)
        if roi.polygon is None:
            half = self.size_nm / 2
            x0, y0 = np.asarray(roi.center) - half
            x1, y1 = np.asarray(roi.center) + half
        else:
            vertices = np.asarray(roi.polygon)
            x0, y0 = vertices.min(axis=0)
            x1, y1 = vertices.max(axis=0)
        candidates = state.index.query(x0, x1, y0, y1)
        candidates = candidates[state.filter.mask[candidates]]
        x = np.asarray(state.locs['x_nm'][candidates], dtype=float)
        y = np.asarray(state.locs['y_nm'][candidates], dtype=float)
        if roi.polygon is not None:
            keep = inside_polygon(x, y, roi.polygon)
        elif self.shape == "circle":
            keep = (x - roi.center[0])**2 + (y - roi.center[1])**2 <= half**2
        else:
            keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return np.sort(candidates[keep])

    def extract(self, roi):
        return self.state(roi.file_id).locs[self.indices(roi)]

    def inputs(self, roi):
        state = self.state(roi.file_id)
        return {"source": self.sources[roi.file_id].fingerprint,
                "geometry": self.geometry(roi), "filters": state.filter.ranges,
                "grouped": self.grouped,
                "group_settings": asdict(self.group_settings) if self.grouped else None}

    # ------------------------------------------------- what is still current
    def resolved(self, steps=None, instances=None):
        """The steps that would run: resolved once, for callers that share them.

        The project's pipeline, or -- when it has none -- the pipeline of the
        newest run, which is what a script that handed its steps straight to
        `evaluate` leaves behind, and what an older file carries.  Falling
        back to the installed evaluators instead would report a pipeline
        nobody ran.
        """
        from . import pipeline as pipeline_module
        if steps is not None:
            return list(steps)
        if instances is None:
            instances = self.pipeline
        if not instances:
            for run in reversed(self.runs):
                instances = pipeline_module.instances_from_run(run.get("pipeline") or [])
                if instances:
                    break
        if not instances:
            instances = pipeline_module.default_instances()
        return pipeline_module.resolve(instances)

    def step_signature(self, step, inputs):
        """What this step's result depends on: the data *and* the parameters.

        The record used to carry one signature over the ROI's inputs alone,
        so editing an evaluator's parameter left every stored number looking
        perfectly current -- the one case where "out of date" matters most.
        """
        return digest({"inputs": inputs, "step": step.identity()})

    def entries(self, roi_id, steps=None):
        """The newest stored entry for each step of this ROI, and its state.

        ``{label: (entry or None, state)}`` where state is "current" (same
        data and same parameters), "stale" (something changed), "unverified"
        (stored before signatures were per step, so it cannot say) or
        "missing".  Only the steps that would run are reported: a step taken
        out of the pipeline stops contributing columns.
        """
        steps = self.resolved(steps)
        inputs = json.loads(json_text(self.inputs(self.rois[roi_id])))
        wanted = {step.label: self.step_signature(step, inputs) for step in steps}
        # by signature as well as by label: a step that was renamed, or moved
        # up the pipeline, measured the same thing and its numbers stand.  The
        # label is what the columns are called, and calling them something
        # else is not a reason to run anything again.
        by_signature: Dict[str, Dict] = {}
        by_label: Dict[str, Dict] = {}
        for run in reversed(self.runs):
            record = (run.get("records") or {}).get(roi_id)
            for label, entry in ((record or {}).get("steps") or {}).items():
                by_label.setdefault(label, entry)
                if entry.get("signature"):
                    by_signature.setdefault(entry["signature"], entry)
        found = {}
        for label, signature in wanted.items():
            if signature in by_signature:
                found[label] = (by_signature[signature], "current")
            elif label in by_label:
                entry = by_label[label]
                found[label] = (entry, "stale" if entry.get("signature")
                                else "unverified")
            else:
                found[label] = (None, "missing")
        return found

    def stale_steps(self, roi_id, steps=None):
        """The steps of this ROI that would have to run to be sure."""
        steps = self.resolved(steps)
        states = self.entries(roi_id, steps)
        return [step for step in steps
                if states[step.label][1] != "current"]

    def needs_evaluation(self, steps=None, roi_ids=None):
        """Which reviewed, included ROIs have a step that is not current."""
        steps = self.resolved(steps)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        return [i for i in ids if self.rois[i].reviewed and self.rois[i].use
                and self.stale_steps(i, steps)]

    # -------------------------------------------------------------- running
    def evaluate(self, instances=None, roi_ids=None, progress=None, steps=None,
                 reuse=False):
        """Run a pipeline over every reviewed, included ROI.

        `instances` is a list of `workspace.Instance` -- the same type a tab
        pins -- resolved here into steps.  Each step contributes columns to a
        site's row, and a step that raises costs its own columns and not the
        ROI, nor the ROIs after it.

        With `reuse`, a step whose stored result is still current is copied
        forward instead of run again -- which is what "re-evaluate what
        changed" means: editing one evaluator's parameter costs that
        evaluator over the sites, not the whole pipeline over all of them.
        """
        steps = self.resolved(steps, instances)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        ids = [i for i in ids if self.rois[i].reviewed and self.rois[i].use]
        run = {"id": uuid4().hex, "time": datetime.now(timezone.utc).isoformat(),
               "pipeline": [step.as_record() for step in steps], "records": {}}
        for n, roi_id in enumerate(ids):
            record, _, _ = self._evaluate_record(roi_id, steps, reuse=reuse)
            run["records"][roi_id] = record
            if progress:
                progress(n + 1, len(ids))
        self.runs.append(run)
        return run

    def _evaluate_record(self, roi_id, steps, reuse=False, force=(), run="stale"):
        """One ROI: a complete record, and the results of what actually ran.

        `force` names steps to run even when their stored result is current,
        which is how a figure is got back: the numbers are in the record but
        a plot is a closure over the data it drew, and nothing in a file can
        bring that back.

        `run="forced"` runs nothing else -- a stored result that is out of
        date is carried forward as it is, still signed with what made it, so
        it goes on reading as out of date.  That is the ROI manager with
        re-evaluation turned off: scrolling a list stays free, and what is
        shown is what was measured, labelled for what it is.
        """
        roi = self.rois[roi_id]
        inputs = json.loads(json_text(self.inputs(roi)))
        geometry = json.loads(json_text(self.geometry(roi)))
        known = self.entries(roi_id, steps) if reuse else {}
        locs = None
        record = {"inputs": inputs, "signature": digest(inputs),
                  "file_id": roi.file_id, "steps": {}}
        results = {}
        updated = []
        for step in steps:
            entry, state = known.get(step.label, (None, "missing"))
            keep = entry is not None and (state == "current" or run == "forced")
            if keep and step.label not in force:
                record["steps"][step.label] = dict(entry)
                results[step.label] = None          # kept, so not drawn
                continue
            if locs is None:                        # only if something runs
                locs = self.extract(roi)
            if state != "current":
                # a step re-run only for its figure tells nobody anything
                # new, and storing it again would grow the file per click
                updated.append(step.label)
            entry, result = self._one_step(step, roi, locs, geometry)
            entry["signature"] = self.step_signature(step, inputs)
            record["steps"][step.label] = entry
            results[step.label] = result
        return record, results, updated

    def _one_step(self, step, roi, locs, geometry):
        """One evaluator on one ROI, as (what is recorded, what it returned).

        The record is data and outlives the session; the result is the live
        object, with the figures the evaluator drew, and is kept by nobody
        unless the caller wants it.  A failure is recorded, not raised.
        """
        from ..plugins import Context
        try:
            context = Context(locs=locs, site=geometry, rois=self)
            result = step.plugin.run(context, step.settings)
            return {"values": json.loads(json_text(result.data or {}))}, result
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}, None

    def evaluate_one(self, roi_id, instances=None, steps=None, reuse=False,
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
        steps = self.resolved(steps, instances)
        record, results, updated = self._evaluate_record(roi_id, steps, reuse=reuse,
                                                         force=force, run=run)
        if store and updated:
            self.runs.append({"id": uuid4().hex, "scope": "site",
                              "time": datetime.now(timezone.utc).isoformat(),
                              "pipeline": [step.as_record() for step in steps],
                              "records": {roi_id: record}})
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
            for run in reversed(self.runs):
                record = (run.get("records") or {}).get(roi_id)
                if record is None:
                    continue
                inputs = self.inputs(self.rois[roi_id])
                state = ("unverified" if record.get("signature") == digest(inputs)
                         else "stale")
                return record, {label: state
                                for label in (record.get("steps") or {})}
            return None, {}
        found = self.entries(roi_id, steps)
        if all(entry is None for entry, _ in found.values()):
            return None, {label: state for label, (_, state) in found.items()}
        record = {"file_id": self.rois[roi_id].file_id,
                  "steps": {label: entry for label, (entry, _) in found.items()
                            if entry is not None}}
        return record, {label: state for label, (_, state) in found.items()}

    def results(self, steps=None):
        """Current successful rows for reviewed, included ROIs only.

        A row is left out while any of its steps is out of date or missing:
        half a row of this pipeline's numbers and half of the last one's is
        worse than no row, and `needs_evaluation` says how many are waiting.
        A step that cannot be checked -- stored before signatures were per
        step, or produced by an evaluator that is not installed here -- is
        trusted and reported: it was true when it was written, and dropping
        it would lose an older file's results on opening it.
        """
        from . import pipeline as pipeline_module
        steps = self.resolved(steps)
        rows = []
        for roi in self.rois.values():
            if not roi.reviewed or not roi.use:
                continue
            record, states = self.latest(roi.id, steps)
            if record is None or any(state in ("stale", "missing")
                                     for state in states.values()):
                continue
            values = pipeline_module.merged_values(record)
            if values:
                rows.append({"roi_id": roi.id, "file_id": roi.file_id, **values})
        return rows

    def find(self, file_id, plugin=None, settings=None, parameters=None):
        """Propose candidate ROIs on one file with a segmentation plugin."""
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
        existing = [roi.center for roi in self.rois.values() if roi.file_id == file_id]
        centers = plugin.propose(state.locs[state.filter.indices], settings, existing)
        return [self.add_roi(file_id, c, reviewed=False, origin=origin) for c in centers]

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
               "runs": self.runs, "pipeline": [asdict(i) for i in self.pipeline],
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
            roi = ROI(**saved)
            point(roi.center)
            if roi.polygon is not None:
                polygon_vertices(roi.polygon)
            if roi.file_id not in project.sources or roi.id in project.rois:
                raise ValueError("Invalid ROI source or duplicate ROI ID")
            project.rois[roi.id] = roi
        project.runs = doc['runs']
        project.navigation = doc['navigation']
        from .pipeline import from_dict as pipeline_from_dict
        project.pipeline = pipeline_from_dict(doc)
        return project
