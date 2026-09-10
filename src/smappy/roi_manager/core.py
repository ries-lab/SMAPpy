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
        self.filters = dict(DEFAULT_BOUNDS)
        self.grouped = False
        self.group_settings = GroupSettings()
        self.runs = []
        self.navigation = {}  # GUI state only; never used for extraction
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

    def evaluate(self, plugin=None, roi_ids=None, parameters=None, progress=None):
        from .plugins import Statistics
        plugin = plugin or Statistics()
        parameters = parameters or {}
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        ids = [i for i in ids if self.rois[i].reviewed and self.rois[i].use]
        run = {"id": uuid4().hex, "time": datetime.now(timezone.utc).isoformat(),
               "plugin": plugin.name, "version": plugin.version,
               "parameters": json.loads(json_text(parameters)), "records": {}}
        for n, roi_id in enumerate(ids):
            roi = self.rois[roi_id]
            inputs = json.loads(json_text(self.inputs(roi)))
            record = {"inputs": inputs, "signature": digest(inputs),
                      "file_id": roi.file_id}
            try:
                record['values'] = json.loads(json_text(plugin.evaluate(
                    self.extract(roi), json.loads(json_text(self.geometry(roi))),
                    json.loads(json_text(run['parameters'])))))
            except Exception as error:
                record['error'] = f"{type(error).__name__}: {error}"
            run['records'][roi_id] = record
            if progress:
                progress(n + 1, len(ids))
        self.runs.append(run)
        return run

    def latest(self, roi_id, plugin="statistics"):
        for run in reversed(self.runs):
            if run['plugin'] == plugin and roi_id in run['records']:
                record = run['records'][roi_id]
                return record, record['signature'] != digest(self.inputs(self.rois[roi_id]))
        return None, False

    def results(self, plugin="statistics"):
        """Current successful results for reviewed, included ROIs only."""
        rows = []
        for roi in self.rois.values():
            if not roi.reviewed or not roi.use:
                continue
            record, stale = self.latest(roi.id, plugin)
            if record is not None and not stale and 'values' in record:
                rows.append({"roi_id": roi.id, "file_id": roi.file_id,
                             **record['values']})
        return rows

    def find(self, file_id, plugin=None, parameters=None):
        from .plugins import DensityPeaks
        plugin = plugin or DensityPeaks()
        parameters = {**getattr(plugin, 'defaults', {}), **(parameters or {})}
        state = self.state(file_id)
        origin = {"method": plugin.name, "version": plugin.version,
                  "parameters": parameters, "filters": state.filter.ranges,
                  "grouped": self.grouped,
                  "group_settings": asdict(self.group_settings) if self.grouped else None}
        existing = [roi.center for roi in self.rois.values() if roi.file_id == file_id]
        centers = plugin.find(state.locs[state.filter.indices], parameters, existing)
        return [self.add_roi(file_id, c, reviewed=False, origin=origin) for c in centers]

    def save(self, path):
        """Atomic sidecar save. In-memory sources must first be saved as localizations."""
        path = Path(path).resolve()
        for source in self.sources.values():
            if source.path is None:
                raise ValueError(f"Save localization source {source.name!r} before saving the project")
            if path == Path(source.path):
                raise ValueError("The project must not overwrite a localization source")
        doc = {"size_nm": self.size_nm, "shape": self.shape, "filters": self.filters,
               "grouped": self.grouped, "group_settings": asdict(self.group_settings),
               "navigation": self.navigation, "rois": [asdict(r) for r in self.rois.values()],
               "runs": self.runs,
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
        return project
