"""The ROI project, backed by the session instead of by files of its own.

`ROIProject` normally owns immutable snapshots of localization files.  In the
GUI the session already holds the tables, the filters and the grouping, and a
plugin may replace the table (a drift correction, a fit).  `SessionROIs` keeps
the project's model -- ROIs, review flags, finder provenance, evaluation runs --
and takes the data from the session: one source per loaded file, filters and
grouping from a layer, so what an ROI sees is what the image shows.

The project's state travels in the localization file's metadata, so ROIs and
their results are saved with `File -> Save` and come back with the file.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from ..group import GroupSettings
from ..locs import Localizations
from ..viewer import ViewState
from .core import ROI, ROIProject, digest, json_text, point, polygon_vertices

FORMAT_VERSION = 1


class SessionSource:
    """What the project needs of a source; the data stays in the session."""

    def __init__(self, source_id: str, name: str, path: Optional[str], number: int):
        self.id = source_id
        self.name = name
        self.path = path
        self.number = number          # the table's `filenumber`
        self.state: Optional[ViewState] = None
        self.fingerprint = ""
        self.group_key = None


class SessionROIs(ROIProject):
    """An `ROIProject` whose sources are the session's files.

    ``layer`` says which layer's filter and grouping the ROIs follow; the
    layer's *file* choice is deliberately ignored, since an ROI already knows
    its file.
    """

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.layer = 0
        # review is not exposed in the GUI: an ROI counts as reviewed when it
        # is made, and `use` is what includes or excludes it
        self.auto_review = True
        self.preview_nm = 0.0        # the ROI image's width; 0 = three ROI widths
        self._states: Dict[str, ViewState] = {}
        self._built = None            # what the sources were built from

    # ------------------------------------------------------------- sources
    def sync(self) -> None:
        """Follow the session: files, filters, grouping.  Cheap when nothing moved."""
        locs = self.session.locs
        names = self.session.file_names() or (["(unsaved)"] if len(locs) else [])
        signature = (id(locs), len(locs), tuple(names))
        if self._built != signature:
            self._rebuild(names)
            self._built = signature
        layer = self.session.layers[min(self.layer, len(self.session.layers) - 1)]
        if layer.is_image:
            layer = self.session.layers[self.session.first_locs_layer()]
        ranges = {f: b for f, b in layer.state.sets["ungrouped"].filter.ranges.items()}
        self.filters = ranges
        self.grouped = layer.grouped
        self.group_settings = layer.group_settings
        self.render = layer.state.settings          # so the manager's images
        self.display = layer.state.display          # look like the main view
        for source in self.sources.values():         # the states follow the layer
            if source.state is not None:
                self._apply(source)

    def _rebuild(self, names: List[str]) -> None:
        """Sources for the session's files, keeping the ids ROIs refer to."""
        by_number = {s.number: s for s in self.sources.values()}
        self.sources = {}
        self._states = {}
        for number, name in enumerate(names):
            info = self.session.files[number] if number < len(self.session.files) else None
            old = by_number.get(number)
            source = SessionSource(old.id if old else f"file{number}", name,
                                   info.path if info else None, number)
            source.fingerprint = self._fingerprint(source)
            self.sources[source.id] = source
        # ROIs of files that are gone stay in the project but cannot be used
        self._states.clear()

    def _fingerprint(self, source: SessionSource) -> str:
        """What an evaluation was run on: the data, not the file's name.

        Row count, column names and a sample of the positions -- enough to
        notice a corrected or refitted table, cheap enough to recompute on
        every staleness check, and unchanged by a rename or a "save as".
        """
        locs = self.session.locs
        if not len(locs):
            return digest([0, sorted(locs.columns)])
        rows = np.arange(len(locs))
        if "filenumber" in locs and len(self.sources) > 1:
            rows = rows[np.asarray(locs["filenumber"]) == source.number]
        sample = rows[:: max(1, len(rows) // 512)][:512]
        values = [np.asarray(locs[c])[sample] for c in ("x_nm", "y_nm") if c in locs]
        return digest([len(rows), sorted(locs.columns),
                       [np.round(v.astype(float), 3).tolist() for v in values]])

    def _slice(self, source: SessionSource) -> Localizations:
        locs = self.session.locs
        if "filenumber" not in locs or len(self.sources) <= 1:
            return locs
        return locs[np.asarray(locs["filenumber"]) == source.number]

    def _apply(self, source: SessionSource) -> None:
        """Put the layer's bounds, grouping and look on this source's state."""
        state = source.state
        if state is None:
            return
        settings, display = getattr(self, "render", None), getattr(self, "display", None)
        if settings is not None:
            state.settings, state.display = settings, display
        key = digest(asdict(self.group_settings))
        if self.grouped and (source.group_key != key or "grouped" not in state.sets):
            state.sets.pop("grouped", None)
            state.group(self.group_settings)
            source.group_key = key
        state.use_grouped = self.grouped and "grouped" in state.sets
        for locset in state.sets.values():
            locset.filter.clear()
            for field, (lo, hi) in self.filters.items():
                if field in locset.locs:
                    locset.filter.set(field, lo, hi)

    def state(self, file_id) -> ViewState:
        """The data an ROI of this file sees: the session's rows for that file,
        under the layer's filter and grouping."""
        source = self.sources[file_id]
        if source.state is None:
            source.state = ViewState(self._slice(source))
            self._apply(source)
        return source.state

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        """The project as JSON-compatible data, for the localization file."""
        return {"format_version": FORMAT_VERSION,
                "size_nm": self.size_nm, "shape": self.shape,
                "preview_nm": self.preview_nm, "tile_nm": self.tile_nm,
                "navigation": self.navigation,
                "sources": [{"id": s.id, "number": s.number, "name": s.name}
                            for s in self.sources.values()],
                "rois": [asdict(r) for r in self.rois.values()],
                "runs": self.runs,
                # the pipeline is provenance for the columns in `runs`, so it
                # is saved with them rather than only in the workspace
                "pipeline": [asdict(i) for i in self.pipeline]}

    def from_dict(self, doc: Optional[dict]) -> None:
        """Restore ROIs, runs and navigation saved with a file."""
        self.rois = {}
        self.runs = []
        self.navigation = {}
        self.pipeline = []
        if not doc:
            return
        doc = json.loads(json_text(doc)) if not isinstance(doc, dict) else doc
        if doc.get("format_version", FORMAT_VERSION) > FORMAT_VERSION:
            raise ValueError("this file's ROIs were written by a newer smappy")
        self.set_geometry(doc.get("size_nm", self.size_nm), doc.get("shape", self.shape))
        self.preview_nm = float(doc.get("preview_nm", 0.0) or 0.0)
        self.set_tiles(doc.get("tile_nm", 0.0) or 0.0)
        self.navigation = doc.get("navigation", {})
        # ids the file used, mapped onto the files this session has
        by_number = {s.number: s for s in self.sources.values()}
        remap = {}
        for saved in doc.get("sources", []):
            source = by_number.get(saved.get("number"))
            if source is not None:
                remap[saved["id"]] = source.id     # the name stays the session's
        for saved in doc.get("rois", []):
            roi = ROI(**saved)
            point(roi.center)
            if roi.polygon is not None:
                polygon_vertices(roi.polygon)
            roi.file_id = remap.get(roi.file_id, roi.file_id)
            if roi.file_id in self.sources:
                self.rois[roi.id] = roi
        for run in doc.get("runs", []):
            records = {i: r for i, r in run.get("records", {}).items() if i in self.rois}
            if records:
                self.runs.append({**run, "records": records})
        from .pipeline import from_dict as pipeline_from_dict
        self.pipeline = pipeline_from_dict(doc)

    # ---------------------------------------------------------------- misc
    def add_roi(self, file_id, center, polygon=None, reviewed=None, origin=None):
        if reviewed is None:
            reviewed = self.auto_review
        return super().add_roi(file_id, center, polygon,
                               reviewed or self.auto_review, origin)

    @property
    def preview_width(self) -> float:
        """What the ROI image covers, in nm."""
        return self.preview_nm or 3.0 * self.size_nm

    def rois_of(self, file_id) -> List[ROI]:
        return [r for r in self.rois.values() if r.file_id == file_id]

    def numbers(self) -> Dict[str, int]:
        """A 1-based number per ROI, in the order they were made."""
        return {roi_id: i + 1 for i, roi_id in enumerate(self.rois)}

    def file_number(self, file_id) -> int:
        source = self.sources.get(file_id)
        return source.number + 1 if source is not None else 0

    def source_of(self, number: int) -> Optional[SessionSource]:
        return next((s for s in self.sources.values() if s.number == number), None)
