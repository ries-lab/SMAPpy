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
from dataclasses import asdict, replace
from typing import Dict, List, Optional

import numpy as np

from ..group import GROUP_COLUMNS, GroupSettings
from ..locs import Localizations
from ..render import RenderAxes
from ..layer_state import Tables, ViewState
from ..region_table import RegionTable
from .core import ROI, ROIProject, digest, json_text, point, polygon_vertices, rows_in

FORMAT_VERSION = 1

# Columns the session writes onto the table itself, rather than the data
# bringing them: `Session.load` adds `filenumber`, and grouping adds the group
# columns in place.  Whether they are there yet says nothing about the
# localizations, so the fingerprint counts them as always present -- present
# rather than absent so that the fingerprints already saved, which were taken
# from loaded, usually grouped tables, still match.
BOOKKEEPING = ("filenumber",) + GROUP_COLUMNS


class SessionSource:
    """What the project needs of a source; the data stays in the session."""

    def __init__(self, source_id: str, name: str, path: Optional[str], number: int):
        self.id = source_id
        self.name = name
        self.path = path
        self.number = number          # the table's `filenumber`
        self.state: Optional[ViewState] = None
        self.fingerprint = ""


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
        # (id and length of a table, file number) -> that file's rows of it:
        # computed once, not once per ROI measured
        self._file_masks: Dict[tuple, np.ndarray] = {}

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
        self.filters = layer.ranges
        self.grouped = layer.grouped
        # so the manager's images look like the main view -- except for the
        # axes, which stay the positions: a site is a place, and a versatile
        # render is a picture of two columns rather than of anywhere
        self.render = replace(layer.state.settings, axes=RenderAxes())
        self.display = layer.state.display
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
        every staleness check, and unchanged by a rename or a "save as" --
        or by a save and reopen, which adds `filenumber` to a table that
        never came from a file (see `BOOKKEEPING`).  The columns written from
        regions' results (`region_runs.REGION_COLUMNS`) are not counted either:
        an ROI's number or its measured value written onto its localizations
        changes nothing it was measured on, and used to make the very
        evaluation it came from out of date.
        """
        from ..region_runs import REGION_COLUMNS
        locs = self.session.locs
        written = set(locs.metadata.get(REGION_COLUMNS) or {})
        names = sorted((set(locs.columns) - written) | set(BOOKKEEPING))
        if not len(locs):
            return digest([0, names])
        rows = np.arange(len(locs))
        if "filenumber" in locs and len(self.sources) > 1:
            rows = rows[np.asarray(locs["filenumber"]) == source.number]
        sample = rows[:: max(1, len(rows) // 512)][:512]
        values = [np.asarray(locs[c])[sample] for c in ("x_nm", "y_nm") if c in locs]
        return digest([len(rows), names,
                       [np.round(v.astype(float), 3).tolist() for v in values]])

    def _slice(self, source: SessionSource) -> Localizations:
        locs = self.session.locs
        if "filenumber" not in locs or len(self.sources) <= 1:
            return locs
        return locs[np.asarray(locs["filenumber"]) == source.number]

    @property
    def group_settings(self) -> GroupSettings:
        """The session's: there is one grouped table, and the ROIs see it."""
        return self.session.group_settings

    @group_settings.setter
    def group_settings(self, settings: GroupSettings) -> None:
        # `ROIProject.__init__` sets a default; the session's stands
        pass

    def _apply(self, source: SessionSource) -> None:
        """Put the layer's bounds, grouping and look on this source's state."""
        state = source.state
        if state is None:
            return
        settings, display = getattr(self, "render", None), getattr(self, "display", None)
        if settings is not None:
            state.settings, state.display = settings, display
        state.tables.settings = self.session.group_settings
        state.use_grouped = self.grouped
        state.set_ranges(self.filters)

    def _tables(self, source: SessionSource) -> Tables:
        """The tables an ROI of this file looks at.  One file: the session's
        own, grouped table and all.  One of several: its rows of the session's
        table, and of the session's grouped table -- linking never joins two
        files (`GroupSettings.block_fields`), so those rows are what linking
        the file alone would give, without linking anything again."""
        session = self.session
        if "filenumber" not in session.locs or len(self.sources) <= 1:
            return session.tables

        def link(settings, attach_columns, number=source.number):
            grouped = session.tables.grouped().locs
            if "filenumber" not in grouped:
                return grouped
            return grouped[np.asarray(grouped["filenumber"]) == number]
        return Tables(self._slice(source), session.group_settings, link=link)

    def state(self, file_id) -> ViewState:
        """The data an ROI of this file sees: the session's rows for that file,
        under the layer's filter and grouping."""
        source = self.sources[file_id]
        if source.state is None:
            source.state = ViewState(tables=self._tables(source))
            self._apply(source)
        return source.state

    # ------------------------------------------------ measuring in an ROI
    def _file_mask(self, locs: Localizations, number: int) -> Optional[np.ndarray]:
        """Which rows of ``locs`` are file ``number``'s; None when all are."""
        if "filenumber" not in locs or len(self.sources) <= 1:
            return None
        key = (id(locs), len(locs), number)
        if key not in self._file_masks:
            if len(self._file_masks) > 16:
                self._file_masks.clear()
            self._file_masks[key] = np.asarray(locs["filenumber"]) == number
        return self._file_masks[key]

    def _region_tables(self, roi, region):
        """Every localization layer of the session, cut to the ROI.

        The rows are found through the file's own index (the ROI belongs to
        one file) and are rows of the session's table; each layer's
        selection is its own filter over them -- not the session's
        selection, which is cut to the drawn ROI and the slab -- and its
        grouped table is offered, cut the same way, where it has a current
        one.  The ROI manager's own state for the file decides nothing here
        but where to look.
        """
        from ..plugins import Selection
        session = self.session
        source = self.sources[roi.file_id]
        ungrouped = self.state(roi.file_id).tables.ungrouped
        found = rows_in(ungrouped.locs, ungrouped.index, region)
        in_file = self._file_mask(session.locs, source.number)
        rows = found if in_file is None else np.flatnonzero(in_file)[found]
        locs = session.locs[rows]
        tables = {}
        for i, lay in enumerate(session.layers):
            if lay.is_image:
                continue
            mask = np.asarray(lay.selection(i).mask, dtype=bool)
            entry = {"ungrouped": (locs, Selection(mask[rows], layer=i, roi=region,
                                                   name=f"{lay.name}, {region}"))}
            locset = session.tables.peek_grouped()
            if locset is not None and session.tables.is_current:
                keep = self._file_mask(locset.locs, source.number)
                g = rows_in(locset.locs, locset.index, region, keep)
                entry["grouped"] = (locset.locs[g], Selection(
                    np.asarray(lay.state.filter_for(locset).mask, dtype=bool)[g],
                    layer=i, roi=region, name=f"{lay.name}, blinks merged, {region}"))
            tables[i] = entry
        layer = min(self.layer, len(session.layers) - 1)
        if session.layers[layer].is_image:
            layer = session.first_locs_layer()
        return tables, rows, layer, session

    def inputs(self, roi):
        """What a measurement of this ROI depends on: the ROI's own inputs and
        every layer's filters and grouping -- a plugin in an ROI may read any
        layer (`_region_tables`), so a filter changed on any of them makes
        its numbers out of date."""
        out = super().inputs(roi)
        from ..region_runs import _layer_inputs
        out["layers"] = [_layer_inputs(lay) for lay in self.session.layers
                         if not lay.is_image]
        return out

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
                "next_number": self.next_number,
                "runs": self.runs,
                # the results, a columnar table per evaluation (`RegionTable`)
                "tables": self.tables_dict(self.rois),
                # the pipeline is provenance for the columns in `runs`, so it
                # is saved with them rather than only in the workspace
                "pipeline": [asdict(i) for i in self.pipeline],
                "block_options": self.block_options}

    def from_dict(self, doc: Optional[dict]) -> None:
        """Restore ROIs, runs and navigation saved with a file."""
        self.rois = {}
        self.next_number = 1
        self.runs = []
        self.tables = {}
        self.navigation = {}
        self.pipeline = []
        from .pipeline import DEFAULT_OPTIONS
        self.block_options = dict(DEFAULT_OPTIONS)
        self.segment_preview = None
        self.history.clear()        # another file's edits
        if not doc:
            return
        doc = json.loads(json_text(doc)) if not isinstance(doc, dict) else doc
        if doc.get("format_version", FORMAT_VERSION) > FORMAT_VERSION:
            raise ValueError("this file's ROIs were written by a newer smappy")
        self.set_geometry(doc.get("size_nm", self.size_nm), doc.get("shape", self.shape))
        self.preview_nm = float(doc.get("preview_nm", 0.0) or 0.0)
        self.set_tiles(doc.get("tile_nm", 0.0) or 0.0)
        self.navigation = dict(doc.get("navigation", {}))
        # ids the file used, mapped onto the files this session has
        by_number = {s.number: s for s in self.sources.values()}
        remap = {}
        for saved in doc.get("sources", []):
            source = by_number.get(saved.get("number"))
            if source is not None:
                remap[saved["id"]] = source.id     # the name stays the session's
        if self.navigation.get("file") in remap:
            self.navigation["file"] = remap[self.navigation["file"]]
        for saved in doc.get("rois", []):
            roi = ROI(**saved)
            point(roi.center)
            if roi.polygon is not None:
                polygon_vertices(roi.polygon)
            roi.file_id = remap.get(roi.file_id, roi.file_id)
            if roi.file_id in self.sources:
                self.rois[roi.id] = roi
        self.number_rois(doc.get("next_number"))
        for run in doc.get("runs", []):
            records = {i: r for i, r in (run.get("records") or {}).items()
                       if i in self.rois}
            self.absorb_run({**run, "records": records})
        for saved in doc.get("tables", []):
            table = RegionTable.from_dict(saved)
            table.keep_only(self.rois)
            if len(table):
                self.tables[table.label] = table
        from .pipeline import DEFAULT_OPTIONS, from_dict as pipeline_from_dict
        self.pipeline = pipeline_from_dict(doc)
        self.block_options = {**DEFAULT_OPTIONS, **(doc.get("block_options") or {})}

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

    def file_number(self, file_id) -> int:
        source = self.sources.get(file_id)
        return source.number + 1 if source is not None else 0

    def source_of(self, number: int) -> Optional[SessionSource]:
        return next((s for s in self.sources.values() if s.number == number), None)
