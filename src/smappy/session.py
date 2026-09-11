"""The state a GUI or a script works on: one table, its layers, undo, history.

No Qt in here.  The GUI subscribes to `on_change` and redraws; a script uses
it the same way without subscribing to anything.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .filter import LocFilter
from .group import GroupSettings
from .images import ImageData, load_image
from .locs import Localizations, concat
from .plugins import Plugin, Result, Selection
from .io.formats import FileInfo, load as load_any
from .regions import Region
from .view3d import Projection, Slab
from .render import DisplaySettings, RenderSettings, SigmaSettings, positions
from .viewer import ViewState

# The bounds a layer opens with: what is thrown away is on screen, not hidden.
DEFAULT_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "loc_precision_nm": (None, 25.0),
    "logl_rel": (-2.0, None),
    "z_nm": (-500.0, 500.0),
}
# only for a 2D table: a 3D fit's PSF size varies with z by design
DEFAULT_BOUNDS_2D = {"sigma_nm": (None, 180.0)}
PRECISION_FACTOR = 0.5           # rendering sigma = factor * localization precision
GROUPED_BY_DEFAULT = True
# grouping in the GUI: SMAP's 50 nm / 1 frame, xy only.  Two emitters within
# that box in consecutive frames have overlapping PSFs and could not have been
# fitted apart anyway, so a z window would only split real blinks; it stays an
# option in the parameters dialog, with that caveat.
DEFAULT_GROUP_SETTINGS = GroupSettings(dx=50.0, dt=1, dz=None)


class Layer:
    """A filter and how its localizations are drawn -- or a pixel image.

    A ``"locs"`` layer wraps a `ViewState`, which owns the filter, the
    spatial index and the render/display settings -- everything the render
    view needs per layer.  An ``"image"`` layer holds an `ImageData` and only
    the display settings; it is resampled onto the view instead of rendered.
    """

    def __init__(self, locs: Localizations, name: str = "layer 1",
                 defaults: bool = True,
                 settings: Optional[RenderSettings] = None,
                 display: Optional[DisplaySettings] = None,
                 live: bool = False, extent=None, share: Optional["Layer"] = None,
                 group_settings: Optional[GroupSettings] = None,
                 grouped: Optional[Localizations] = None):
        """``share`` is a layer over the same table whose spatial index (and
        grouped table) this one reuses -- a new layer then costs nothing.

        ``grouped`` is that table already linked -- by a worker thread, off the
        GUI's, which is the only way opening a large file does not freeze the
        window.  Given it, the `show_grouped` below finds the set already there
        and links nothing.
        """
        self.kind = "locs"
        self.image: Optional[ImageData] = None
        self.files: Optional[List[int]] = None     # file numbers shown; None = all
        self.name = name
        self.visible = True
        if settings is None:
            settings = RenderSettings(sigma_settings=SigmaSettings(factor=PRECISION_FACTOR))
        other = share.state if share is not None and not share.is_image else None
        self.group_settings = group_settings or DEFAULT_GROUP_SETTINGS
        self.state = ViewState(locs, settings, display, grouped=grouped, live=live,
                               extent=extent, share=other)
        if defaults:
            self.apply_defaults()
        if GROUPED_BY_DEFAULT and not live and len(locs) and "frame" in locs:
            self.show_grouped(True, share)

    @classmethod
    def from_image(cls, image: ImageData, name: Optional[str] = None,
                   display: Optional[DisplaySettings] = None) -> "Layer":
        layer = cls.__new__(cls)
        layer.kind = "image"
        layer.image = image
        layer.name = name or image.name or "image"
        layer.visible = True
        layer.state = None
        layer.display = display or DisplaySettings(lut="gray")
        return layer

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def get_display(self) -> DisplaySettings:
        return self.display if self.is_image else self.state.display

    def set_display(self, display: DisplaySettings) -> None:
        if self.is_image:
            self.display = display
        else:
            self.state.display = display

    def render(self, fov):
        """RGB in [0, 1] and the intensity plane on ``fov``, whatever the kind."""
        if self.is_image:
            rendered = self.image.resample(fov)
            return self.display.apply(rendered), rendered
        return self.state.image(fov)

    def bounds(self):
        """(x0, y0, x1, y1) this layer covers."""
        if self.is_image:
            return self.image.bounds
        return self.state.index.bounds

    def apply_defaults(self) -> None:
        locs = self.locs
        bounds = dict(DEFAULT_BOUNDS)
        if "z_nm" not in locs:
            bounds.update(DEFAULT_BOUNDS_2D)
        for field, (lo, hi) in bounds.items():
            if field in locs:
                self.set_bound(field, lo, hi)

    def set_bound(self, field: str, lo: Optional[float], hi: Optional[float]) -> None:
        """A bound applies to the grouped and the ungrouped table alike, so
        what is drawn and what a plugin gets never disagree."""
        for locset in self.state.sets.values():
            if field in locset.locs:
                locset.filter.set(field, lo, hi)

    def remove_bound(self, field: str) -> None:
        for locset in self.state.sets.values():
            if field in locset.filter:
                locset.filter.remove(field)

    @property
    def locs(self) -> Localizations:
        """The table this layer draws: grouped or not."""
        return self.state.locs

    @property
    def filter(self) -> LocFilter:
        return self.state.filter

    def set_files(self, files: Optional[Sequence[int]]) -> None:
        """Restrict to these file numbers; None means every file."""
        self.files = None if files is None else sorted(int(f) for f in files)
        for locset in self.state.sets.values():
            if files is None:
                if "files" in locset.filter:
                    locset.filter.remove("files")
            elif "filenumber" in locset.locs:
                locset.filter.set_mask("files", np.isin(locset.locs["filenumber"], list(files)))

    def rebind(self, locs: Localizations, share: Optional["Layer"] = None) -> None:
        """Point at a new table, keeping the bounds and display the user set."""
        if self.is_image:
            return
        old = self.state
        other = share.state if share is not None and not share.is_image else None
        self.state = ViewState(locs, old.settings, old.display, share=other)
        if share is not None and not share.is_image:
            self.group_settings = share.group_settings
        for field, (lo, hi) in old.sets["ungrouped"].filter.ranges.items():
            if field in locs:
                self.filter.set(field, lo, hi)
        if old.use_grouped:
            self.state.show_grouped(True, share=other)
        if self.files is not None:
            self.set_files(self.files)

    def append(self, block: Localizations) -> int:
        """Take in a block of a table that is still being produced."""
        first = not len(self.locs)
        n = self.state.append(block)
        if first:
            self.apply_defaults()
        return n

    @property
    def grouped(self) -> bool:
        return self.state.use_grouped

    def show_grouped(self, on: bool, share: Optional["Layer"] = None) -> None:
        """Draw one entry per blink instead of one per frame.  Links on first
        use -- unless ``share``, a layer over the same table, has it already."""
        fresh = on and ("grouped" not in self.state.sets or self.state.grouped_stale)
        other = share.state if share is not None and not share.is_image else None
        self.state.show_grouped(on, self.group_settings, share=other)
        if fresh:                    # the new table gets the bounds already set
            for field, (lo, hi) in self.state.sets["ungrouped"].filter.ranges.items():
                self.set_bound(field, lo, hi)

    def selection(self, index: int = 0) -> Selection:
        """The *ungrouped* localizations this layer's filter keeps.

        Plugins work on the full table, so a grouped layer hands back the
        ungrouped filter, which is what it would apply to it.
        """
        f = self.state.sets["ungrouped"].filter
        return Selection(f.mask, layer=index, name=self.name)


class Session:
    def __init__(self, locs: Optional[Localizations] = None, path=None):
        self.locs = locs if locs is not None else Localizations({}, {})
        self.path: Optional[Path] = Path(path) if path else None
        self.layers: List[Layer] = [Layer(self.locs)]
        self.roi: Optional[Region] = None          # the drawn 2D ROI (regions.py)
        self._rois = None                          # the ROI manager's project
        self.slab: Optional[Slab] = None          # the 3D view's volume
        self.slab_follows_roi = True              # the 2D ROI sets its footprint
        self.select_in_slab = False               # plugins see only the slab
        self.projection = Projection()
        self.files: List[FileInfo] = []
        self.history: List[Dict] = []
        self._undo: Optional[Localizations] = None
        self._live = False                 # the table is being appended to
        self._listeners: List[Callable[[str], None]] = []

    # ----------------------------------------------------------- observers
    def on_change(self, callback: Callable[[str], None]) -> None:
        """``callback(what)``: "locs" (new table), "layer" (a filter or display
        setting changed), "layers" (added/removed/visibility) or "history"."""
        self._listeners.append(callback)

    def changed(self, what: str) -> None:
        for cb in self._listeners:
            cb(what)

    # ---------------------------------------------------------------- data
    def load(self, path, append: bool = False, **reader_args) -> FileInfo:
        """Open a localization file of any known format.

        With ``append`` it joins the table as one more file (a ``filenumber``
        column tells them apart, and the layers can pick); otherwise it
        replaces everything.
        """
        locs, info = load_any(path, **reader_args)
        return self.add_file(locs, info, append=append)

    def add_file(self, locs: Localizations, info: FileInfo, append: bool = False,
                 grouped: Optional[Localizations] = None) -> FileInfo:
        """``grouped`` is ``locs`` already linked, for a caller that did the
        slow part in a worker thread.  It is only used for the first file: an
        append re-links the merged table anyway, since linking cannot be
        extended, and a second file's ids would not follow the first's.
        """
        if not append or not len(self.locs):
            self.files = []
            self.path = Path(info.path)
        number = len(self.files)
        self.files.append(info)
        columns = dict(locs.columns)
        columns["filenumber"] = np.full(len(locs), number, np.int32)
        locs = Localizations(columns, dict(locs.metadata))
        if number == 0:
            if grouped is not None:      # the same column, on the linked table
                gc = dict(grouped.columns)
                gc["filenumber"] = np.full(len(grouped), number, np.int32)
                grouped = Localizations(gc, dict(grouped.metadata))
            self.set_locs(locs, undoable=False, grouped=grouped)
            self.history.clear()
            saved = self.locs.metadata.get("roi")
            self.set_roi(Region.from_dict(saved) if saved else None)
            self._rois = None            # this file's own ROIs, read on first use
            self.changed("rois")
        else:
            merged = concat([self.locs, locs])
            merged.metadata = dict(self.locs.metadata)
            self.set_locs(merged, undoable=True, keep_layers=True)
        self.locs.metadata["files"] = [f.to_dict() for f in self.files]
        self.log("load", str(info.path), append=append)
        return info

    def file_names(self) -> List[str]:
        return [f.name for f in self.files]

    def remove_file(self, number: int) -> None:
        """Drop one file's localizations; the others are renumbered densely."""
        if not 0 <= number < len(self.files):
            return
        numbers = np.asarray(self.locs["filenumber"])
        keep = numbers != number
        columns = {k: np.asarray(v)[keep] for k, v in self.locs.columns.items()}
        renumbered = columns["filenumber"].copy()
        renumbered[renumbered > number] -= 1
        columns["filenumber"] = renumbered
        removed = self.files.pop(number)
        locs = Localizations(columns, dict(self.locs.metadata))
        locs.metadata["files"] = [f.to_dict() for f in self.files]
        for layer in self.layers:               # a layer's file choice follows the numbering
            if not layer.is_image and layer.files is not None:
                layer.files = [n - 1 if n > number else n for n in layer.files if n != number]
        self.set_locs(locs, undoable=True, keep_layers=True)
        self.log("remove file", removed.name)

    # -------------------------------------------------------------- images
    def add_image(self, image: ImageData, name: Optional[str] = None) -> Layer:
        layer = Layer.from_image(image, name)
        self.layers.append(layer)
        self.changed("layers")
        return layer

    def open_image(self, path, pixelsize: Optional[float] = None,
                   x0: float = 0.0, y0: float = 0.0) -> Layer:
        return self.add_image(load_image(path, pixelsize, x0, y0))

    def full_view(self, margin_fraction: float = 0.01):
        """The ranges covering every visible layer -- or all, if none is."""
        boxes = [l.bounds() for l in self.layers if l.visible] or \
                [l.bounds() for l in self.layers]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return (0.0, 1.0), (0.0, 1.0)
        b = np.array(boxes)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        mx, my = (x1 - x0) * margin_fraction, (y1 - y0) * margin_fraction
        return (x0 - mx, x1 + mx), (y0 - my, y1 + my)

    def first_locs_layer(self) -> int:
        return next((i for i, l in enumerate(self.layers) if not l.is_image), 0)

    def save(self, path=None) -> Path:
        from .io.hdf5 import save_localizations
        path = Path(path or self.path)
        metadata = dict(self.locs.metadata)
        metadata["history"] = self.history
        if self.roi is not None:
            metadata["roi"] = self.roi.to_dict()
        rois = self.roi_state()
        # a tile grid alone is worth keeping: it is where a systematic walk
        # through the file was left off
        if rois and (rois["rois"] or rois["runs"] or rois.get("tile_nm")):
            metadata["roi_project"] = rois
        save_localizations(path, self.locs, metadata)
        self.path = path
        return path

    def set_locs(self, locs: Localizations, undoable: bool = True,
                 keep_layers: bool = False,
                 grouped: Optional[Localizations] = None) -> None:
        if self._live:                     # the finished form of the live table
            self.layers = [Layer(locs)]    # (undo already points before the run)
            self._live = False
        elif undoable or keep_layers:      # the same data, corrected: keep the layers
            self._undo = self.locs if undoable else None
            first = None                   # one index for the table, shared by all
            for layer in self.layers:
                if layer.is_image:
                    continue
                layer.rebind(locs, share=first)
                first = first or layer
        else:                              # a new file: start over with one layer
            self._undo = None
            self.layers = [l for l in self.layers if l.is_image]
            self.layers.insert(0, Layer(locs, grouped=grouped))
        self.locs = locs
        self.changed("locs")

    @property
    def group_settings(self) -> GroupSettings:
        layer = self._locs_layer()
        return layer.group_settings if layer is not None else DEFAULT_GROUP_SETTINGS

    def set_group_settings(self, settings: GroupSettings) -> None:
        """New linking parameters: every grouped table is rebuilt (once, shared)."""
        first = None
        for layer in self.layers:
            if layer.is_image:
                continue
            layer.group_settings = settings
            layer.state.grouped_stale = True
            was_on = layer.grouped
            if "grouped" in layer.state.sets:
                bounds = dict(layer.state.sets["ungrouped"].filter.ranges)
                layer.show_grouped(True, first)
                layer.state.use_grouped = was_on
                for field, (lo, hi) in bounds.items():
                    layer.set_bound(field, lo, hi)
                if layer.files is not None:
                    layer.set_files(layer.files)
            first = first or layer
        self.changed("regrouped")

    def _locs_layer(self) -> Optional[Layer]:
        return next((l for l in self.layers if not l.is_image), None)

    def show_grouped(self, index: int, on: bool) -> None:
        """Grouped display for one layer, reusing another layer's grouped table."""
        layer = self.layers[index]
        partner = next((l for l in self.layers if l is not layer and not l.is_image
                        and "grouped" in l.state.sets), None)
        layer.show_grouped(on, partner)

    def add_layer(self, like: Optional[int] = None) -> Layer:
        """A new layer on the same table.

        ``like`` is the index of a localization layer to copy: bounds, files,
        render and display settings, grouping.  A second layer is nearly
        always the first one with one thing changed, so copying is the useful
        start; without ``like`` (or when it names an image) the layer gets the
        defaults and shows no file until one is ticked.
        """
        template = None
        if like is not None and 0 <= like < len(self.layers) and not self.layers[like].is_image:
            template = self.layers[like]
        layer = Layer(self.locs, name=f"layer {len(self.layers) + 1}",
                      share=self._locs_layer(),
                      defaults=template is None,
                      settings=(dataclasses.replace(template.state.settings)
                                if template is not None else None),
                      display=(dataclasses.replace(template.get_display())
                               if template is not None else None),
                      group_settings=template.group_settings if template is not None else None)
        if template is None:
            if "filenumber" in self.locs:
                layer.set_files([])        # starts empty: pick the file(s) it shows
        else:
            for field, (lo, hi) in template.state.sets["ungrouped"].filter.ranges.items():
                layer.set_bound(field, lo, hi)
            layer.set_files(template.files)
            if layer.grouped != template.grouped:
                layer.show_grouped(template.grouped, template)
        self.layers.append(layer)
        self.changed("layers")
        return layer

    def remove_layer(self, index: int) -> None:
        if len(self.layers) > 1:
            del self.layers[index]
            self.changed("layers")

    # ------------------------------------------------------------ streaming
    def begin_live(self, extent=None, path=None) -> None:
        """Start an empty table that `append` grows, for a running fit.

        ``extent`` (x0, x1, y0, y1) frames the view before any data arrives.
        The old table is kept for undo.
        """
        self._undo = self.locs if len(self.locs) else None
        self._live = True
        self.locs = Localizations({}, {})
        self.layers = [Layer(self.locs, live=True, extent=extent)]
        if path is not None:
            self.path = Path(path)
        self.changed("locs")

    def append(self, block: Localizations) -> int:
        """Add a block to the live table; every layer takes it in."""
        n = 0
        for layer in self.layers:
            if not layer.is_image:
                n = layer.append(block)
        self.locs = self.layers[self.first_locs_layer()].state.sets["ungrouped"].locs
        if n:
            self.changed("append")
        return n

    @property
    def can_undo(self) -> bool:
        return self._undo is not None

    def undo(self) -> None:
        if self._undo is not None:
            self.locs, self._undo = self._undo, None
            first = None
            for layer in self.layers:
                if layer.is_image:
                    continue
                layer.rebind(self.locs, share=first)
                first = first or layer
            self.files = [FileInfo(**{k: v for k, v in f.items() if k != "n"})
                          for f in self.locs.metadata.get("files", [])] or self.files
            self.log("undo")
            self.changed("locs")

    # -------------------------------------------------------- ROI manager
    @property
    def rois(self):
        """The ROI manager's project over this session's files.

        Separate from `roi`, the rectangle or line drawn in the 2D view: this
        is the collection of analysis ROIs, their review state and their
        evaluation runs, and it is saved with the localization file.
        """
        if self._rois is None:
            from .roi_manager.link import SessionROIs
            self._rois = SessionROIs(self)
            saved = self.locs.metadata.get("roi_project")
            self._rois.sync()
            if saved:
                self._rois.from_dict(saved)
        else:
            self._rois.sync()
        return self._rois

    def roi_state(self) -> Optional[dict]:
        """The ROI manager's project as data, or None if it was never opened."""
        return None if self._rois is None else self._rois.to_dict()

    # ----------------------------------------------------------------- roi
    def set_roi(self, roi: Optional[Region]) -> None:
        self.roi = roi
        if self.slab_follows_roi:
            self.slab_from_roi()
        self.changed("roi")

    # ---------------------------------------------------------------- 3D
    def z_range(self) -> Tuple[float, float]:
        """The z the first locs layer's filter keeps, else the data's."""
        layer = self._locs_layer()
        if layer is None or "z_nm" not in self.locs or not len(self.locs):
            return (-1.0, 1.0)
        lo, hi = layer.state.sets["ungrouped"].filter.ranges.get("z_nm", (None, None))
        z = np.asarray(self.locs["z_nm"])
        finite = z[np.isfinite(z)]
        dlo, dhi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        return (dlo if lo is None else lo, dhi if hi is None else hi)

    def slab_from_roi(self) -> Slab:
        """The slab from the ROI (or the whole field), z from the filter."""
        z0, z1 = self.z_range()
        if self.roi is not None:
            self.slab = Slab.from_region(self.roi, (z0, z1))
        else:
            (x0, x1), (y0, y1) = self.full_view(0.0)
            self.slab = Slab.from_bounds(x0, x1, y0, y1, z0, z1)
        self.changed("slab")
        return self.slab

    def set_slab(self, slab: Slab, follow_roi: bool = False) -> None:
        self.slab = slab
        self.slab_follows_roi = follow_roi
        self.changed("slab")

    def set_projection(self, projection: Projection) -> None:
        self.projection = projection
        self.changed("projection")

    # ------------------------------------------------------------- plugins
    def selection(self, layer: int = 0) -> Selection:
        """What a plugin looks at: the layer's filter, inside the ROI if any.

        An image layer has no localizations; the first locs layer stands in.
        """
        if self.layers[layer].is_image:
            layer = self.first_locs_layer()
        sel = self.layers[layer].selection(layer)
        if self.roi is not None and len(self.locs):
            x, y = positions(self.locs)
            sel.mask = sel.mask & self.roi.mask(x, y)     # never in place: the
            # filter's cached mask is what `Selection` was handed
            sel.roi = self.roi
            sel.name += f", {self.roi}"
        if self.select_in_slab and self.slab is not None and len(self.locs):
            x, y = positions(self.locs)
            z = self.locs["z_nm"] if "z_nm" in self.locs else None
            sel.mask = sel.mask & self.slab.mask(x, y, z)
            sel.roi = self.slab
            sel.name += f", {self.slab}"
        return sel

    def run(self, plugin: Plugin, settings=None, layer: int = 0,
            progress: Optional[Callable[[str], None]] = None) -> Result:
        """Run a plugin on this session's table; apply what comes back.

        The plugin runs on whatever thread calls this; only `apply` touches
        the session, so a GUI can run the plugin in a worker and apply here.
        """
        result = plugin.run(self.locs, self.selection(layer), settings, progress)
        self.apply(plugin, result)
        return result

    def apply(self, plugin: Plugin, result: Result) -> None:
        settings = result.settings
        self.log(plugin.path, result.text,
                 settings=asdict(settings) if is_dataclass(settings) else settings)
        if result.locs is not None:
            self.set_locs(result.locs)

    def log(self, what: str, text: str = "", **extra) -> None:
        self.history.append({"time": datetime.now().isoformat(timespec="seconds"),
                             "what": what, "text": text, **extra})
        self.changed("history")
