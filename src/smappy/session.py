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
from .group import GROUP_COLUMNS, GroupSettings
from .images import ImageData, load_image
from .locs import Localizations, concat
from .plugins import Context, Plugin, Result, Selection
from .io.formats import FileInfo, load as load_any
from .regions import Region
from .view3d import Projection, Slab
from .render import DisplaySettings, RenderAxes, RenderSettings, SigmaSettings
from .undo import Edit, configured as undo_stack
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
DEFAULT_GROUP_SETTINGS = GroupSettings(dx=50.0, dt=1)


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

    def render(self, fov, white_background=None):
        """RGB in [0, 1] and the intensity plane on ``fov``, whatever the kind.

        ``white_background`` overrides the display's own: what adds several
        layers up turns the sum over once rather than every layer.
        """
        if self.is_image:
            rendered = self.image.resample(fov)
            return self.display.apply(rendered, white_background), rendered
        return self.state.image(fov, white_background)

    def bounds(self):
        """(x0, y0, x1, y1) this layer covers, in render coordinates."""
        if self.is_image:
            return self.image.bounds
        return self.state.bounds()

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
            # through the layer, not the state: the grouped table is built
            # fresh here and `show_grouped` is what copies the bounds onto it.
            # Going straight to the state would leave the grouped set -- the
            # one on screen -- unfiltered, so a drift correction or a colour
            # assignment would look like it had thrown the filters away.
            self.show_grouped(True, share)
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

    def _grouped_carries_the_filter(self, grouped) -> bool:
        """Does the grouped set already have this layer's bounds and files?

        `set_bound` writes to every set, so the two filters agreeing is the
        invariant; this asks whether it still holds.  It is asked rather than
        remembered because a grouped set arrives from three directions -- built
        here, inherited from another layer over the same table, or carried
        through a `rebind` -- and only the first of them is a relink.
        """
        wanted = {field: bound
                  for field, bound in self.state.sets["ungrouped"].filter.ranges.items()
                  if field in grouped.locs}
        if dict(grouped.filter.ranges) != wanted:
            return False
        return self.files is None or "files" in grouped.filter

    def show_grouped(self, on: bool, share: Optional["Layer"] = None) -> None:
        """Draw one entry per blink instead of one per frame.  Links on first
        use -- unless ``share``, a layer over the same table, has it already."""
        other = share.state if share is not None and not share.is_image else None
        self.state.show_grouped(on, self.group_settings, share=other)
        grouped = self.state.sets.get("grouped")
        if on and grouped is not None and not self._grouped_carries_the_filter(grouped):
            # A grouped set this layer has not filtered yet.  Inheriting one
            # from another layer brings that layer's *table* but a filter of
            # its own, and that is how the second layer came out of a drift
            # correction unfiltered: nothing was linked here, so the old test
            # (had it just been relinked?) said there was nothing to do.
            for field, (lo, hi) in self.state.sets["ungrouped"].filter.ranges.items():
                self.set_bound(field, lo, hi)
            if self.files is not None:
                self.set_files(self.files)

    def selection(self, index: int = 0) -> Selection:
        """The *ungrouped* localizations this layer's filter keeps.

        Plugins work on the full table, so a grouped layer hands back the
        ungrouped filter, which is what it would apply to it.
        """
        f = self.state.sets["ungrouped"].filter
        return Selection(f.mask, layer=index, name=self.name)


# How many log entries a file carries.  One entry is a plugin path, a line of
# text and its settings -- a few hundred bytes -- and the whole metadata block
# is written as a single HDF5 attribute, so the oldest are dropped rather than
# risking a file that cannot be written.
MAX_HISTORY = 500


def file_history(locs: Localizations) -> List[Dict]:
    """What the file says was done to these localizations, read tolerantly.

    The log is provenance: which plugins changed the table since it was
    loaded, and with which settings.  It is written by another version of the
    program than the one reading it, so an entry that is not a dictionary is
    dropped rather than being allowed to stop a file from opening.
    """
    found = locs.metadata.get("history") or []
    if not isinstance(found, list):
        return []
    return [dict(e) for e in found if isinstance(e, dict)][-MAX_HISTORY:]


def read_and_group(path, group_settings: GroupSettings, append: bool = False,
                   progress: Optional[Callable[[str], None]] = None,
                   group: Optional[bool] = None, reader=None, **reader_args):
    """Read a localization file and link it: ``(locs, info, grouped)``.

    The one implementation, used by the GUI's `LoadTask`, by `Session.load` and
    by the `File/Load` plugins -- all of which need the slow half off whatever
    thread owns the session, and none of which should own a copy of it.

    An appended file is not linked: `Session.add_file` re-links the merged
    table anyway, so linking here would be work done twice.
    """
    if progress:
        progress("loading")
    locs, info = load_any(path, reader=reader, **reader_args)
    if group is None:
        group = GROUPED_BY_DEFAULT
    grouped = None
    if not append and group and len(locs) and "frame" in locs:
        from .group import group as link
        grouped, _ = link(locs, group_settings,
                          progress=(lambda text, _f: progress(f"Grouper: {text}"))
                          if progress else None)
    return locs, info, grouped


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
        # what the tools that ran on this file left behind, by plugin path:
        # enough to draw their figures again, saved with the file and read
        # back when it is opened.  See `Plugin.keep`.
        self.results: Dict[str, Dict] = {}
        # every table this session replaced, newest last; see `smappy.undo`
        self.undo_stack = undo_stack()
        self._live = False                 # the table is being appended to
        self._listeners: List[Callable[[str], None]] = []
        # set by the GUI: what to write into the file's `gui` group.  A session
        # in a script has none, and saves only data.
        self.gui_state_provider: Optional[Callable[[], Dict]] = None
        # False in a chain's scratch copy: the chain keeps one record and one
        # undo step for all its steps, so the copy keeps neither (`scratch`)
        self.recording = True

    # ----------------------------------------------------------- observers
    def on_change(self, callback: Callable[[str], None]) -> None:
        """``callback(what)``: "locs" (new table), "layer" (a filter or display
        setting changed), "layers" (added/removed/visibility), "history" or
        "results" (a tool kept something worth drawing again)."""
        self._listeners.append(callback)

    def changed(self, what: str) -> None:
        for cb in self._listeners:
            cb(what)

    # ---------------------------------------------------------------- data
    def load(self, path, append: bool = False, progress=None, **reader_args) -> FileInfo:
        """Open a localization file of any known format.

        With ``append`` it joins the table as one more file (a ``filenumber``
        column tells them apart, and the layers can pick); otherwise it
        replaces everything.
        """
        locs, info, grouped = read_and_group(path, self.group_settings, append=append,
                                             progress=progress, **reader_args)
        added = self.add_file(locs, info, append=append, grouped=grouped)
        if not append:
            # after `add_file`, which clears them with the rest of the session
            from .io.hdf5 import load_results
            self.results = load_results(path)
            self.changed("results")
        return added

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
            # the file's own log continues rather than starting again: what
            # was done to these localizations before they were saved is the
            # part of the record that cannot be reconstructed from anything
            # else, and clearing it here is what used to lose it -- `save`
            # writes this list back over the file's
            self.history = file_history(locs)
            self.results = {}
            saved = self.locs.metadata.get("roi")
            self.set_roi(Region.from_dict(saved) if saved else None)
            self._rois = None            # this file's own ROIs, read on first use
            self.changed("rois")
        else:
            merged = concat([self.locs, locs])
            merged.metadata = dict(self.locs.metadata)
            # the linking of the first file says nothing about the merged
            # table, so its group columns go rather than being carried over
            # as a plausible-looking wrong answer; the next grouping writes
            # them again (`group.attach`)
            for column in GROUP_COLUMNS:
                merged.columns.pop(column, None)
            self.set_locs(merged, undoable=True, keep_layers=True,
                          label=f"add {info.name}", text=str(info.path))
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
        self.set_locs(locs, undoable=True, keep_layers=True,
                      label=f"remove {removed.name}")
        self.log("remove file", removed.name, changed=True)

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

    def save(self, path=None, gui_state: Optional[bool] = None) -> Path:
        """Write the table, its history, its ROIs and -- optionally -- the GUI.

        `gui_state` defaults to the `save_gui_state_in_files` preference.  What
        gets written is whatever `gui_state_provider` returns, so the session
        stays ignorant of tabs and windows; the GUI sets it.
        """
        from .io.hdf5 import save_gui_state, save_localizations, save_results
        path = Path(path or self.path)
        metadata = dict(self.locs.metadata)
        # all of it goes in one JSON attribute, so the log is capped rather
        # than allowed to grow without limit over a file's life
        metadata["history"] = self.history[-MAX_HISTORY:]
        if self.roi is not None:
            metadata["roi"] = self.roi.to_dict()
        rois = self.roi_state()
        # a tile grid alone is worth keeping: it is where a systematic walk
        # through the file was left off
        if rois and (rois["rois"] or rois["runs"] or rois.get("tile_nm")):
            metadata["roi_project"] = rois
        save_localizations(path, self.locs, metadata)
        # after the table: both write into the same file, and the results are
        # a convenience where the localizations are the point
        if self.results:
            save_results(path, self.results)
        if gui_state is None:
            from . import config
            gui_state = bool(config.get("save_gui_state_in_files", True))
        if gui_state and self.gui_state_provider is not None:
            try:
                save_gui_state(path, self.gui_state_provider())
            except Exception as e:
                # deliberately everything: the localizations are already on
                # disk by now, and the GUI state is a convenience.  Whatever a
                # provider manages to raise must not turn a good save into a
                # reported failure.
                self.log("save", f"the GUI state was not saved: {e}")
        self.path = path
        return path

    def set_locs(self, locs: Localizations, undoable: bool = True,
                 keep_layers: bool = False,
                 grouped: Optional[Localizations] = None,
                 label: str = "", text: str = "") -> None:
        """Replace the table.  ``label`` is what the undo menu will call this.

        A step is pushed only when ``undoable``; everything else -- a new file,
        the finished form of a live run -- starts the history again, because
        there is nothing left to go back to.
        """
        if self._live:                     # the finished form of the live table
            self.layers = [Layer(locs)]    # (the step before the run is already pushed)
            self._live = False
        elif undoable or keep_layers:      # the same data, corrected: keep the layers
            if undoable:
                self._push_undo(label or "change", text)
            first = None                   # one index for the table, shared by all
            for layer in self.layers:
                if layer.is_image:
                    continue
                layer.rebind(locs, share=first)
                first = first or layer
        else:                              # a new file: start over with one layer
            self.undo_stack.clear()
            self.layers = [l for l in self.layers if l.is_image]
            self.layers.insert(0, Layer(locs, grouped=grouped))
        self.locs = locs
        self.changed("locs")

    @property
    def group_settings(self) -> GroupSettings:
        layer = self._locs_layer()
        return layer.group_settings if layer is not None else DEFAULT_GROUP_SETTINGS

    def set_group_settings(self, settings: GroupSettings) -> None:
        """New linking parameters: every grouped table is rebuilt (once, shared).

        Logged like a plugin run: the grouped table is data the user reads
        numbers off, and which `dx` and `dt` produced it is not recoverable
        from the file afterwards.
        """
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
        self.log("regroup", f"dx = {settings.dx:g}, dt = {settings.dt}",
                 settings=asdict(settings), changed=True)
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

    def copy_layer(self, source: int, target: int, display: bool = True,
                   bounds: bool = False, files: bool = False,
                   grouping: bool = False) -> None:
        """Make one layer like another, in the parts that are asked for.

        `add_layer` copies everything, because a new layer starts as the old
        one with one thing changed.  This is the other half of that: two
        layers that have drifted apart over a session, and one of them is
        right.  What to copy is a choice rather than everything, since the
        thing that makes two layers two -- the filter, or which file each
        shows -- is usually exactly what must *not* be carried over.
        """
        layers = self.layers
        if not (0 <= source < len(layers) and 0 <= target < len(layers)) \
                or source == target:
            return
        src, dst = layers[source], layers[target]
        if src.is_image or dst.is_image:
            raise ValueError("an image layer has no localization settings to copy")
        if display:
            dst.state.settings = dataclasses.replace(src.state.settings)
            dst.set_display(dataclasses.replace(src.get_display()))
        if bounds:
            for field in list(dst.state.sets["ungrouped"].filter.ranges):
                dst.remove_bound(field)
            for field, (lo, hi) in src.state.sets["ungrouped"].filter.ranges.items():
                dst.set_bound(field, lo, hi)
        if files:
            dst.set_files(src.files)
        if grouping:
            dst.group_settings = src.group_settings
            if dst.grouped != src.grouped:
                dst.show_grouped(src.grouped, src)
        self.changed("layer")

    def layer_configs(self) -> List[Dict]:
        """What each localization layer keeps and shows: bounds, grouping,
        files.  The form a chain hands its layers back in, and `Chain/Layers`
        reads the current bounds from."""
        out = []
        for layer in self.layers:
            if layer.is_image:
                continue
            ranges = layer.state.sets["ungrouped"].filter.ranges
            out.append({"grouped": bool(layer.grouped),
                        "bounds": {f: [lo, hi] for f, (lo, hi) in ranges.items()},
                        "files": None if layer.files is None else list(layer.files)})
        return out

    def set_layer_configs(self, configs: Sequence[Dict]) -> None:
        """Make the localization layers say what ``configs`` say.

        A layer that is named and missing is made as a copy of the first (a
        new layer without a template shows no file until one is ticked, which
        is not what a chain that asked for a second layer means).  Layers
        beyond the list are left as they are: the user may have more than the
        chain knows about.  Each layer's bounds are replaced, not merged --
        the config is the whole filter.
        """
        indices = [i for i, l in enumerate(self.layers) if not l.is_image]
        for n, config in enumerate(configs):
            if n >= len(indices):
                self.add_layer(like=indices[0] if indices else None)
                indices = [i for i, l in enumerate(self.layers) if not l.is_image]
            index = indices[n]
            layer = self.layers[index]
            if "bounds" in config:
                for name in list(layer.state.sets["ungrouped"].filter.ranges):
                    layer.remove_bound(name)
                for name, (lo, hi) in (config.get("bounds") or {}).items():
                    if name in self.locs:
                        layer.set_bound(name, lo, hi)
            if "files" in config:
                layer.set_files(config["files"])
            if "grouped" in config and bool(config["grouped"]) != layer.grouped:
                self.show_grouped(index, bool(config["grouped"]))
        self.changed("layer")

    def scratch(self) -> "Session":
        """A headless copy to run a chain on: same table, layers, filters,
        grouping, ROI and slab; no undo, no log, no listeners.

        The arrays are shared, not copied -- a plugin hands back a new table
        and never edits the one it was given -- and so is every grouped table
        that is still current, so a copy costs neither memory nor a relink.
        What the chain does to the copy reaches this session only as the
        chain's result.
        """
        copy = Session()
        copy.recording = False
        copy.locs = self.locs
        copy.path = self.path
        copy.files = list(self.files)
        copy.history = list(self.history)
        copy.roi, copy.slab = self.roi, self.slab
        copy.select_in_slab = self.select_in_slab
        copy.projection = self.projection
        copy.layers = []
        first = None
        for layer in self.layers:
            if layer.is_image:
                copy.layers.append(layer)
                continue
            sets = layer.state.sets
            grouped = (sets["grouped"].locs if "grouped" in sets
                       and not layer.state.grouped_stale else None)
            made = Layer(self.locs, name=layer.name, defaults=False,
                         settings=dataclasses.replace(layer.state.settings),
                         display=dataclasses.replace(layer.get_display()),
                         group_settings=layer.group_settings,
                         grouped=grouped if first is None else None, share=first)
            for name, (lo, hi) in sets["ungrouped"].filter.ranges.items():
                made.set_bound(name, lo, hi)
            made.set_files(layer.files)
            if made.grouped != layer.grouped:
                made.show_grouped(layer.grouped, first)
            copy.layers.append(made)
            first = first or made
        if not any(not l.is_image for l in copy.layers):
            copy.layers.insert(0, Layer(copy.locs))
        return copy

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
        if len(self.locs):
            self._push_undo("live acquisition")
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

    # ------------------------------------------------------------- undo
    def _push_undo(self, label: str, text: str = "") -> None:
        """Remember the table as it is now, under the name of what replaces it."""
        if not self.recording:
            return
        self.undo_stack.push(Edit(label=label, locs=self.locs, text=text))

    def _here(self, label: str, text: str = "") -> Edit:
        """The current state as a step, for the other side of the stack."""
        return Edit(label=label, locs=self.locs, text=text)

    def _restore(self, edit: Edit) -> None:
        """Put a remembered table back, rebinding every layer onto it.

        The layers keep their bounds and their display, and a grouped layer is
        linked again on the way (`Layer.rebind`): coming back from a drift
        correction has to show what it showed before, filters and all.
        """
        self.locs = edit.locs
        first = None
        for layer in self.layers:
            if layer.is_image:
                continue
            layer.rebind(self.locs, share=first)
            first = first or layer
        self.files = [FileInfo(**{k: v for k, v in f.items() if k != "n"})
                      for f in self.locs.metadata.get("files", [])] or self.files

    @property
    def can_undo(self) -> bool:
        return self.undo_stack.can_undo

    @property
    def can_redo(self) -> bool:
        return self.undo_stack.can_redo

    def undo_entries(self) -> List[Edit]:
        """The steps that can be undone, most recent first."""
        return self.undo_stack.entries()

    def redo_entries(self) -> List[Edit]:
        return self.undo_stack.redo_entries()

    def undo_labels(self) -> List[str]:
        return self.undo_stack.labels()

    def redo_labels(self) -> List[str]:
        return self.undo_stack.redo_labels()

    def undo(self, steps: int = 1) -> None:
        """Go back ``steps`` table changes, the most recent first.

        More than one at a time because the menu lists them: picking the third
        entry means the last three, as it does in every other program.  The log
        keeps its entries and gains one saying what was undone -- it is the
        provenance of the localizations, not a second copy of this stack.
        """
        self._travel(steps, "undo")

    def redo(self, steps: int = 1) -> None:
        """Take back an undo, one step or several."""
        self._travel(steps, "redo")

    def _travel(self, steps: int, direction: str) -> None:
        """Walk the stack, restoring each step on the way.

        One method for both directions: they differ in which side of the stack
        is read and which is handed the state being left behind, and nothing
        else.  The log gets a single entry naming every step that was passed,
        rather than one per step, because it is one thing the user did.
        """
        back = direction == "undo"
        done = []
        for _ in range(max(1, int(steps))):
            entries = self.undo_entries() if back else self.redo_entries()
            if not entries:
                break
            here = self._here(entries[0].label, entries[0].text)
            edit = (self.undo_stack.undo(redo_of=here) if back
                    else self.undo_stack.redo(undo_of=here))
            if edit is None:
                break
            self._restore(edit)
            done.append(entries[0].label)
        if done:
            self.log(direction, ", ".join(done), changed=True)
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
        """The depth the first locs layer's filter keeps, else the data's.

        In render units, like the slab it goes into: the third axis is ``z_nm``
        unless the render axes say otherwise, and a filter bound is in the
        column's own units and has to be scaled with it.
        """
        layer = self._locs_layer()
        if layer is None or not len(self.locs):
            return (-1.0, 1.0)
        axes = self.axes()
        name = axes.depth_name(self.locs)
        if name is None:
            return (-1.0, 1.0)
        lo, hi = layer.state.sets["ungrouped"].filter.ranges.get(name, (None, None))
        z = np.asarray(axes.depth(self.locs))
        finite = z[np.isfinite(z)]
        dlo, dhi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        return (dlo if lo is None else lo / axes.z_scale,
                dhi if hi is None else hi / axes.z_scale)

    def slab_from_roi(self, use_roi: bool = True) -> Slab:
        """The slab from the ROI (or the whole field), z from the filter."""
        z0, z1 = self.z_range()
        if use_roi and self.roi is not None:
            self.slab = Slab.from_region(self.roi, (z0, z1))
        else:
            (x0, x1), (y0, y1) = self.full_view(0.0)
            self.slab = Slab.from_bounds(x0, x1, y0, y1, z0, z1)
        self.changed("slab")
        return self.slab

    def roi_edited(self, roi: Region) -> None:
        """The same ROI, dragged or resized on screen.

        Not `set_roi`: the item under the mouse is the truth and redrawing it
        from here would fight the drag.  The slab still follows it, which is
        what made a 3D view of a line ROI stale until `from ROI` was pressed --
        including the width, which is the one thing a line ROI is dragged for.
        """
        self.roi = roi
        if self.slab_follows_roi:
            self.slab_from_roi()
        self.changed("roi-edited")

    def set_slab(self, slab: Slab, follow_roi: bool = False) -> None:
        self.slab = slab
        self.slab_follows_roi = follow_roi
        self.changed("slab")

    def axes(self, layer: int = 0) -> RenderAxes:
        """Which columns the picture's axes are.

        One set for the whole session in practice -- the layers are composited
        into one grid, so two layers on different axes would mean nothing --
        but it is kept on each layer's `RenderSettings`, as
        `DisplaySettings.white_background` is, so that a render outside a
        session needs nothing else.
        """
        if not self.layers or self.layers[layer].is_image:
            layer = self.first_locs_layer()
        state = self.layers[layer].state if self.layers else None
        return state.settings.axes if state is not None else RenderAxes()

    def set_axes(self, axes: RenderAxes) -> None:
        """Put every localization layer on these axes, and say so.

        The slab is a box in *render* units, so it belongs to the axes it was
        built on: kept across a change it is a box in the wrong quantity, and
        a 3D window showing a ROI-sized box of nanometres against a picture of
        photons against frame is empty -- and stays empty on the way back,
        which is what made it look as though nothing could be selected.  So a
        real change rebuilds it, from the ROI when there is one.
        """
        before = self.axes()
        for layer in self.layers:
            if not layer.is_image:
                layer.state.settings = dataclasses.replace(layer.state.settings, axes=axes)
        if axes != before and len(self.locs):
            # not from the ROI: that was drawn on the old picture and is in its
            # coordinates too, so the whole field on the new axes is the only
            # box that means anything until the user draws another one
            self.slab_from_roi(use_roi=False)
        self.changed("layer")

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
        return self._clip(self.layers[layer].selection(layer), self.locs, layer)

    def _clip(self, sel: Selection, locs: Localizations, layer: int) -> Selection:
        """Narrow a layer's selection to the ROI and, if asked, the slab."""
        axes = self.axes(layer)
        if self.roi is not None and len(locs):
            # an ROI is drawn on the picture, so it is in render coordinates:
            # a rectangle on a photons-against-frame view selects those
            # localizations, and the usual picture is unchanged
            x, y = axes.coordinates(locs)
            sel.mask = sel.mask & self.roi.mask(x, y)     # never in place: the
            # filter's cached mask is what `Selection` was handed
            sel.roi = self.roi
            sel.name += f", {self.roi}"
        if self.select_in_slab and self.slab is not None and len(locs):
            x, y = axes.coordinates(locs)
            z = axes.depth(locs)
            sel.mask = sel.mask & self.slab.mask(x, y, z)
            sel.roi = self.slab
            sel.name += f", {self.slab}"
        return sel

    def table(self, layer: int = 0, grouping: Optional[str] = None
              ) -> Tuple[Localizations, Selection]:
        """A layer's table and selection, grouped or not: ``(locs, selection)``.

        ``grouping`` is "grouped", "ungrouped", or None for whatever the layer
        itself shows.  This is where the choice a chain or a plugin makes
        (`Context.table`) meets the session: the ungrouped answer is
        `self.locs` with `selection`, and the grouped one is the layer's
        linked table with its own filter, cut to the same ROI and slab.

        Asking for the grouped table of a layer that has none links it.  The
        rule elsewhere is to say what is missing rather than spend minutes
        behind the user's back, but here somebody has asked for exactly this
        table -- a chain step set to grouped -- and linking is the answer.
        What the layer *shows* is left as it was.
        """
        if self.layers[layer].is_image:
            layer = self.first_locs_layer()
        lay = self.layers[layer]
        grouped = lay.grouped if grouping is None else grouping == "grouped"
        if not grouped:
            return self.locs, self.selection(layer)
        if "grouped" not in lay.state.sets or lay.state.grouped_stale:
            showing = lay.state.use_grouped
            partner = next((l for l in self.layers if l is not lay and not l.is_image
                            and "grouped" in l.state.sets
                            and not l.state.grouped_stale), None)
            lay.show_grouped(True, partner)
            lay.state.use_grouped = showing
        locset = lay.state.sets["grouped"]
        sel = Selection(locset.filter.mask, layer=layer, name=f"{lay.name}, grouped")
        return locset.locs, self._clip(sel, locset.locs, layer)

    def context(self, layer: int = 0,
                progress: Optional[Callable[[str], None]] = None,
                stream: Optional[Callable[[str, Any], None]] = None,
                **extra) -> Context:
        """What a plugin run against this session is given.

        Built on the caller's thread: the table and the selection are read
        here, so handing the context to a worker cannot race a live fit
        rebinding them.
        """
        return Context(session=self, layer=layer, progress=progress,
                       stream=stream, **extra)

    def run(self, plugin: Plugin, settings=None, layer: int = 0,
            progress: Optional[Callable[[str], None]] = None,
            stream: Optional[Callable[[str, Any], None]] = None) -> Result:
        """Run a plugin on this session's table; apply what comes back.

        The plugin runs on whatever thread calls this; only `apply` touches
        the session, so a GUI can run the plugin in a worker and apply here.
        """
        result = plugin.run(self.context(layer, progress, stream,
                                         grouping=plugin.grouping), settings)
        self.apply(plugin, result)
        return result

    def apply(self, plugin: Plugin, result: Result) -> None:
        settings = result.settings
        # The log is the provenance of the localizations, so what goes in it
        # is what changed them -- a measurement can be repeated from the file
        # and would only record that somebody looked.  `Plugin.logged`
        # overrides the rule either way; see it for when that is right.
        changed = result.locs is not None or bool(result.files)
        for n, (locs, info, grouped) in enumerate(result.files or ()):
            # the first replaces unless the plugin asked to append; the rest
            # always join, or opening three files would keep only the last
            self.add_file(locs, info, append=result.data.get("append", False) or n > 0,
                          grouped=grouped)
        if result.locs is not None and not self.files and not len(self.locs) \
                and not self._live:
            # A table made from nothing -- a fit run where no file is open,
            # which is every fit in a batch -- is a file being opened, not a
            # correction: it gets its layers and its file entry, and its own
            # log, rather than an undo step back to an empty session.
            from .io.formats import FileInfo
            where = (result.data or {}).get("path")
            name = Path(where).name if where else plugin.path.rsplit("/", 1)[-1]
            self.add_file(result.locs, FileInfo(name=name, path=str(where or ""),
                                                format=plugin.path, n=len(result.locs)))
            if not where:
                self.path = None         # nowhere to save back to until asked
        elif result.locs is not None:
            # the menu says what it is about to undo, so the step is named
            # after the plugin rather than after the fact that a table changed
            self.set_locs(result.locs, label=plugin.path.rsplit("/", 1)[-1],
                          text=result.text or "")
        # After the files: opening one starts the log again from the file's
        # own, and an entry written before it would be the one thing lost --
        # which is how a loader's settings never reached the history.
        if plugin.logged if plugin.logged is not None else changed:
            self.log(plugin.path, result.text,
                     settings=asdict(settings) if is_dataclass(settings) else settings,
                     changed=changed, **(result.log or {}))
        # A plugin that writes a column the user is meant to *filter* on says
        # so here, as `{field: (lo, hi)}`.  It cannot set the bound itself: a
        # filter belongs to the table it was built from, and the table the
        # column is in is the one that has just been set here.
        bounds = (result.data or {}).get("bounds") or {}
        for field, (lo, hi) in bounds.items():
            for layer in self.layers:
                if not layer.is_image:
                    layer.set_bound(field, lo, hi)
        if bounds:
            self.changed("locs")
        # the layers a chain step set up -- bounds and grouping per layer, the
        # Render tab's work done by a plugin (`Chain/Layers`)
        if (result.data or {}).get("layers"):
            self.set_layer_configs(result.data["layers"])
        # last: a plugin that opened a file has just cleared the session, this
        # one included, and what it worked out belongs to the file it opened
        self.remember(plugin, result)

    def remember(self, plugin: Plugin, result: Result) -> None:
        """Keep what this tool worked out, so its figure can be drawn again.

        What is kept is the plugin's own business -- most keep nothing -- and
        it goes into the localization file, so that a drift curve is still
        there to look at when the corrected file is opened again.  A plugin
        that raises here has run successfully and is not going to be told
        otherwise over a figure, so the failure is logged and dropped.
        """
        try:
            saved = plugin.keep(result)
        except Exception as error:
            self.log(plugin.path, f"its result was not kept: {error}")
            return
        if not saved:
            return
        # the settings go with it: a figure a week later has to say what it
        # was made with, and for a measurement this is the only record there
        # is -- its run is not in the log, on purpose (see `apply`)
        self.results[plugin.path] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "text": result.text, "data": saved,
            "settings": (asdict(result.settings) if is_dataclass(result.settings)
                         else result.settings)}
        self.changed("results")

    def restore_result(self, plugin: Plugin) -> Optional[Result]:
        """The result this tool left in the file, drawable again, or None."""
        saved = self.results.get(plugin.path)
        if not saved:
            return None
        try:
            return plugin.restore(saved.get("data") or {})
        except Exception:
            return None

    def log(self, what: str, text: str = "", **extra) -> None:
        if not self.recording:
            return
        self.history.append({"time": datetime.now().isoformat(timespec="seconds"),
                             "what": what, "text": text, **extra})
        self.changed("history")
