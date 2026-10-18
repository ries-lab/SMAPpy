"""The File tab: getting localizations in, out, and made up.

Loading, saving, exporting and simulating were menu items and a script, which
meant they had no settings form, no place in the tree and nothing saved between
sessions.  They are ordinary plugins now.

A loader runs in the panel's worker thread and must not touch the session, so
it hands its work back as `Result.files` and `Session.apply` adds them on the
thread that owns the session.  The reading itself is `session.read_localizations`,
the same function the window's own `LoadTask` uses -- one implementation, so a
format behaves the same whichever way it was opened.

There is one loader, `File/Load/Auto`, with an advanced *format* choice for
the file whose name does not say what it is.  There used to be a plugin per
format as well, which put five entries in the tree for one thing and a csv
setting where only one of them read it; the csv *columns* is now greyed for
anything but a csv (`describe`).  Adding a format is a one-line
`register(Reader(...))` in `io.formats`, and it is a choice here at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..io.formats import (FileInfo, name_filter, reader_for, reader_named,
                          save_filter, writer_for)
from ..simulate.settings import SimulationSettings
from . import Context, FormState, Notice, Plugin, Result, param, register

MSR = "Abberior MSR"          # the reader's name (`io.formats`)
LOCS_FILTER = "Localizations (*.hdf5 *.h5 *.mat *.csv *.npy *.zip *.json *.msr)"


# --------------------------------------------------------------------- load

def _format_choices():
    from ..io.formats import reader_names
    return [("", "auto: by the file name")] + [(name, name) for name in reader_names()]


def _conversion_choices():
    from ..io.conversions import conversions
    return [("", "auto: by the headers")] + [(n, n) for n in conversions()]


@dataclass
class CsvSettings:
    """How a delimited text file's columns are read; `io.formats._load_csv`."""
    conversion: str = param("", label="conversion", choices=_conversion_choices,
                            help="a saved conversion; auto picks the one whose "
                                 "headers the file has")
    mapping: str = param("", label="columns", kind="text",
                         help="header=column pairs, e.g. X=x_nm, Y=y_nm, T=frame; "
                              "empty means the conversion's, or a guess from the header")
    units: str = param("", label="lengths in",
                       choices=(("", "auto"), ("nm", "nm"), ("um", "µm"),
                                ("mm", "mm"), ("px", "camera pixels")),
                       help="what x, y, z and their errors are in")
    pixelsize_nm: Optional[float] = param(None, label="camera pixel", unit="nm",
                                          min=0.01, help="for lengths in pixels")
    first_frame: Optional[int] = param(None, label="first frame",
                                       help="the number the file's first frame has: "
                                            "1 for ThunderSTORM and SMAP, 0 for smappy")


@dataclass
class MsrSettings:
    """What of an Abberior `.msr` is read; `io.msr`.  The GUI's File > Open
    asks in a dialog instead (`gui.dialogs.MsrImportDialog`)."""
    measurements: str = param("", label="measurements",
                              help="empty: all; numbers from 1, labels or a "
                                   "pattern of the labels, separated by commas")
    merge: bool = param(False, label="merge into one file",
                        help="one table for all of them, one after another in "
                             "time, with a measurement column; otherwise a "
                             "file each")
    keep_invalid: bool = param(False, label="keep invalid",
                               help="every final iteration, with vld and sta, "
                                    "not only the valid ones")
    z_scale: float = param(1.0, label="z scale", min=0.01, max=10.0,
                           help="multiplies z: the refractive index correction; "
                                "1 keeps Imspector's numbers")
    images: bool = param(True, label="images",
                         help="read the file's confocal and STED images, placed "
                              "in nm, for an image layer to show")


@dataclass
class LoadSettings:
    path: str = param("", label="file", kind="open_file", file_filter=LOCS_FILTER,
                      help="the localization file to read")
    append: bool = param(False, label="add to the open files",
                         help="join the table as one more file instead of "
                              "replacing everything; Add in File > Open")
    format: str = param("", label="format", choices=_format_choices, advanced=True,
                        help="auto: from the file's name and, where that is not "
                             "enough, its contents")
    csv: CsvSettings = param(default_factory=CsvSettings, label="csv columns",
                             collapsed=True)
    msr: MsrSettings = param(default_factory=MsrSettings, label="Abberior MSR",
                             collapsed=True)


def _reader(path: Path, settings: LoadSettings):
    """The reader ``settings`` ask for: the one named, or the file's own."""
    if settings.format:
        reader = reader_named(settings.format)
        if reader is None:
            raise ValueError(f"no reader called {settings.format!r}")
        return reader
    return reader_for(path)          # raises with the known formats listed


def csv_args(csv: CsvSettings) -> Dict[str, Any]:
    """The csv reader's arguments from the form's csv part."""
    args: Dict[str, Any] = {}
    if csv.mapping.strip():
        pairs = [p.split("=", 1) for p in csv.mapping.split(",") if "=" in p]
        args["mapping"] = {a.strip(): b.strip() for a, b in pairs}
    if csv.conversion:
        args["conversion"] = csv.conversion
    if csv.units:
        args["units"] = csv.units
    if csv.pixelsize_nm:
        args["pixelsize_nm"] = csv.pixelsize_nm
    if csv.first_frame is not None:
        args["first_frame"] = csv.first_frame
    return args


@dataclass
class OpenSettings:
    """Nothing to set: replacing or adding is which button is clicked."""


@register("File/Load/Load")
class Open(Plugin):
    """Choose files and open them: the first line of the File tab."""

    description = ("Open: choose files and open them in place of the open ones; "
                   "Add: add them as more files.  Localization tables in any "
                   "format Load from path reads; images as image layers")
    Settings = OpenSettings
    changes_locs = False          # a launcher: the window loads, not `run`
    version = "3"
    # the window's own File > Open, told which: a csv no conversion knows asks
    # for its columns there (`gui.dialogs.CsvMappingDialog`).  Two buttons
    # that say what they do, one row -- the menu's Open asks instead
    window = "open files"
    window_buttons = (("Add", "add files"),)

    def run(self, ctx: Context, settings: OpenSettings) -> Result:
        raise ValueError("Load opens a file dialog; in a script or a chain, "
                         "use File/Load/Auto with the file's path")


@register("File/Load/Auto")
class LoadAuto(Plugin):
    """Read a localization file of any known format into the session."""

    Settings = LoadSettings
    name = "Load from path"
    changes_locs = True
    # 2: no *link blinks*; a table with a trace id is grouped by trace;
    # 3: the per-format loaders are a *format* choice here, and the csv
    # *columns* came with them; 4: a csv keeps every column, and its
    # conversion (units, first frame) is a part of its own; 5: Abberior
    # `.msr` files, m2410 MINFLUX exports, and a MINFLUX localization's cfr
    # from the iteration that measures it; 6: a MINFLUX localization's
    # precision is the Cramér-Rao bound of its pattern, not 150 nm / sqrt(N)
    version = "6"

    def describe(self, ctx: Optional[Context], settings: LoadSettings) -> FormState:
        """The csv part is shown for a csv only, the MSR part for an `.msr`."""
        name = settings.format
        if not name and settings.path:
            try:
                name = reader_for(Path(settings.path)).name
            except (ValueError, OSError):
                name = ""
        return FormState(shown={"csv": name == "csv", "msr": name == MSR},
                         active={"csv.pixelsize_nm": settings.csv.units == "px"})

    def reader_args(self, path: Path, reader, settings) -> Dict[str, Any]:
        """Anything the reader needs beyond the path: a csv's columns."""
        return csv_args(settings.csv) if reader.name == "csv" else {}

    def run(self, ctx: Context, settings: LoadSettings) -> Result:
        from ..session import read_localizations
        if not settings.path:
            raise ValueError("choose a file")
        path = Path(settings.path)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        reader = _reader(path, settings)
        ctx.report(f"reading {path.name} as {reader.name}")
        if reader.name == MSR:
            return self._run_msr(ctx, path, settings)
        locs, info = read_localizations(path, progress=ctx.report, reader=reader,
                                        **self.reader_args(path, reader, settings))
        text = f"{len(locs)} localizations from {path.name} ({info.format})"
        # not merged here: the layer merges its blinks when first read, and
        # the GUI in the background once the file is open
        return Result(files=[(locs, info, None)], text=text,
                      append=settings.append, data={"info": info}, settings=settings)

    def _run_msr(self, ctx: Context, path: Path, settings: LoadSettings) -> Result:
        """An `.msr`: a file per measurement (or one for all), and its images."""
        from ..io import msr
        m = settings.msr
        found = msr.contents(path)
        files = msr.read_files(path, m.measurements or None, m.merge,
                               not m.keep_invalid, m.z_scale, ctx.report, found)
        images = [msr.image(path, im) for im in found.images
                  if not im.render] if m.images else []
        n = sum(len(locs) for locs, _ in files)
        text = (f"{n} localizations from {path.name} (Abberior MSR): "
                f"{len(files)} file{'s' * (len(files) != 1)}"
                + (f", {len(images)} images" if images else ""))
        return Result(files=[(locs, info, None) for locs, info in files],
                      image_sources=images, text=text, append=settings.append,
                      data={"info": [info for _, info in files]}, settings=settings)


# --------------------------------------------------------------------- save

@dataclass
class SaveSettings:
    path: str = param("", label="file", kind="save_file", file_filter="",
                      help="an .hdf5 or .h5 file; empty saves over the open file")
    gui_state: bool = param(
        True, label="save the GUI state too",
        help="the tabs, their plugins and their parameters, for File > "
             "Workspace > From data file to put back")
    region_results: bool = param(
        True, label="save the ROI results",
        help="the numbers measured in the ROIs; off keeps "
             "the ROIs and drops their results")
    region_figures: bool = param(
        True, label="save the ROI figures' data",
        help="what each ROI kept to draw its figures again; off keeps the "
             "numbers, and a figure is drawn by measuring again")


def region_counts(session) -> Dict[str, int]:
    """What a save could leave out: the ROIs with results, the regions of id
    columns with results (clusters), and the results that kept their
    figures' data."""
    rois = list(session.rois.tables.values())
    per_column = [list(tables.values()) for tables in session.id_regions.values()]
    return {"rois": len({r for t in rois for r in t.ids}),
            "regions": sum(max((len(t) for t in tables), default=0)
                           for tables in per_column),
            "kept": sum(1 for t in rois + [t for ts in per_column for t in ts]
                        for k in t.kept if k)}


def left_out(session, settings: SaveSettings, done: bool = False) -> Optional[str]:
    """The sentence saying what ``settings`` leave out of the file that the
    session has, before the save (``done`` False) or after it; None when
    nothing is."""
    if settings.region_results and settings.region_figures:
        return None
    counts = region_counts(session)
    parts = [f"{n} {what}" for n, what in ((counts["rois"], "ROIs"),
                                           (counts["regions"], "regions of id columns"))
             if n]
    if not settings.region_results and parts:
        return (f"The results measured in {' and '.join(parts)} "
                + ("were not saved: they are" if done else "will not be saved: they are")
                + " measured again after opening.")
    if not settings.region_figures and counts["kept"]:
        return (f"The figure data of {counts['kept']} region results "
                + ("was" if done else "will") + " not "
                + ("saved" if done else "be saved")
                + ": their figures are drawn by measuring again.")
    return None


@register("File/Save/smappy HDF5")
class Save(Plugin):
    """Write the current table, its history and its ROIs."""

    Settings = SaveSettings
    changes_locs = False
    logged = True          # where a file went is not visible in the file

    def describe(self, ctx: Optional[Context], settings: SaveSettings) -> FormState:
        """Results there are that these settings would not save, said before
        the save rather than found missing when the file is opened again."""
        if ctx is None or ctx.session is None:
            return FormState()
        text = left_out(ctx.session, settings)
        return FormState(notice=None if text is None else Notice(text, kind="info"))

    def run(self, ctx: Context, settings: SaveSettings) -> Result:
        if ctx.session is None:
            raise ValueError("nothing to save: there is no session")
        path = Path(settings.path) if settings.path else ctx.session.path
        if not path:
            raise ValueError("choose where to save")
        writer_for(path)                       # fail before doing the work
        written = ctx.session.save(path, gui_state=settings.gui_state,
                                   region_results=settings.region_results,
                                   region_figures=settings.region_figures)
        text = f"{len(ctx.session.locs)} localizations to {written}"
        missing = left_out(ctx.session, settings, done=True)
        if missing:                       # said before, and kept in the log
            text += "; " + missing[0].lower() + missing[1:]
        return Result(text=text, written=str(written), settings=settings)


# ------------------------------------------------------------------- export

#: what the export writes, and what the file name chooses when it is left to
#: it: a text suffix is a csv, an HDF5 one Picasso's -- smappy's own HDF5 is
#: what *Save* writes, with everything else the session has
EXPORT_FORMATS = (("", "by the file name"), ("csv", "csv (smappy's names)"),
                  ("ThunderSTORM", "csv (ThunderSTORM's names)"),
                  ("Picasso", "Picasso HDF5"),
                  ("smappy HDF5", "smappy HDF5, the table only"))
EXPORT_FILTER = ("Text (*.csv *.tsv *.txt);;Picasso HDF5 (*.hdf5 *.h5);;"
                 "All files (*)")


def export_format(settings) -> str:
    """The format ``settings`` write: the one chosen, or the file name's."""
    if settings.format:
        return settings.format
    suffix = Path(settings.path).suffix.lower() if settings.path else ""
    return "Picasso" if suffix in (".hdf5", ".h5") else "csv"


@dataclass
class ExportSettings:
    path: str = param("", label="file", kind="save_file", file_filter=EXPORT_FILTER,
                      help="where to write; .csv, .tsv, .txt or .hdf5")
    format: str = param("", label="format", choices=EXPORT_FORMATS,
                        help="what to write; by the file name means a csv for a "
                             "text suffix and Picasso's for .hdf5")
    localizations: str = param("visible", label="localizations",
                               choices=(("visible", "visible: filter and ROI"),
                                        ("all", "all")),
                               help="what the layer shows, or the whole table")
    grouping: str = param("layer", label="table",
                          choices=(("layer", "as the layer shows it"),
                                   ("ungrouped", "localizations"),
                                   ("grouped", "merged blinks")),
                          help="the localizations, or one row per blink")
    columns: list = param(default_factory=list, label="columns", kind="columns",
                          help="the columns to write; all ticked writes every "
                               "column, also those a later table adds")
    group: str = param("", label="Picasso group",
                       help="an integer column written as Picasso's group, "
                            "its picks: cluster_id, roi_id, ...")


@register("File/Export/Localizations")
class ExportLocalizations(Plugin):
    """Write the localizations for another program: csv, ThunderSTORM's csv,
    or Picasso's HDF5."""

    Settings = ExportSettings
    name = "Save localizations"
    changes_locs = False
    logged = True          # where a file went is not visible in the file
    version = "1"

    def describe(self, ctx: Optional[Context], settings: ExportSettings) -> FormState:
        """Columns to choose for a free-form format, a group column for
        Picasso's fixed one; the session's columns in both lists."""
        picasso = export_format(settings) == "Picasso"
        state = FormState(shown={"columns": not picasso, "group": picasso})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            locs = ctx.locs
            state.choices["columns"] = list(locs.keys())
            state.choices["group"] = [("", "none")] + [
                (n, n) for n in locs.keys()
                if np.asarray(locs[n]).dtype.kind in "iu" and n != "frame"]
        return state

    def run(self, ctx: Context, settings: ExportSettings) -> Result:
        import copy

        from ..io.formats import writer_named
        if not settings.path:
            raise ValueError("choose where to write")
        path = Path(settings.path)
        fmt = export_format(settings)
        view = copy.copy(ctx)
        view.grouping = None if settings.grouping == "layer" else settings.grouping
        locs, selection = view.table()
        if settings.localizations == "visible":
            locs = locs[selection.mask]
        columns = [c for c in settings.columns if c in locs]
        if settings.columns and not columns:
            raise ValueError("none of the chosen columns is in the table; it has "
                             + ", ".join(locs.keys()))
        said = []
        if fmt in ("csv", "ThunderSTORM"):
            written = writer_named("csv").save(
                locs, path, columns=columns or None,
                names="ThunderSTORM" if fmt == "ThunderSTORM" else "smappy")
        elif fmt == "Picasso":
            written = writer_named("Picasso").save(
                locs, path, group=settings.group or None, report=said.append)
        elif fmt == "smappy HDF5":
            if columns:
                from ..locs import Localizations
                locs = Localizations({c: locs[c] for c in columns}, dict(locs.metadata))
            written = writer_named("smappy HDF5").save(locs, path)
        else:
            raise ValueError(f"no export format {fmt!r}")
        for line in said:
            ctx.warn(line)
        text = f"{len(locs)} localizations to {written} ({fmt})"
        if said:
            text += "; " + "; ".join(said)
        return Result(text=text, written=str(written), settings=settings)


# The picture's choices, as `smappy.picture` names them; written out here so
# that discovery and the form need nothing imported
_WHAT = (("view", "as the view"), ("composite", "composite only"),
         ("layers", "each layer, colour"), ("layers grey", "each layer, grey"),
         ("intensity", "intensity, 32-bit"))
_REGIONS = (("view", "the view"), ("roi", "the ROI"), ("all", "everything"))
IMAGE_PATH = "File/Export/Image"


@dataclass
class ExportImageSettings:
    path: str = param("", label="file", kind="save_file",
                      file_filter="PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg)",
                      help="the picture; its extension (.png, .tif, .jpg) is its format")
    what: str = param("view", choices=_WHAT,
                      help="the panels as the window shows them, the layers "
                           "added up, a file per layer, or the summed intensity "
                           "as numbers (a TIFF)")
    region: str = param("view", label="area", choices=_REGIONS,
                        help="what the window shows, the drawn ROI's box, or "
                             "all the data")
    pixelsize_nm: Optional[float] = param(None, label="pixel", unit="nm", min=0.01,
                                          help="the picture's pixel; auto: as "
                                               "the window displays it")
    scale_bar: bool = param(True, label="scale bar",
                            help="a bar of a round length, bottom right")
    scale_bar_nm: Optional[float] = param(None, label="bar length", unit="nm", min=0.0,
                                          help="auto: a round fifth of the width, "
                                               "as the window's")
    colour_bars: bool = param(True, label="colour bars",
                              help="the colour scale of each layer coloured by "
                                   "a column, top right")
    labels: bool = param(True, label="labels",
                         help="each panel's or layer's name, top left")
    view: str = param("2D", label="window", choices=(("2D", "2D picture"),
                                                     ("3D", "3D view")),
                      help="which window's picture: the 3D view's is its "
                           "projection")


def image_settings_for(session, view: str = "2D", path: str = "",
                       **values) -> ExportImageSettings:
    """*Export Image* filled from a window: what its *Save image...* opens
    with, and what its quick PNG writes (``path`` given)."""
    settings = ExportImageSettings(path=path, view=view, **values)
    if not settings.path and session is not None and session.path:
        p = Path(session.path)
        suffix = "_3d" if view == "3D" else ""
        settings.path = str(p.with_name(f"{p.stem}{suffix}_image.png"))
    return settings


@register("File/Export/Image")
class ExportImage(Plugin):
    """Save the picture as the window shows it -- its layers, panels, scale
    bar and colour bars -- at a pixel size, with how it was made written into
    the file."""

    Settings = ExportImageSettings
    name = "Save image"
    changes_locs = False
    logged = True          # which picture came from this table, and how
    # 2: the colours scaled to 8 bits (the file was black); the layer's
    #    grouped table when it shows blinks; an empty selection refused.
    # 3: the window's picture -- every visible layer, the panels, the scale
    #    bar and colour bars -- not one layer's; each layer, or the summed
    #    intensity; the area and the settings written into the file
    version = "4"

    def describe(self, ctx: Optional[Context], settings: ExportImageSettings) -> FormState:
        from ..picture import auto_bar
        session = ctx.session if ctx is not None else None
        frames = getattr(session, "view_frames", {}) or {}
        frame = frames.get(settings.view)
        frame = frame() if callable(frame) else None
        hints, notice = {}, None
        if frame is not None:
            hints["pixelsize_nm"] = round(frame.pixelsize, 3)
        try:
            fov = _image_fov(session, frame, settings.region, settings.pixelsize_nm)
            hints["scale_bar_nm"] = auto_bar(fov.x1 - fov.x0)
            if settings.what != "intensity":
                notice = Notice(f"{fov.nx} x {fov.ny} pixels per panel", kind="info")
        except Exception:
            pass
        looks = settings.what != "intensity"
        if settings.what == "intensity" and Path(settings.path).suffix.lower() not in (
                ".tif", ".tiff", ""):
            notice = Notice("the intensity is numbers: written as a TIFF", kind="info")
        return FormState(
            hints=hints, notice=notice,
            active={"scale_bar": looks, "scale_bar_nm": looks and settings.scale_bar,
                    "colour_bars": looks and settings.what != "layers grey",
                    "labels": looks},
            shown={"view": "3D" in frames})

    def run(self, ctx: Context, settings: ExportImageSettings) -> Result:
        written = export_image(ctx, settings)
        if not written:
            raise ValueError("nothing to save")
        names = ", ".join(p.name for p in written)
        return Result(text=f"written: {names}", written=str(written[0]),
                      data={"files": [str(p) for p in written]}, settings=settings)


def _image_fov(session, frame, region: str, pixelsize_nm: Optional[float]):
    from ..picture import frame_fov
    roi = getattr(session, "roi", None)
    everything = session.full_view() if session is not None and session.layers else None
    return frame_fov(frame, region, pixelsize_nm,
                     roi_bounds=roi.bounds if roi is not None else None,
                     everything=everything)


def export_image(ctx: Context, settings: ExportImageSettings):
    """Render and write the picture *Export Image* describes; the paths.

    The one function behind the plugin and both windows' *Save*
    (`image_pictures`), each picture with how it was made written into it.
    """
    from ..picture import describe_view, description_text, file_paths, write_picture
    if not settings.path:
        raise ValueError("choose where to save")
    pictures, layers, frame = image_pictures(ctx, settings)
    written = []
    for picture, path in zip(pictures, file_paths(settings.path, pictures)):
        ctx.report(f"writing {path.name}...")
        text = description_text(describe_view(ctx.session, layers, settings,
                                              picture, frame))
        written.append(write_picture(picture, path, text))
    return written


def picture_setup(ctx: Context, view: str = "2D", region: str = "view",
                  pixelsize_nm: Optional[float] = None):
    """What a picture of the view is drawn from: ``(layers, fov, frame)``.

    The layers are a snapshot of the session's (`snapshot_layers`, what the
    windows render), framed by the window's `ViewFrame`
    (`Session.view_frames`).  Without a session -- a script -- the context's
    table is the one layer, drawn with the default settings, over its
    selection.  *Export Image* and *Export Stack* both start here.
    """
    from ..layer_state import LayerSnapshot, snapshot_layers
    from ..picture import frame_fov
    from ..render import RenderSettings
    session = ctx.session
    frame = None
    if session is not None and session.layers:
        frames = getattr(session, "view_frames", {}) or {}
        maker = frames.get(view)
        frame = maker() if maker is not None else None
        if view == "3D" and (frame is None or frame.projection is None):
            raise ValueError("the 3D view's picture needs the 3D window open")
        from ..view3d import snapshot_3d
        layers = (snapshot_3d(session.layers) if view == "3D"
                  else snapshot_layers(session.layers))
        fov = _image_fov(session, frame, region, pixelsize_nm)
    else:
        locs, selection = ctx.table()
        if not len(locs):
            raise ValueError("nothing to render")
        mask = np.asarray(selection.mask, bool)
        layers = (LayerSnapshot(name="localizations", locs=locs, mask=mask,
                                settings=RenderSettings()),)
        x, y = RenderSettings().axes.coordinates(locs, np.flatnonzero(mask))
        if not len(x):
            raise ValueError("nothing selected to render")
        everything = ((float(np.min(x)), float(np.max(x))),
                      (float(np.min(y)), float(np.max(y))))
        fov = frame_fov(None, "all", pixelsize_nm, everything=everything)
    shown = [l for l in layers if l.visible]
    if not shown:
        raise ValueError("no layer is visible")
    if all(l.mask is not None and not l.mask.any() for l in shown if not l.is_image):
        if not any(l.is_image for l in shown):
            # an empty selection used to draw the whole table, which is not
            # what the window shows
            raise ValueError("nothing selected to render: the filter or the "
                             "ROI leaves no localization")
    return layers, fov, frame


def image_pictures(ctx: Context, settings: ExportImageSettings):
    """The pictures *Export Image* describes, the layer snapshots they were
    drawn from and the window's frame: ``(pictures, layers, frame)``.
    What the render window copies is the first picture."""
    from ..picture import render_picture
    layers, fov, frame = picture_setup(ctx, settings.view, settings.region,
                                       settings.pixelsize_nm)
    panels = frame.panels if frame is not None else ("overlay",)
    pictures = render_picture(
        layers, fov, what=settings.what, panels=panels,
        stacked=frame.stacked if frame is not None else False,
        scale_bar=settings.scale_bar, scale_bar_length=settings.scale_bar_nm or None,
        colour_bars=settings.colour_bars, labels=settings.labels,
        projection=frame.projection if frame is not None else None,
        slab=frame.slab if frame is not None else None)
    return pictures, layers, frame


# -------------------------------------------------------------------- stack

@dataclass
class StackFramesSettings:
    step: int = param(1000, label="step", unit="frames", min=1,
                      help="camera frames from one movie frame to the next")
    window: Optional[int] = param(None, label="window", unit="frames", min=1,
                                  help="the camera frames each movie frame "
                                       "shows; auto: the step")
    cumulative: bool = param(False, label="cumulative",
                             help="each movie frame shows everything from the "
                                  "first camera frame on")
    tail: Optional[int] = param(None, label="track tail", unit="frames", min=1,
                                help="a layer in the tracks mode: the frames "
                                     "before now a track is drawn for; auto: "
                                     "the frames the others show")


@dataclass
class StackZSettings:
    step_nm: float = param(50.0, label="step", unit="nm", min=0.1,
                           help="from one slice to the next")
    thickness_nm: Optional[float] = param(None, label="thickness", unit="nm", min=0.1,
                                          help="of each slice; auto: the step")
    z_min_nm: Optional[float] = param(None, label="from", unit="nm",
                                      help="the first slice's lower edge; auto: "
                                           "the lowest z shown")
    z_max_nm: Optional[float] = param(None, label="to", unit="nm",
                                      help="where the last slice ends; auto: the "
                                           "highest z shown")


_ALONG = (("frames", "frames: a movie"), ("z", "z: slices"))
_LAYOUTS = (("view", "as the view"), ("composite", "composite only"),
            ("raw", "camera frames beside"))
_QUALITIES = (("high", "high"), ("medium", "medium"), ("low", "small file"))


@dataclass
class ExportStackSettings:
    path: str = param("", label="file", kind="save_file",
                      file_filter="TIFF stack (*.tif);;MP4 (*.mp4);;GIF (*.gif)",
                      help="the stack; its extension (.tif, .mp4, .gif) is its format")
    along: str = param("frames", choices=_ALONG,
                       help="what moves from one frame of the stack to the next")
    frames: StackFramesSettings = param(default_factory=StackFramesSettings,
                                        label="frames")
    z: StackZSettings = param(default_factory=StackZSettings, label="z")
    layout: str = param("view", choices=_LAYOUTS,
                        help="the panels as the window shows them, the layers "
                             "added up, or the camera frames the fit kept on "
                             "the left of the composite")
    region: str = param("view", label="area", choices=_REGIONS,
                        help="what the window shows, the drawn ROI's box, or "
                             "all the data")
    pixelsize_nm: Optional[float] = param(None, label="pixel", unit="nm", min=0.01,
                                          help="the picture's pixel; auto: as "
                                               "the window displays it")
    fps: float = param(10.0, label="frame rate", unit="1/s", min=0.1, max=120.0,
                       help="frames per second of the movie")
    quality: str = param("medium", choices=_QUALITIES,
                         help="MP4: how faithful, against how large")
    stamp: bool = param(True, label="write time / z",
                        help="the frames or seconds, or the z range, on each frame")
    scale_bar: bool = param(True, label="scale bar",
                            help="a bar of a round length, bottom right")
    colour_bars: bool = param(True, label="colour bars",
                              help="the colour scale of each layer coloured by "
                                   "a column, top right")
    labels: bool = param(True, label="labels",
                         help="each panel's name, top left")


@register("File/Export/Stack")
class ExportStack(Plugin):
    """A movie along the frames, or a stack of z slices: the view rendered
    step by step, as a TIFF stack, an MP4 or a GIF."""

    Settings = ExportStackSettings
    name = "Save stack"
    changes_locs = False
    logged = True
    version = "1"

    def describe(self, ctx: Optional[Context], settings: ExportStackSettings) -> FormState:
        from ..movie import estimate_size, has_encoder, kind_of
        frames = settings.along == "frames"
        try:
            kind = kind_of(settings.path) if settings.path else "tiff"
        except ValueError:
            kind = "tiff"
        notice = None
        if kind == "mp4" and not has_encoder():
            notice = Notice("an MP4 needs imageio-ffmpeg: pip install imageio-ffmpeg; "
                            "TIFF and GIF need nothing",
                            kind="warning")
        elif ctx is not None and ctx.session is not None and ctx.session.layers:
            try:
                plan = stack_plan(ctx, settings)
                h, w = plan.shape
                size = estimate_size(kind, h, w, len(plan.steps), settings.quality)
                about = "" if kind == "tiff" else "about "
                notice = Notice(f"{len(plan.steps)} frames of {w} x {h}: "
                                f"{about}{_megabytes(size)}", kind="info")
            except Exception:
                pass
        return FormState(
            notice=notice,
            shown={"frames": frames, "z": not frames,
                   "quality": kind == "mp4", "fps": kind != "tiff"})

    def preflight(self, ctx: Context, settings: ExportStackSettings):
        """The number of frames and how long they will take: one is rendered
        here and timed, and a stack of more than a minute is asked about."""
        import time
        plan = stack_plan(ctx, settings)
        started = time.perf_counter()
        next(iter(plan.frames(plan.steps[len(plan.steps) // 2:][:1])))
        seconds = (time.perf_counter() - started) * len(plan.steps)
        text = f"{len(plan.steps)} frames, about {_duration(seconds)} to render"
        ctx.report(text)
        if seconds > 60:
            return text + ".  Make it?"
        return None

    def run(self, ctx: Context, settings: ExportStackSettings) -> Result:
        from ..movie import kind_of, write_stack
        from ..picture import describe_view, description_text, Picture
        from . import Stopped
        if not settings.path:
            raise ValueError("choose where to save")
        kind_of(settings.path)
        plan = stack_plan(ctx, settings)
        n = len(plan.steps)

        def frames():
            for k, image in enumerate(plan.frames(plan.steps)):
                if ctx.stopped:
                    raise Stopped()
                ctx.report(f"frame {k + 1} of {n}")
                yield image

        described = describe_view(ctx.session, plan.layers, settings,
                                  Picture(np.zeros((1, 1), np.uint8), plan.fov), plan.frame)
        described["stack"] = {"along": settings.along, "column": plan.column,
                              "steps": [[s.lo, s.hi] for s in plan.steps]}
        path = Path(settings.path)
        try:
            written = write_stack(path, frames(), n, fps=settings.fps,
                                  quality=settings.quality, pixel_nm=plan.fov.pixelsize,
                                  interval=plan.interval, along=settings.along,
                                  description=description_text(described))
        except BaseException:
            path.unlink(missing_ok=True)       # half a movie is no movie
            raise
        return Result(text=f"{n} frames written to {written.name}", written=str(written),
                      data={"frames": n}, settings=settings)


@dataclass
class StackPlan:
    """What *Export Stack* renders: the layers, the grid, the steps, and
    how one frame is made (`frames`)."""

    layers: tuple
    fov: object
    frame: object
    steps: list
    column: str
    settings: ExportStackSettings
    raw: list
    shape: tuple                 # (height, width) of a frame
    interval: Optional[float]    # seconds between frames, or the z step in nm
    frame_time_ms: Optional[float]

    def frames(self, steps):
        from ..movie import stack_frames
        s = self.settings
        tail = s.frames.tail if s.along == "frames" else None
        panels = self.frame.panels if self.frame is not None else ("overlay",)
        return stack_frames(
            self.layers, self.fov, steps, column=self.column,
            what="composite" if s.layout != "view" else "view", panels=panels,
            stacked=self.frame.stacked if self.frame is not None else False,
            tail=tail, raw=self.raw, scale_bar=s.scale_bar,
            colour_bars=s.colour_bars, labels=s.labels,
            text=self.text if s.stamp else None)

    def text(self, step) -> str:
        if self.column != "frame":
            return f"z {step.lo:g} to {step.hi:g} nm"
        if self.frame_time_ms:
            return f"{step.hi * self.frame_time_ms / 1000:.1f} s"
        return f"frames {step.lo:.0f}-{step.hi - 1:.0f}"


def stack_plan(ctx: Context, settings: ExportStackSettings) -> StackPlan:
    """The layers, grid and steps of a stack, from the view (`picture_setup`)."""
    from ..metadata import file_frame_time
    from ..movie import frame_steps, shown_range, z_steps
    layers, fov, frame = picture_setup(ctx, "2D", settings.region, settings.pixelsize_nm)
    raw = []
    if settings.along == "frames":
        column = "frame"
        first, last = shown_range(layers, column)
        f = settings.frames
        steps = frame_steps(first, last, f.step, f.window, f.cumulative)
        session = ctx.session
        locs = session.locs if session is not None else ctx.locs
        frame_time = file_frame_time(getattr(locs, "metadata", None))
        interval = f.step * frame_time / 1000.0 if frame_time else None
        if settings.layout == "raw":
            raw = session.raw_images() if session is not None else []
            if not raw:
                raise ValueError("no camera frames are kept with this table: a fit "
                                 "keeps them (Localize, raw frames), and the "
                                 "localization file it wrote has them")
    else:
        column = "z_nm"
        if settings.layout == "raw":
            raise ValueError("the camera frames go beside a movie along frames, "
                             "not a stack along z")
        z = settings.z
        lo, hi = shown_range(layers, column)
        z0 = lo if z.z_min_nm is None else z.z_min_nm
        z1 = hi if z.z_max_nm is None else z.z_max_nm
        if not z1 > z0:
            raise ValueError(f"z from {z0:g} to {z1:g} nm is no range")
        steps = z_steps(z0, z1, z.step_nm, z.thickness_nm)
        interval, frame_time = z.step_nm, None
    if settings.layout == "view" and frame is not None and len(frame.panels) > 1:
        n = len(frame.panels)
        shape = ((fov.ny * n + 2 * (n - 1), fov.nx) if frame.stacked
                 else (fov.ny, fov.nx * n + 2 * (n - 1)))
    elif raw:
        shape = (fov.ny, 2 * fov.nx + 2)
    else:
        shape = (fov.ny, fov.nx)
    return StackPlan(layers, fov, frame, steps, column, settings, raw, shape,
                     interval, frame_time)


def _megabytes(size: float) -> str:
    if size < 1e6:
        return f"{size / 1e3:.0f} kB"
    return f"{size / 1e6:.1f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# ----------------------------------------------------------------- simulate

@dataclass
class SimulateStructureSettings(SimulationSettings):
    # here, not in `SimulationSettings`: what happens to the open files is the
    # plugin's business, not the model's, and scripts build the model alone
    append: bool = param(False, label="add as a new file",
                         help="keep the open files and add the simulation as one "
                              "more; off: it replaces them")


@register("File/Simulate/Blinking Structure")
class SimulateBlinks(Plugin):
    """A labelled structure, blinking: localizations to try, or camera frames
    to fit (`smappy.simulate`)."""

    # the path keeps the old name, so the GUIs and chains saved with it still
    # find the plugin; the name is what the panel says, and "Blinking
    # Structure" did not say that it simulates
    name = "Simulate structure"
    Settings = SimulateStructureSettings
    changes_locs = True
    # 3: localizations down to 10 photons; free linkage, poses.  4: the
    # precision is Mortensen's maximum-likelihood one, not least squares'
    version = "4"

    def describe(self, ctx: Optional[Context], settings: SimulationSettings) -> FormState:
        """Once the simulation file exists, offer to fit it: the GUI puts it
        into the Localize tab's fitter (the window ``"fit file"``, which is
        what a stack dropped on the window goes through), so nobody retypes
        a path the plugin has just written."""
        from ..simulate.source import recipe_path
        if settings.output != "camera" or not settings.camera.path:
            return FormState()
        path = recipe_path(settings.camera.path)
        if not path.is_file():
            return FormState()
        return FormState(notice=Notice(
            f"{path.name} is written: fit it in the Localize tab",
            action="Fit this", window="fit file", settings=str(path),
            kind="info"))

    def run(self, ctx: Context, settings: SimulationSettings) -> Result:
        from ..simulate import camera_truth, ground_truth, localizations
        ctx.report(f"simulating {settings.n_frames} frames...")
        truth = ground_truth(settings)
        if settings.output == "camera":
            return self._camera(ctx, settings, truth, camera_truth)
        locs = localizations(truth)
        name = "simulation (drift)" if settings.drift else "simulation"
        info = FileInfo(name=name, path=name, format="simulated", n=len(locs))
        md = locs.metadata
        close = ("removed" if settings.localizations.close == "remove" else "averaged")
        text = (f"{len(locs)} localizations of {md['n_emitters']} emitters "
                f"({md['n_labels']} labels), {md['n_blinks']} blinks over "
                f"{settings.n_frames} frames, off time {md['off_time_frames']:.0f} "
                f"frames; {md['n_close']} too close together, {close}")
        return Result(files=[(locs, info, None)], text=text, settings=settings,
                      append=getattr(settings, "append", False))

    def _camera(self, ctx, settings, truth, camera_truth) -> Result:
        from ..simulate.source import write_recipe
        cam = settings.camera
        if not cam.path:
            raise ValueError("camera frames: choose where to write the simulation "
                             "file (camera frames > simulation file)")
        path = write_recipe(settings, cam.path)
        text = (f"{len(truth.emission)} spots of {len(truth.fluorophores)} emitters "
                f"over {settings.n_frames} frames of {cam.size_px} x {cam.size_px} "
                f"pixels: {path} -- open it in a fitter as its file")
        data = {}
        if cam.tiff:
            import tifffile
            from ..simulate import render
            tif = path.with_name(path.name[:-len(".sim.yaml")] + ".tif")
            ctx.report(f"writing {tif}...")
            with tifffile.TiffWriter(tif, bigtiff=True) as writer:
                for a in range(0, settings.n_frames, 500):
                    writer.write(render(truth, a, a + 500), contiguous=True)
            data["tiff"] = str(tif)
            text += f"; the frames in {tif.name}"
        files = []
        if cam.load_truth:
            locs = camera_truth(truth)
            name = f"{path.name} (truth)"
            files = [(locs, FileInfo(name=name, path=str(path), format="simulated",
                                     n=len(locs)), None)]
        return Result(files=files, text=text, data=data, written=str(path),
                      settings=settings, append=getattr(settings, "append", False))


@dataclass
class TrackDiffusionSettings:
    d_um2_s: float = param(0.5, label="D", unit="µm²/s", min=0.0,
                           help="the diffusion coefficient of the first state")
    slow_fraction: float = param(0.0, label="second state", min=0.0, max=1.0,
                                 help="the fraction of the time in a second state; "
                                      "0: one state")
    d2_um2_s: float = param(0.02, label="D of the second", unit="µm²/s", min=0.0,
                            help="the diffusion coefficient of the second state")
    switch_probability: float = param(0.0, label="switching", min=0.0, max=1.0,
                                      help="per frame: the state is drawn again")


@dataclass
class TrackBlinkingSettings:
    p_dark: float = param(0.1, label="going dark", min=0.0, max=1.0,
                          help="per frame: the probability that a bright "
                               "fluorophore goes dark")
    dark_frames: float = param(1.5, label="dark frames", min=1.0,
                               help="the mean length of a dark run")


@dataclass
class TrackLocalizationSettings:
    precision_nm: float = param(20.0, label="precision", unit="nm", min=0.0,
                                help="the localization error per lateral axis")
    use_z: bool = param(False, label="3D", help="diffusion and an error in z too")
    z_precision_nm: float = param(50.0, label="z precision", unit="nm", min=0.0,
                                  help="the localization error in z")
    z_range_nm: float = param(800.0, label="z range", unit="nm", min=0.0,
                              advanced=True, help="where the particles start in z")
    noise_density_um2: float = param(0.0, label="spurious", unit="/µm²", min=0.0,
                                     help="localizations in no track, per µm² "
                                          "and frame")


@dataclass
class SimulateTracksSettings:
    n_frames: int = param(1000, label="frames", min=1, help="the length of the movie")
    frame_time_ms: float = param(10.0, label="frame time", unit="ms", min=0.001,
                                 help="the time between frames")
    field_um: float = param(20.0, label="field", unit="µm", min=0.1,
                            help="the side of the square field the particles "
                                 "start in")
    density_um2: float = param(0.05, label="density", unit="/µm²", min=0.0,
                               help="particles present in a frame, per µm²")
    track_frames: float = param(20.0, label="track length", unit="frames", min=1.0,
                                help="the mean frames a particle lasts before it "
                                     "bleaches")
    seed: int = param(0, label="seed", min=0,
                      help="the same seed and settings give the same tracks")
    append: bool = param(False, label="add as a new file",
                         help="keep the open files and add the simulation as one "
                              "more; off: it replaces them")
    diffusion: TrackDiffusionSettings = param(default_factory=TrackDiffusionSettings,
                                              label="diffusion")
    blinking: TrackBlinkingSettings = param(default_factory=TrackBlinkingSettings,
                                            label="blinking")
    localizations: TrackLocalizationSettings = param(
        default_factory=TrackLocalizationSettings, label="localizations",
        collapsed=True)

    def model(self):
        """The `smappy.simulate.TrackSimulationSettings` these stand for: a
        second state only when it has time in it."""
        from ..simulate import TrackSimulationSettings
        d, b, loc = self.diffusion, self.blinking, self.localizations
        two = d.slow_fraction > 0
        return TrackSimulationSettings(
            n_frames=self.n_frames, field_um=self.field_um,
            density_um2=self.density_um2, track_frames=self.track_frames,
            frame_time_ms=self.frame_time_ms,
            d_um2_s=(d.d_um2_s, d.d2_um2_s) if two else (d.d_um2_s,),
            fractions=(1 - d.slow_fraction, d.slow_fraction) if two else (1.0,),
            switch_probability=d.switch_probability, p_dark=b.p_dark,
            dark_frames=b.dark_frames, precision_nm=loc.precision_nm,
            use_z=loc.use_z, z_range_nm=loc.z_range_nm,
            z_precision_nm=loc.z_precision_nm,
            noise_density_um2=loc.noise_density_um2, seed=self.seed)


@register("File/Simulate/Tracks")
class SimulateTracks(Plugin):
    """Single-particle tracks: Brownian particles with diffusive states and
    blinking, the truth in track_id_true (`smappy.simulate.simulate_tracks`)."""

    name = "Simulate tracks"
    Settings = SimulateTracksSettings
    changes_locs = True
    version = "1"

    def describe(self, ctx: Optional[Context], settings: SimulateTracksSettings
                 ) -> FormState:
        """The second state's fields are read only when it has time in it,
        z's only in 3D; the jump and the spacing, to set Link tracks by."""
        from ..simulate import jump_nm, spacing_nm
        d, loc = settings.diffusion, settings.localizations
        two = d.slow_fraction > 0
        state = FormState(active={"diffusion.d2_um2_s": two,
                                  "diffusion.switch_probability": two,
                                  "localizations.z_precision_nm": loc.use_z,
                                  "localizations.z_range_nm": loc.use_z})
        dims = 3 if loc.use_z else 2
        jump = jump_nm(max(d.d_um2_s, d.d2_um2_s if two else 0.0),
                       settings.frame_time_ms, dims)
        state.notice = Notice(
            f"rms jump {jump:.0f} nm, molecules {spacing_nm(settings.density_um2):.0f} "
            f"nm apart", kind="info")
        return state

    def run(self, ctx: Context, settings: SimulateTracksSettings) -> Result:
        from ..simulate import simulate_tracks
        ctx.report(f"simulating {settings.n_frames} frames of tracks...")
        locs = simulate_tracks(settings.model())
        info = FileInfo(name="tracks (simulation)", path="tracks (simulation)",
                        format="simulated", n=len(locs),
                        frame_time_ms=settings.frame_time_ms)
        md = locs.metadata
        text = (f"{len(locs)} localizations of {md['n_tracks']} tracks over "
                f"{settings.n_frames} frames ({settings.frame_time_ms:g} ms each); "
                f"the truth in track_id_true and state_true")
        return Result(files=[(locs, info, None)], text=text, settings=settings,
                      append=getattr(settings, "append", False))
