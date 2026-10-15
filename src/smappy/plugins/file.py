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

LOCS_FILTER = "Localizations (*.hdf5 *.h5 *.mat *.csv *.npy *.zip *.json)"


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
class LoadSettings:
    path: str = param("", label="file", kind="open_file", file_filter=LOCS_FILTER,
                      help="the localization file to read")
    append: bool = param(False, label="add to the open files",
                         help="join the table as one more file instead of "
                              "replacing everything; this is File > Add file")
    format: str = param("", label="format", choices=_format_choices, advanced=True,
                        help="auto: from the file's name and, where that is not "
                             "enough, its contents")
    csv: CsvSettings = param(default_factory=CsvSettings, label="csv columns",
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
    append: bool = param(False, label="add",
                         help="add the files to the open ones instead of "
                              "replacing them")


@register("File/Load/Load")
class Open(Plugin):
    """Choose localization files and open them: the first line of the File tab."""

    description = ("choose localization files in a dialog and open them, or "
                   "add them to the open ones; any format File/Load/Auto reads")
    Settings = OpenSettings
    changes_locs = False          # a launcher: the window loads, not `run`
    version = "1"
    # the window's own File > Open: a csv no conversion knows asks for its
    # columns there (`gui.dialogs.CsvMappingDialog`)
    window = "open files"

    def run(self, ctx: Context, settings: OpenSettings) -> Result:
        raise ValueError("Load opens a file dialog; in a script or a chain, "
                         "use File/Load/Auto with the file's path")


@register("File/Load/Auto")
class LoadAuto(Plugin):
    """Read a localization file of any known format into the session."""

    Settings = LoadSettings
    changes_locs = True
    # 2: no *link blinks*; a table with a trace id is grouped by trace;
    # 3: the per-format loaders are a *format* choice here, and the csv
    # *columns* came with them; 4: a csv keeps every column, and its
    # conversion (units, first frame) is a part of its own
    version = "4"

    def describe(self, ctx: Optional[Context], settings: LoadSettings) -> FormState:
        """The csv part is shown for a csv only."""
        csv = settings.format == "csv"
        if not settings.format and settings.path:
            try:
                csv = reader_for(Path(settings.path)).name == "csv"
            except (ValueError, OSError):
                csv = False
        return FormState(shown={"csv": csv},
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
        locs, info = read_localizations(path, progress=ctx.report, reader=reader,
                                        **self.reader_args(path, reader, settings))
        text = f"{len(locs)} localizations from {path.name} ({info.format})"
        # not merged here: the layer merges its blinks when first read, and
        # the GUI in the background once the file is open
        return Result(files=[(locs, info, None)], text=text,
                      append=settings.append, data={"info": info}, settings=settings)


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


@dataclass
class ExportImageSettings:
    path: str = param("", label="file", kind="save_file",
                      file_filter="PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg)",
                      help="the picture; its extension (.png, .tif, .jpg) is its format")
    pixelsize_nm: float = param(10.0, label="pixel", unit="nm", min=0.01,
                                help="the rendered pixel, in the table's units")


@register("File/Export/Image")
class ExportImage(Plugin):
    """Render the localizations to a picture file, with no window."""

    Settings = ExportImageSettings
    changes_locs = False
    logged = True          # which picture came from this table, and how
    # 2: the colours scaled to 8 bits (the file was black); the layer's
    #    grouped table when it shows blinks; an empty selection refused
    version = "2"

    def run(self, ctx: Context, settings: ExportImageSettings) -> Result:
        from ..render import save_image
        if not settings.path:
            raise ValueError("choose where to save")
        if not len(ctx.locs):
            raise ValueError("nothing to render")
        # the picture should look like the window does: the layer's render
        # and display settings, and its table -- the grouped one when the
        # layer shows blinks, which drawn ungrouped was a localization per
        # frame -- cut to what it selects
        render = display = None
        locs, selection = ctx.table()
        if ctx.session is not None and ctx.session.layers:
            index = ctx.layer
            if not 0 <= index < len(ctx.session.layers) or \
                    ctx.session.layers[index].is_image:
                index = ctx.session.first_locs_layer()
            layer = ctx.session.layers[index]
            render, display = layer.state.settings, layer.get_display()
        if not len(selection):
            # an empty selection drew the whole table, which is not what the
            # window shows
            raise ValueError("nothing selected to render: the filter or the "
                             "ROI leaves no localization")
        written = save_image(locs, settings.path,
                             pixelsize=settings.pixelsize_nm,
                             settings=render, display=display,
                             select=selection.mask)
        return Result(text=f"written to {written}", written=str(written),
                      settings=settings)


# ----------------------------------------------------------------- simulate

@register("File/Simulate/Blinking Structure")
class SimulateBlinks(Plugin):
    """A labelled structure, blinking: localizations to try, or camera frames
    to fit (`smappy.simulate`)."""

    # the path keeps the old name, so the GUIs and chains saved with it still
    # find the plugin; the name is what the panel says, and "Blinking
    # Structure" did not say that it simulates
    name = "Simulate structure"
    Settings = SimulationSettings
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
        return Result(files=[(locs, info, None)], text=text, settings=settings)

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
                      settings=settings)


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
                        format="simulated", n=len(locs))
        md = locs.metadata
        text = (f"{len(locs)} localizations of {md['n_tracks']} tracks over "
                f"{settings.n_frames} frames ({settings.frame_time_ms:g} ms each); "
                f"the truth in track_id_true and state_true")
        return Result(files=[(locs, info, None)], text=text, settings=settings)
