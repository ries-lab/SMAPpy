"""The File tab: getting localizations in, out, and made up.

Loading, saving, exporting and simulating were menu items and a script, which
meant they had no settings form, no place in the tree and nothing saved between
sessions.  They are ordinary plugins now.

A loader runs in the panel's worker thread and must not touch the session, so
it hands its work back as `Result.files` and `Session.apply` adds them on the
thread that owns the session.  The reading itself is `session.read_and_group`,
the same function the window's own `LoadTask` uses -- one implementation, so a
format behaves the same whichever way it was opened.

There is one small plugin per format rather than one with a dropdown, because a
format has its own settings: only the csv reader needs a column mapping.
Adding a format is still a one-line `register(Reader(...))` in `io.formats`;
giving it a plugin of its own here is optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..io.formats import (FileInfo, name_filter, reader_for, reader_named,
                          save_filter, writer_for)
from ..simulate.settings import SimulationSettings
from . import Context, Plugin, Result, param, register

LOCS_FILTER = "Localizations (*.hdf5 *.h5 *.mat *.csv *.npy *.zip *.json)"


# --------------------------------------------------------------------- load

@dataclass
class LoadSettings:
    path: str = param("", label="file", kind="open_file", file_filter=LOCS_FILTER,
                      help="the localization file to read")
    append: bool = param(False, label="add to the open files",
                         help="join the table as one more file instead of "
                              "replacing everything; this is File > Add file")
    group: Optional[bool] = param(None, label="link blinks", advanced=True,
                                  help="auto: link unless the file is being added")


class _Load(Plugin):
    """Read a localization file into the session."""

    Settings = LoadSettings
    format: str = ""            # "" is by extension; else a reader's name

    def reader_args(self, path: Path, settings) -> Dict[str, Any]:
        """Anything the reader needs beyond the path."""
        return {}

    def run(self, ctx: Context, settings: LoadSettings) -> Result:
        from ..group import GroupSettings
        from ..session import read_and_group
        if not settings.path:
            raise ValueError("choose a file")
        path = Path(settings.path)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if self.format:
            reader = reader_named(self.format)
            if reader is None:
                raise ValueError(f"no reader called {self.format!r}")
        else:
            reader = reader_for(path)          # raises with the known formats listed
        ctx.report(f"reading {path.name} as {reader.name}")
        group_settings = (ctx.session.group_settings if ctx.session is not None
                          else GroupSettings())
        locs, info, grouped = read_and_group(
            path, group_settings, append=settings.append, group=settings.group,
            progress=ctx.report, reader=reader,
            **self.reader_args(path, settings))
        text = f"{len(locs)} localizations from {path.name} ({info.format})"
        if grouped is not None:
            text += f", {len(grouped)} after linking"
        return Result(files=[(locs, info, grouped)], text=text,
                      data={"append": settings.append, "info": info}, settings=settings)


@register("File/Load/Auto")
class LoadAuto(_Load):
    """Read any known format, choosing the reader by the file name."""


@register("File/Load/smappy HDF5")
class LoadSmappy(_Load):
    """Read a file smappy wrote."""
    format = "smappy HDF5"
    favorite = False


@register("File/Load/SMAP")
class LoadSMAP(_Load):
    """Read a SMAP ``_sml.mat`` file."""
    format = "SMAP"
    favorite = False


@register("File/Load/MINFLUX")
class LoadMinflux(_Load):
    """Read a MINFLUX ``.npy``, ``.zip`` or ``.json`` export."""
    format = "MINFLUX"
    favorite = False


@dataclass
class LoadCsvSettings(LoadSettings):
    mapping: str = param("", label="columns", advanced=True,
                         help="header=column pairs, e.g. X=x_nm, Y=y_nm, T=frame; "
                              "empty means guess from the header")


@register("File/Load/csv")
class LoadCsv(_Load):
    """Read a csv, tsv or plain text table, guessing its columns."""

    Settings = LoadCsvSettings
    format = "csv"
    favorite = False

    def reader_args(self, path: Path, settings) -> Dict[str, Any]:
        from ..io.formats import csv_columns, guess_csv_mapping
        if settings.mapping.strip():
            pairs = [p.split("=", 1) for p in settings.mapping.split(",") if "=" in p]
            return {"mapping": {a.strip(): b.strip() for a, b in pairs}}
        headers, _, _ = csv_columns(path)
        return {"mapping": guess_csv_mapping(headers)}


# --------------------------------------------------------------------- save

@dataclass
class SaveSettings:
    path: str = param("", label="file", kind="save_file", file_filter="",
                      help="an .hdf5 or .h5 file; empty saves over the open file")
    gui_state: bool = param(
        True, label="save the GUI state too",
        help="the tabs, their plugins and their parameters, for File > "
             "Restore GUI state from file to put back")


@register("File/Save/smappy HDF5")
class Save(Plugin):
    """Write the current table, its history and its ROIs."""

    Settings = SaveSettings
    logged = True          # where a file went is not visible in the file

    def run(self, ctx: Context, settings: SaveSettings) -> Result:
        if ctx.session is None:
            raise ValueError("nothing to save: there is no session")
        path = Path(settings.path) if settings.path else ctx.session.path
        if not path:
            raise ValueError("choose where to save")
        writer_for(path)                       # fail before doing the work
        written = ctx.session.save(path, gui_state=settings.gui_state)
        return Result(text=f"{len(ctx.session.locs)} localizations to {written}",
                      data={"path": written}, settings=settings)


# ------------------------------------------------------------------- export

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
    logged = True          # which picture came from this table, and how

    def run(self, ctx: Context, settings: ExportImageSettings) -> Result:
        from ..render import save_image
        if not settings.path:
            raise ValueError("choose where to save")
        if not len(ctx.locs):
            raise ValueError("nothing to render")
        # the picture should look like the window does, so the render and
        # display settings come from the layer rather than from defaults
        render = display = None
        if ctx.session is not None and ctx.session.layers:
            layer = ctx.session.layers[ctx.session.first_locs_layer()]
            render, display = layer.state.settings, layer.get_display()
        written = save_image(ctx.locs, settings.path,
                             pixelsize=settings.pixelsize_nm,
                             settings=render, display=display,
                             select=ctx.selection.mask if len(ctx.selection) else None)
        return Result(text=f"written to {written}", data={"path": written},
                      settings=settings)


# ----------------------------------------------------------------- simulate

@register("File/Simulate/Blinking Structure")
class SimulateBlinks(Plugin):
    """A labelled structure, blinking: localizations to try, or camera frames
    to fit (`smappy.simulate`)."""

    Settings = SimulationSettings
    # 3: localizations down to 10 photons; free linkage, poses.  4: the
    # precision is Mortensen's maximum-likelihood one, not least squares'
    version = "4"

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
        return Result(files=[(locs, info, None)], text=text,
                      data={"append": False}, settings=settings)

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
        data = {"path": str(path)}
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
            data["append"] = False
        return Result(files=files, text=text, data=data, settings=settings)
