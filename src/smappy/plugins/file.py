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
from . import Context, Plugin, Result, param, register

LOCS_FILTER = "Localizations (*.hdf5 *.h5 *.mat *.csv *.npy *.zip *.json)"


# --------------------------------------------------------------------- load

@dataclass
class LoadSettings:
    path: str = param("", label="file", kind="open_file", file_filter=LOCS_FILTER)
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
                         help="x_nm=xnm, y_nm=ynm, ... ; empty means guess from "
                              "the header")


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
    path: str = param("", label="file", kind="save_file", file_filter="")
    gui_state: bool = param(
        True, label="save the GUI state too",
        help="the tabs, their plugins and their parameters, so reopening this "
             "file restores the session that produced it")


@register("File/Save/smappy HDF5")
class Save(Plugin):
    """Write the current table, its history and its ROIs."""

    Settings = SaveSettings

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
                      file_filter="PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg)")
    pixelsize_nm: float = param(10.0, label="pixel", unit="nm", min=0.01,
                                help="the rendered pixel, in the table's units")


@register("File/Export/Image")
class ExportImage(Plugin):
    """Render the localizations to a picture file, with no window."""

    Settings = ExportImageSettings

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

@dataclass
class SimulateSettings:
    n_frames: int = param(20000, label="frames", min=1)
    density: float = param(1.0, label="density", min=0.01,
                           help="scales the number of blinks per emitter")
    drift: bool = param(False, label="add drift",
                        help="a smooth random walk plus a slow creep of ~100 nm; "
                             "the truth is kept in the metadata as drift_truth")
    min_separation_nm: float = param(
        250.0, label="minimum separation", unit="nm", min=0,
        help="two emitters active in one frame closer than this could not have "
             "been fitted apart, so both are dropped")
    seed: int = param(0, label="seed", min=0)


@register("File/Simulate/Blinking Structure")
class SimulateBlinks(Plugin):
    """A ring, two lines and some scattered points, blinking: a dataset to try."""

    Settings = SimulateSettings

    def run(self, ctx: Context, settings: SimulateSettings) -> Result:
        from ..simulate import simulate
        ctx.report(f"simulating {settings.n_frames} frames...")
        locs = simulate(settings.n_frames, settings.seed, settings.drift,
                        settings.density, settings.min_separation_nm)
        name = "simulation (drift)" if settings.drift else "simulation"
        info = FileInfo(name=name, path=name, format="simulated", n=len(locs))
        text = (f"{len(locs)} localizations, {locs.metadata['n_emitters']} emitters, "
                f"{settings.n_frames} frames, "
                f"{locs.metadata['n_unresolvable_dropped']} overlapping dropped")
        return Result(files=[(locs, info, None)], text=text,
                      data={"append": False}, settings=settings)
