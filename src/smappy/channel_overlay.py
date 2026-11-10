"""Two channels of a split camera put on top of each other, one dye in each.

The other two-channel workflow.  In a ratiometric experiment every molecule is
seen in both halves of the chip, and the two-colour fitters pair the spots and
fit one emitter (`multichannel.MultiChannelEngine`).  Here each dye is seen in
one half only -- spectrally well separated, no crosstalk to speak of -- so
there is nothing to pair.  The movie is fitted as *two channels*, each half on
its own (`multichannel.SplitEngine`, ``camera_channel``), and the secondary
half's localizations are then moved onto the main half by a registration
measured on beads, which *are* seen in both (`calibrate.transform`).

This module is that last step, on arrays and tables, and where the
registration comes from:

* `overlay_table` moves the ``camera_channel == 1`` rows through a
  `ChannelTransform` (chip pixels, read with the table's pixel size by
  `transformation.from_channel_transform`), and writes ``channel`` -- the
  dye -- from ``camera_channel``.  Through `transformation.transform_table`,
  so the move is recorded in the table's metadata like any other.
* `fit_bead_movies` fits bead acquisitions with Gaussian 2D on *two channels*,
  in the arrangement the sample was fitted with; `plugins.registration.
  register_table` registers that table, as *Calibrate transform* does.
* `bead_key` / `cached_transform`: a registration saved beside the beads
  remembers which files it was measured on, so a batch over fifty samples
  registers the bead folder once.

No correction of the bead map from the sample itself.  One from crosstalk pairs (a
molecule and its partner in the other half, in the same frame) was tried and removed,
2026-11-10: on a real two-dye movie (Pan1, M1) its pairs were mostly chance meetings and
badly fitted spots, and in the dense first frames it applied a 100 nm shift that the
real pairs (about 20 nm) did not support; real crosstalk of a 561 dye, some 15 photons
a frame, is not detected at all.  A residual shift between the dyes is measured instead
by cross-correlating the two overlaid channels, *Register layers*.

Coordinates are chip pixels, as in `calibrate.transform`, until the table is
moved.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .locs import Localizations

#: the name the move is recorded under in ``metadata["transformations"]``; a table that has it is
#: not moved again
RECORD_NAME = "overlay channels"
SECONDARY = 1


# ----------------------------------------------------------------- the table
def secondary_rows(locs: Localizations) -> np.ndarray:
    """The localizations of the secondary half (``camera_channel == 1``)."""
    if "camera_channel" not in locs:
        raise ValueError("the table was not fitted as two channels: fit it with channels > "
                         "read: two channels, so that each localization keeps the half it "
                         "was found in (camera_channel)")
    return np.asarray(locs["camera_channel"]) == SECONDARY


def is_overlaid(locs: Localizations) -> bool:
    """True when the table's secondary half has already been moved."""
    return any(record.get("name") == RECORD_NAME
               for record in locs.metadata.get("transformations") or [])


def overlay_table(locs: Localizations, transform, what: str = "") -> Localizations:
    """``locs`` with its secondary half moved onto the main half, and ``channel`` written.

    ``transform`` is a `ChannelTransform` (secondary to main, chip pixels).
    Refused for a table already moved: twice is never what is meant.
    """
    from .locs import table_pixel_sizes
    from .transformation import from_channel_transform, transform_table
    if is_overlaid(locs):
        raise ValueError("the secondary channel has already been put onto the main one; "
                         "undo that first to overlay it again")
    rows = secondary_rows(locs)
    sizes = table_pixel_sizes(locs)
    moved = transform_table(locs, from_channel_transform(transform, sizes), rows,
                            name=RECORD_NAME, what=what or "camera_channel 1")
    columns = dict(moved.columns)
    # the dye: in this workflow, the half it was seen in
    columns["channel"] = np.asarray(locs["camera_channel"]).astype(np.int32)
    return Localizations(columns, moved.metadata)


def check_layout(locs: Localizations, transform) -> Optional[str]:
    """Why the registration may not be this table's, or None: the two record
    where on the chip their channels are, and they should agree."""
    recorded = locs.metadata.get("channels")
    theirs = (transform.geometry or {}).get("channels")
    if not recorded or not theirs:
        return None
    if [list(r) for r in recorded.get("regions", [])] != [list(r) for r in theirs.get("regions", [])]:
        return (f"the table's channels lie at {recorded.get('regions')} on the chip and the "
                f"registration's at {theirs.get('regions')}: fit the sample and the beads in "
                "the same arrangement")
    return None


# ---------------------------------------------------------------- the beads
def bead_files(path) -> List[Path]:
    """The bead acquisitions at ``path``: the movie itself, or every one in a folder."""
    from .calibrate.input import discover_acquisitions
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no bead movies at {path}")
    found = discover_acquisitions([path], recursive=False)
    if not found:
        raise ValueError(f"no image stacks in {path}")
    return [Path(p) for p in found]


def bead_key(paths: Sequence[Path], **settings) -> dict:
    """What a registration measured on these beads depends on: the files as they are now (size
    and time of change) and the settings it was measured with."""
    files = []
    for p in paths:
        p = Path(p)
        st = os.stat(p)
        files.append({"path": str(p.resolve()), "size": int(st.st_size),
                      "mtime_ns": int(st.st_mtime_ns)})
    return {"files": files, "settings": _plain(settings)}


def _plain(value):
    """``value`` as JSON has it, so that a key read back compares equal to one made."""
    import json
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(value, default=str))


def transform_path(path) -> Path:
    """Where the registration of the beads at ``path`` is kept: ``<name>_2ct.h5`` beside the
    movie or the folder -- not inside the folder, which stays images only."""
    path = Path(path)
    stem = path.name.split(".")[0] if path.is_file() else path.name
    return path.parent / f"{stem}_2ct.h5"


def cached_transform(path, key: dict):
    """The registration saved at ``path`` if it was measured on the same beads with the same
    settings (`bead_key`), else None."""
    from .calibrate.transform import load_transform
    path = Path(path)
    if not path.exists():
        return None
    try:
        transform = load_transform(path)
    except (OSError, ValueError, KeyError):
        return None
    return transform if transform.parameters.get("beads") == _plain(key) else None


def fit_bead_movies(paths: Sequence[Path], camera, channels, cutoff: float = 1.7,
                    max_frames: Optional[int] = None, report=None, stop=None) -> Localizations:
    """The bead movies fitted as two channels, one table, a ``filenumber`` per movie so that
    their frames are never paired across movies (`locs.frames_apart`).

    ``camera`` is a fitter's `CameraSettings` (the sample's, so the beads are read alike) and
    ``channels`` its `ChannelsSettings` or an `ArrangementSettings`: where the halves lie.
    Gaussian 2D whatever fitted the sample: a bead's centre is all a 2D registration needs, and a
    bead is bright enough for any model.
    """
    from dataclasses import replace
    from .locs import concat
    from .plugins import Context
    from .plugins.after_fit import AfterFitSettings
    from .plugins.fit import (ChannelsSettings, DetectionSettings, GaussianFit,
                              GaussianFitSettings, OutputSettings, SourceSettings)
    split = ChannelsSettings(split="two channels", layout=channels.layout,
                             main_channel=channels.main_channel,
                             split_position=channels.split_position)
    nothing = AfterFitSettings(astigmatic_z=False, fourpi_z=False, drift="none",
                               assign_colors=False, chain="")
    tables = []
    for k, path in enumerate(paths):
        if report:
            report(f"fitting beads: {Path(path).name} ({k + 1} of {len(paths)})")
        settings = GaussianFitSettings(
            source=SourceSettings(path=str(path), stop=max_frames),
            camera=replace(camera), channels=split,
            detection=DetectionSettings(cutoff_mode="dynamic", cutoff=float(cutoff)),
            output=OutputSettings(save=False, raw_frames=0, show_tags=""))
        settings.finish = nothing
        locs = GaussianFit().run(Context(progress=report, stop=stop), settings).locs
        if locs is None or not len(locs):
            continue
        locs.columns["filenumber"] = np.full(len(locs), k, np.int32)
        tables.append(locs)
    if not tables:
        raise ValueError("no beads found in " + ", ".join(Path(p).name for p in paths))
    out = concat(tables) if len(tables) > 1 else tables[0]
    out.metadata.setdefault("source", str(paths[0]))
    return out


def camera_of(locs: Localizations, given: Optional[dict] = None):
    """The `CameraSettings` to read the beads with: ``given`` (a fitter's own, filled when this
    runs as its step), else what the sample's table recorded of its camera, else nothing --
    the bead files' metadata and the databases, as for any fit."""
    from dataclasses import fields
    from .plugins.fit import CameraSettings
    known = {f.name for f in fields(CameraSettings)}
    values = dict(given or {})
    if not values:
        recorded = locs.metadata.get("camera") or {}
        for name in ("conversion", "offset", "em_on", "emgain", "read_noise_e"):
            if recorded.get(name) is not None:
                values[name] = recorded[name]
        pixel = recorded.get("pixelsize_um")
        if isinstance(pixel, (list, tuple)):
            values["pixelsize_um"], values["pixelsize_y_um"] = float(pixel[0]), float(pixel[1])
        elif pixel is not None:
            values["pixelsize_um"] = float(pixel)
    if "pixelsize_um" not in values and "pixelsize_um" not in (given or {}):
        from .locs import table_pixel_sizes
        try:
            px, py = table_pixel_sizes(locs)
            values["pixelsize_um"] = px / 1e3
            if py != px:
                values["pixelsize_y_um"] = py / 1e3
        except Exception:
            pass
    return CameraSettings(**{k: v for k, v in values.items() if k in known})
