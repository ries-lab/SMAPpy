"""Picasso's HDF5 localization files (Jungmann lab), read and written.

What Picasso (``picassosr``, checked against 0.11.3's ``io.save_locs`` and
``lib.ensure_sanity``) puts in a file:

* ``/locs`` -- **one** compound dataset, a record per localization.  That is
  how it is told from smappy's own HDF5, whose ``/locs`` is a group of
  columns.  ``x``, ``y``, ``sx``, ``sy``, ``lpx``, ``lpy`` are in **camera
  pixels**; ``z`` and ``lpz`` in **nm**; ``frame`` counts from 0, as here.
* the metadata -- a list of dicts, one per processing step, as JSON in a
  ``/metadata`` dataset since Picasso 0.9 and as a ``.yaml`` beside the file
  before that (and still, by default).  ``Width``, ``Height`` and ``Frames``
  are required when Picasso opens a file, ``Pixelsize`` (nm) is what turns
  pixels into nm.

Reading keeps every field: the ones smappy has a name for get it, the rest
keep Picasso's (``net_gradient``, ``ellipticity``, ``d_zcalib``, a linked
table's ``len`` and ``n``).  ``group``, Picasso's *pick*, is read as
``pick_id``: it is an id column -- every per-region tool runs on picks -- and
``group`` would read as smappy's blink grouping (``group_id``), which it is
not.

Writing produces the fields Picasso's Render needs.  Picasso's own loader
drops every row with a NaN anywhere, a negative position, width or
precision, or a position outside ``Width`` x ``Height``; the writer drops
them itself and says how many, rather than let them vanish in Picasso.

A table of merged blinks is written as Picasso writes a linked one
(``postprocess.link``): ``len``, the event's frames from its first to its
last, dark frames included, and ``n``, its localizations, with
``photon_rate``, the photons per localization.  Localizations have none of
the three, as in Picasso's unlinked files.  The grouped table does not keep
a blink's last frame, so it is read from the localizations
(`group.last_frames`).

The metadata list is Picasso's record of what was done to the file, one
entry per step, and Picasso reads a key from the **last** entry that has it
(``lib.get_from_metadata``).  A file that came from Picasso keeps its list,
and this program's entry is appended -- with the image's size and pixel
size, which then win, and the steps the session's log has -- as Picasso
appends its own; so a file that goes Picasso -> smappy -> Picasso keeps its
history.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations

#: Picasso field -> smappy column; pixel columns are converted to nm on the
#: way in by `formats.with_nm`, with the file's pixel size
READ_NAMES = {
    "x": "x_pix", "y": "y_pix", "lpx": "x_err_pix", "lpy": "y_err_pix",
    "sx": "sigma_pix", "sy": "sigma_y_pix", "bg": "background",
    "z": "z_nm", "lpz": "z_err_nm", "group": "pick_id",
}
INTEGER_FIELDS = ("frame", "group", "n", "len")


def is_picasso(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return isinstance(f.get("locs"), h5py.Dataset)
    except OSError:
        return False


def read_info(path: Path) -> List[Dict]:
    """The metadata list: the embedded ``/metadata`` JSON, or the ``.yaml``
    beside the file, or nothing."""
    import h5py
    with h5py.File(path, "r") as f:
        raw = f["metadata"][()] if "metadata" in f else None
    if raw is not None:
        try:
            info = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            return [d for d in info if isinstance(d, dict)]
        except (ValueError, TypeError):
            pass
    sidecar = path.with_suffix(".yaml")
    if sidecar.exists():
        import yaml
        try:
            # Picasso writes its YAML in the locale's encoding, cp1252 on
            # Windows: a non-ASCII path in it is replaced, the numbers survive
            text = sidecar.read_text(encoding="utf-8", errors="replace")
            return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
        except yaml.YAMLError:
            pass
    return []


def from_info(info: Sequence[Dict], key: str):
    """The value of ``key`` in the steps' metadata as Picasso's
    ``lib.get_from_metadata`` reads it: from the last step with one."""
    return next((d[key] for d in reversed(list(info)) if d.get(key)), None)


def load(path: Path):
    import h5py

    from .formats import FileInfo
    with h5py.File(path, "r") as f:
        records = f["locs"][...]
    columns = {}
    for name in records.dtype.names:
        values = records[name]
        target = READ_NAMES.get(name, name)
        if name in INTEGER_FIELDS:
            columns[target] = values.astype(np.int64 if name == "frame" else np.int32)
        elif values.dtype.kind == "f":
            columns[target] = values.astype(np.float32)
        else:
            columns[target] = values
    info = read_info(path)
    px = from_info(info, "Pixelsize")
    px = float(px) if px else None
    locs = Localizations(columns, {"source": str(path), "picasso": info})
    return locs, FileInfo(path.name, str(path), "Picasso", pixelsize_nm=px,
                          metadata={"picasso": info})


# ------------------------------------------------------------------- write

def _extent(locs: Localizations, x_pix: np.ndarray, y_pix: np.ndarray) -> Tuple[int, int]:
    """The camera image Picasso is told about: the chip region the fit read,
    which the positions are on, or the image a Picasso file said it came
    from, or else as much as the positions need.  This program's entry is
    the last of the list, which Picasso reads first, so it must not say less
    than the file it came from did."""
    camera = locs.metadata.get("camera") or {}
    roi = camera.get("roi") if isinstance(camera, dict) else None
    width = int(np.ceil(np.nanmax(x_pix))) + 1 if len(x_pix) else 1
    height = int(np.ceil(np.nanmax(y_pix))) + 1 if len(y_pix) else 1
    if roi is not None and len(roi) == 4:
        width = max(width, int(roi[0]) + int(roi[2]))
        height = max(height, int(roi[1]) + int(roi[3]))
    before = _before(locs)
    width = max(width, int(from_info(before, "Width") or 0))
    height = max(height, int(from_info(before, "Height") or 0))
    return width, height


def _before(locs: Localizations) -> List[Dict]:
    """The metadata list the table's Picasso file came with, if it did."""
    before = locs.metadata.get("picasso")
    return [d for d in before if isinstance(d, dict)] if isinstance(before, list) else []


def _linked(locs: Localizations, events: Optional[Localizations]
            ) -> Tuple[Optional[Dict[str, np.ndarray]], bool]:
    """Picasso's fields of a linked table -- ``len``, ``n``, ``photon_rate``
    -- and whether ``len`` had to be guessed; None for localizations.

    A table merged here (``metadata["grouped"]``) has its blinks' sizes in
    `n_in_group`; their length needs their last frame, from ``events``, the
    localizations they were merged from.  A blink not found there, or no
    ``events`` at all, is given ``len`` = ``n``, which is right only without
    dark frames.  A table read from a linked Picasso file keeps the ``len``
    and ``n`` it came with.
    """
    from ..group import last_frames
    guessed = False
    if locs.metadata.get("grouped") and "n_in_group" in locs:
        n = np.asarray(locs["n_in_group"], np.int64)
        first = np.asarray(locs["frame"], np.int64)
        last = (last_frames(events, locs["group_id"])
                if events is not None and "group_id" in locs
                else np.full(len(n), -1, np.int64))
        found = last >= first
        guessed = not found.all()
        length = np.where(found, last - first + 1, n)
    elif "len" in locs and "n" in locs:
        n, length = np.asarray(locs["n"], np.int64), np.asarray(locs["len"], np.int64)
    else:
        return None, False
    out = {"len": length, "n": n}
    if "photons" in locs:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["photon_rate"] = np.asarray(locs["photons"], np.float64) / n
    return out, guessed


def _group_ids(values) -> np.ndarray:
    """An id column as Picasso's ``group``: whole numbers.  On merged blinks
    an id written as a recipe is a float (its reduction); a row with none is
    -1 there."""
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = np.where(np.isfinite(values), np.rint(values), -1)
    return values.astype(np.int32)


def records(locs: Localizations, group: Optional[str] = None,
            events: Optional[Localizations] = None
            ) -> Tuple[np.ndarray, Dict, int, List[str]]:
    """The table as Picasso's records: ``(records, info, dropped, filled)``.

    ``group`` names an id column written as Picasso's ``group`` (its picks),
    ``cluster_id`` say.  ``events`` are the localizations a table of merged
    blinks was made from, for each blink's last frame (`_linked`).
    ``dropped`` counts the rows Picasso would refuse; ``filled`` the fields
    the table has no column for, written as 0 -- and ``len``, when it was
    taken to be ``n``.
    """
    from ..locs import in_pixels, table_pixel_sizes
    px = float(np.mean(table_pixel_sizes(locs)))      # raises with the reason
    x, y = in_pixels(locs, "x_pix", "y_pix")
    filled: List[str] = []

    def pixels(*names):
        for name in names:
            nm = name.replace("_pix", "_nm")
            if nm in locs or name in locs:
                return in_pixels(locs, name)
        return None

    def column(*names):
        return next((np.asarray(locs[n], np.float64) for n in names if n in locs), None)

    fields = {
        "frame": np.asarray(locs["frame"]).astype(np.uint32),
        "x": x, "y": y,
        "photons": column("photons"),
        "sx": pixels("sigma_x_pix", "sigma_pix"),
        "sy": pixels("sigma_y_pix", "sigma_pix"),
        "bg": column("background"),
        "lpx": pixels("x_err_pix", "xy_err_pix"),
        "lpy": pixels("y_err_pix", "xy_err_pix"),
    }
    for name in ("photons", "sx", "sy", "bg", "lpx", "lpy"):
        if fields[name] is None:
            filled.append(name)
            fields[name] = np.zeros(len(locs))
    big, small = np.maximum(fields["sx"], fields["sy"]), np.minimum(fields["sx"], fields["sy"])
    with np.errstate(divide="ignore", invalid="ignore"):
        fields["ellipticity"] = np.where(big > 0, (big - small) / big, 0.0)
    fields["net_gradient"] = np.zeros(len(locs))
    if "z_nm" in locs:
        fields["z"] = np.asarray(locs["z_nm"], np.float64)
        fields["lpz"] = column("z_err_nm")
        if fields["lpz"] is None:
            filled.append("lpz")
            fields["lpz"] = np.zeros(len(locs))
    if group:
        if group not in locs:
            raise ValueError(f"no attribute {group!r} to write as Picasso's group")
        fields["group"] = _group_ids(locs[group])
    linked, guessed = _linked(locs, events)
    if linked is not None:
        fields.update(linked)
        if guessed:
            filled.append("len")

    width, height = _extent(locs, x, y)
    keep = (x < width) & (y < height)
    for name, values in fields.items():
        if values.dtype.kind == "f":
            keep &= np.isfinite(values)
        if name in ("x", "y", "lpx", "lpy", "lpz", "photons", "ellipticity", "sx", "sy"):
            keep &= values >= 0
    integer = {"frame": np.uint32, "group": np.int32, "len": np.int32, "n": np.int32}
    dtype = [(name, integer.get(name, np.float32)) for name in fields]
    out = np.zeros(int(keep.sum()), dtype=dtype)
    for name, values in fields.items():
        out[name] = values[keep]
    frames = locs.metadata.get("n_frames") or (int(np.max(locs["frame"])) + 1 if len(locs) else 0)
    frames = max(int(frames), int(from_info(_before(locs), "Frames") or 0))
    info = {"Generated by": "smappy", "Width": width, "Height": height,
            "Frames": int(frames), "Pixelsize": px}
    return out, info, int(len(locs) - keep.sum()), filled


def _plain(value):
    """``value`` as JSON has it -- tuples as lists, numpy numbers as numbers
    -- so that the JSON and the YAML writer both take it."""
    return json.loads(json.dumps(value, default=lambda v: v.item()
                                 if hasattr(v, "item") else str(v)))


def steps(history: Optional[Sequence[Dict]]) -> List[Dict]:
    """The session's log as the steps of this program's metadata entry: the
    loads and what changed the table, with their settings."""
    return [{k: e[k] for k in ("time", "what", "text", "settings") if k in e}
            for e in history or ()
            if isinstance(e, dict) and (e.get("changed") or e.get("what") == "load")]


def metadata_list(locs: Localizations, entry: Dict,
                  history: Optional[Sequence[Dict]] = None) -> List[Dict]:
    """What the file's metadata says: the list a Picasso file came with
    (`load` keeps it in ``metadata["picasso"]``) and this program's entry
    appended, with its steps."""
    from .. import __version__
    entry = {**entry, "Generated by": f"smappy {__version__}"}
    done = steps(history)
    if done:
        entry["Steps"] = done
    return _plain(_before(locs) + [entry])


def save(locs: Localizations, path: Path, metadata=None, group: Optional[str] = None,
         report: Optional[Callable[[str], None]] = None,
         events: Optional[Localizations] = None) -> Path:
    """Write a Picasso file and its ``.yaml``; ``report`` hears what was
    dropped or filled in.  The steps in the metadata are those of
    ``metadata["history"]``, else of the table's own; ``events`` as for
    `records`."""
    import h5py
    import yaml
    rows, entry, dropped, filled = records(locs, group, events)
    history = (metadata or {}).get("history")
    info = metadata_list(locs, entry, locs.metadata.get("history")
                         if history is None else history)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=rows)
        f.create_dataset("metadata", data=json.dumps(info))
    path.with_suffix(".yaml").write_text(yaml.safe_dump_all(info, sort_keys=False),
                                         encoding="utf-8")
    if report is not None:
        if dropped:
            report(f"{dropped} localizations left out: Picasso refuses a NaN, a "
                   "negative position, width or precision, or a position off the image")
        zeros = [f for f in filled if f != "len"]
        if zeros:
            report(f"no attribute for {', '.join(zeros)}: written as 0")
        if "len" in filled:
            report("len written as n for blinks whose localizations were not "
                   "given: right only without dark frames")
    return path
