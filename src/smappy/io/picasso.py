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
            return [d for d in yaml.safe_load_all(sidecar.read_text()) if isinstance(d, dict)]
        except yaml.YAMLError:
            pass
    return []


def from_info(info: Sequence[Dict], key: str):
    """The first value of ``key`` in the steps' metadata, as Picasso's
    ``lib.get_from_metadata`` reads it."""
    return next((d[key] for d in info if key in d), None)


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
    which the positions are on, or else as much as the positions need."""
    camera = locs.metadata.get("camera") or {}
    roi = camera.get("roi") if isinstance(camera, dict) else None
    width = int(np.ceil(np.nanmax(x_pix))) + 1 if len(x_pix) else 1
    height = int(np.ceil(np.nanmax(y_pix))) + 1 if len(y_pix) else 1
    if roi is not None and len(roi) == 4:
        width = max(width, int(roi[0]) + int(roi[2]))
        height = max(height, int(roi[1]) + int(roi[3]))
    return width, height


def records(locs: Localizations, group: Optional[str] = None
            ) -> Tuple[np.ndarray, Dict, int, List[str]]:
    """The table as Picasso's records: ``(records, info, dropped, filled)``.

    ``group`` names an integer column written as Picasso's ``group`` (its
    picks), ``cluster_id`` say.  ``dropped`` counts the rows Picasso would
    refuse; ``filled`` the fields the table has no column for, written as 0.
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
            raise ValueError(f"no column {group!r} to write as Picasso's group")
        fields["group"] = np.asarray(locs[group]).astype(np.int32)

    width, height = _extent(locs, x, y)
    keep = (x < width) & (y < height)
    for name, values in fields.items():
        if values.dtype.kind == "f":
            keep &= np.isfinite(values)
        if name in ("x", "y", "lpx", "lpy", "lpz", "photons", "ellipticity", "sx", "sy"):
            keep &= values >= 0
    dtype = [(name, np.uint32 if name == "frame" else np.int32 if name == "group"
              else np.float32) for name in fields]
    out = np.zeros(int(keep.sum()), dtype=dtype)
    for name, values in fields.items():
        out[name] = values[keep]
    frames = locs.metadata.get("n_frames") or (int(np.max(locs["frame"])) + 1 if len(locs) else 0)
    info = {"Generated by": "smappy", "Width": width, "Height": height,
            "Frames": int(frames), "Pixelsize": px}
    return out, info, int(len(locs) - keep.sum()), filled


def save(locs: Localizations, path: Path, metadata=None, group: Optional[str] = None,
         report: Optional[Callable[[str], None]] = None) -> Path:
    """Write a Picasso file and its ``.yaml``; ``report`` hears what was
    dropped or filled in."""
    import h5py
    import yaml
    rows, info, dropped, filled = records(locs, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=rows)
        f.create_dataset("metadata", data=json.dumps([info]))
    path.with_suffix(".yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    if report is not None:
        if dropped:
            report(f"{dropped} localizations left out: Picasso refuses a NaN, a "
                   "negative position, width or precision, or a position off the image")
        if filled:
            report(f"no column for {', '.join(filled)}: written as 0")
    return path
