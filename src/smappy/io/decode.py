"""DECODE's HDF5 output (Turaga lab), read.

What ``decode.utils.emitter_io.save_h5`` writes (checked against the
TuragaLab/DECODE master branch, ``EmitterSet.save`` to ``.h5``), and what
``decode.neuralfitter.inference`` saves a fit as:

* ``/data`` -- ``xyz`` (N x 3), ``phot``, ``bg``, ``frame_ix`` (from 0, as
  here), ``id``, ``prob`` (the detection probability), and, where the network
  predicted them, ``xyz_sig``, ``phot_sig``, ``bg_sig`` (standard deviations)
  and ``xyz_cr``, ``phot_cr``, ``bg_cr`` (Cramer-Rao bounds, as variances).
  A quantity DECODE did not have is an empty dataset.
* ``/meta`` attributes -- ``xy_unit`` (``px`` for a fit, or ``nm``) and
  ``px_size`` (DECODE's x and y, nm).  ``z`` is in nm whatever ``xy_unit``
  says: DECODE's own conversion scales x and y only.

**DECODE's x is the frame's first axis, the rows: SMAPpy's y.**  DECODE
indexes a frame (rows, columns) and calls the first index x, so its
``xyz[:, 0]`` and ``px_size[0]`` are SMAPpy's ``y_nm`` and its ``xyz[:, 1]``
``x_nm``.  Read the other way, a fit came in transposed; the example
package's fit, matched to Spline 3D on the same frames, settled it
(docs/decode/poc-plan.md §6, item 4).  What the file cannot say is how the
frames were prepared: DECODE's Fit notebook cropped them to a multiple of 8
(``AutoCenterCrop``, up to 3 pixels off each edge) and, for a model trained
on a SMAP calibration of mirrored EM beads, flipped them, and the positions
are on those frames.
* ``/decode`` attributes -- the version that wrote it.

The precision is ``xyz_sig``: the uncertainty DECODE predicts per
localization is what it reports and filters by, so that is
``x_err_nm`` / ``y_err_nm`` / ``z_err_nm``; the CRLBs are kept beside it as
standard deviations (``x_crlb_nm``, ...).  ``prob`` is kept under its name,
the column to filter DECODE's detections by.  Everything else in ``/data``
keeps its name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..locs import Localizations

# DECODE's axes 0, 1, 2 by SMAPpy's names: its x is the rows
AXES = ("y", "x", "z")


def is_decode(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return isinstance(f.get("data"), h5py.Group) and "xyz" in f["data"] \
                and "meta" in f
    except OSError:
        return False


def _text(value) -> Optional[str]:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def load(path: Path):
    import h5py

    from .formats import FileInfo
    with h5py.File(path, "r") as f:
        meta = dict(f["meta"].attrs)
        version = {k: _text(v) for k, v in f["decode"].attrs.items()} if "decode" in f else {}
        data = {k: v[()] for k, v in f["data"].items() if v.shape is not None}
    unit = _text(meta.get("xy_unit")) or "px"
    size = meta.get("px_size")
    size = None if size is None else np.broadcast_to(np.asarray(size, np.float64).ravel(), 2)
    if unit == "px" and size is not None:
        lateral, suffix = size, "nm"           # converted here, with both sizes
    else:
        lateral, suffix = np.ones(2), ("nm" if unit == "nm" else "pix")

    columns = {}
    xyz = np.asarray(data.pop("xyz"), np.float64).reshape(len(data["frame_ix"]), -1)
    for i, axis in enumerate(AXES[:xyz.shape[1]]):
        scale, s = (lateral[i], suffix) if i < 2 else (1.0, "nm")
        columns[f"{axis}_{s}"] = (xyz[:, i] * scale).astype(np.float32)
        sig = data.get("xyz_sig")
        if sig is not None:
            columns[f"{axis}_err_{s}"] = (np.asarray(sig).reshape(xyz.shape)[:, i]
                                          * scale).astype(np.float32)
        cr = data.get("xyz_cr")
        if cr is not None:
            columns[f"{axis}_crlb_{s}"] = (np.sqrt(np.asarray(cr).reshape(xyz.shape)[:, i])
                                           * scale).astype(np.float32)
    data.pop("xyz_sig", None)
    data.pop("xyz_cr", None)
    renames = {"frame_ix": "frame", "phot": "photons", "bg": "background",
               "phot_sig": "photons_err", "bg_sig": "background_err"}
    for name, values in data.items():
        values = np.asarray(values)
        if name in ("phot_cr", "bg_cr"):
            columns[renames[name[:-3]] + "_crlb"] = np.sqrt(values).astype(np.float32)
            continue
        target = renames.get(name, name)
        if target == "frame":
            columns[target] = values.astype(np.int64)
        elif values.dtype.kind == "f":
            columns[target] = values.astype(np.float32)
        else:
            columns[target] = values
    from ..columns import add_xy_err
    add_xy_err(columns)
    px = float(np.mean(size)) if size is not None else None
    metadata = {"source": str(path),
                "decode": {"xy_unit": unit, "px_size": None if size is None
                           else [float(v) for v in size], **version}}
    if px is not None and suffix == "nm":
        metadata["pixelsize_nm"] = px
        if size[0] != size[1]:
            metadata["pixelsize_nm_xy"] = [float(size[1]), float(size[0])]
    locs = Localizations(columns, metadata)
    return locs, FileInfo(path.name, str(path), "DECODE", pixelsize_nm=px,
                          metadata=metadata["decode"])
