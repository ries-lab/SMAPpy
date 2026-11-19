"""Writing Abberior `.msr` files: the OBF container with MINFLUX
measurements (and images) inside, in the layout Imspector m2410 writes.

Not a general writer, and not for anyone's real data: SMAPpy never writes
an `.msr` it read.  It exists so that simulated MINFLUX measurements can be
opened the way a measurement from the instrument is -- the tests of
`io.msr` (`tests/test_msr.py`, where it began) and the MINFLUX tutorial,
which needs a file anyone can rebuild (`simulate.minflux.write_msr`).

The layout: an OBF file (`io.obf`) whose stacks are images, or a
measurement as an `MFXDTA` archive (`io.zarr_lite`) of zarr v2 arrays --
the records at ``zarr/mfx``, its sequence and instrument in
``zarr/mfx/.zattrs``, the ROI in ``zarr/.zattrs``, beads at
``zarr/grd/mbm/points`` -- stored as the bytes of a stack tagged
``minflux``.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import blosc, obf, zarr_lite

#: Imspector m2410's record, as the zarr array's dtype spec
M2410 = [["vld", "|b1"], ["fnl", "|b1"], ["bot", "|b1"], ["eot", "|b1"], ["sta", "|u1"],
         ["tim", "<f8"], ["tid", "<u4"], ["gri", "<u4"], ["thi", "|u1"], ["sqi", "|u1"],
         ["itr", "<i4"], ["loc", "<f8", [3]], ["lnc", "<f8", [3]], ["eco", "<u4"],
         ["ecc", "<u4"], ["efo", "<f4"], ["efc", "<f4"], ["fbg", "<f4"], ["cfr", "<f2"],
         ["dcr", "<f2", [2]]]


def _string(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _stack(name, res, length, offset, dtype_code, data, tags, next_pos=0, compress=True):
    """One OBF stack, version 7, as bytes; `next_pos` patched in later."""
    rank = len(res)
    pad = lambda v, fmt: struct.pack(f"<15{fmt}", *(list(v) + [0] * (15 - len(v))))
    disk = zlib.compress(data) if compress else data
    head = (obf.STACK_MAGIC + struct.pack("<II", 7, rank) + pad(res, "I")
            + pad(length, "d") + pad(offset, "d")
            + struct.pack("<IIIIIQQQ", dtype_code, int(compress), 1, len(name), 0, 0,
                          len(disk), next_pos) + name.encode())
    fixed = (struct.pack("<15I", *[0] * 15) * 2 + struct.pack("<I", 0) + bytes(1280)
             + struct.pack("<QQQ", 0, 0, 0) + bytes(20) + struct.pack("<QQ", 0, 0))
    extra = bytes(60)                       # a newer version's fields: skipped
    footer = struct.pack("<I", 4 + len(fixed) + len(extra)) + fixed + extra
    after = b"".join(_string(b"x") for _ in range(rank))
    for key, value in tags.items():
        after += _string(key.encode()) + _string(value.encode())
    after += _string(b"")
    return head, disk + footer + after


def write_obf(path, stacks) -> None:
    """``stacks``: (name, res, length_m, offset_m, dtype code, bytes, tags)."""
    out = bytearray(obf.FILE_MAGIC + struct.pack("<IQ", 2, 0) + struct.pack("<I", 0)
                    + struct.pack("<Q", 0))
    struct.pack_into("<Q", out, 14, len(out))
    previous = None
    for spec in stacks:
        start = len(out)
        if previous is not None:            # the last stack's next pointer
            struct.pack_into("<Q", out, previous, start)
        head, rest = _stack(*spec)
        previous = start + len(head) - 8 - len(spec[0].encode())
        out += head + rest
    Path(path).write_bytes(bytes(out))


def archive(files: dict, t: int = 1781190507) -> bytes:
    """An MFXDTA archive of ``{path: bytes}``."""
    head = b"MFXDTA\0" + struct.pack("<II", 3, t)
    out = bytearray(head + bytes(zarr_lite.ARCHIVE_HEADER - len(head)))
    dirs = sorted({str(Path(p).parent) for p in files} - {"."})
    for d in dirs:
        out += struct.pack("<Q", len(d)) + d.encode() + b"\x01"
    for p, data in files.items():
        out += struct.pack("<Q", len(p)) + p.encode() + b"\x00" + struct.pack("<Q", len(data)) + data
    return bytes(out + struct.pack("<Q", 0))


def zarr_array(node, rec, spec, chunk=7, compressor="zlib") -> dict:
    """A one-dimensional zarr v2 array of records, its chunks compressed."""
    files = {f"{node}/.zarray": json.dumps({
        "chunks": [chunk], "shape": [len(rec)], "dtype": spec, "order": "C",
        "filters": None, "fill_value": None, "zarr_format": 2,
        "compressor": {"id": "zlib", "level": 1} if compressor == "zlib" else
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}}).encode()}
    for k in range(0, len(rec), chunk):
        part = np.zeros(chunk, rec.dtype)          # the last chunk padded
        part[:len(rec[k:k + chunk])] = rec[k:k + chunk]
        raw = part.tobytes()
        files[f"{node}/{k // chunk}"] = (zlib.compress(raw) if compressor == "zlib" else
                                         blosc_memcpy(raw))
    return files


def blosc_memcpy(raw: bytes) -> bytes:
    """A blosc frame that stores its bytes as they are."""
    return struct.pack("<BBBBIII", 2, 1, blosc.BLOSC_MEMCPYED, 1, len(raw), len(raw),
                       len(raw) + 16) + raw


def as_records(rec: np.ndarray) -> np.ndarray:
    """``rec`` in the m2410 dtype, field by field (missing fields zero)."""
    out = np.zeros(len(rec), zarr_lite._dtype(M2410))
    for name in out.dtype.names:
        if name in rec.dtype.names:
            out[name] = rec[name]
    return out


def measurement_stack(label: str, rec: np.ndarray, attrs: dict,
                      roi_m: Optional[Sequence] = None, beads: Optional[dict] = None,
                      chunk: int = 7):
    """A MINFLUX measurement as an OBF stack: the records ``rec`` (m2410),
    ``attrs`` its ``zarr/mfx/.zattrs`` (``acquisition_date``,
    ``measurement.threads[i].sequences``, ``lasers``, ...), ``roi_m`` the
    ROI's corners in metres, ``beads`` the MBM part as ``{"points": records,
    "spec": dtype spec, "points_by_gri": ..., "used": [...]}`` -- or, two
    colours, ``{"grids": {name: {"thread", "corrects", "points", "spec",
    "points_by_gri", "used"}}}``, a grid per colour (``attrs`` then names
    them in ``measurement.grids``)."""
    corners = roi_m if roi_m is not None else [[0.0, 0.0], [1e-5, 1e-5]]
    files = {"zarr/.zattrs": json.dumps({"rois": [{"corners": [list(c) for c in corners],
                                                   "type": "ROI", "name": ""}]}).encode()}
    files.update(zarr_array("zarr/mfx", as_records(rec), M2410, chunk=chunk))
    files["zarr/mfx/.zattrs"] = json.dumps(attrs).encode()
    if beads and beads.get("grids"):
        # two colours: a grid per colour, measured by its own thread and
        # correcting another, as Imspector m2410 writes them (grd/mbm_0, ...)
        for name, g in beads["grids"].items():
            node = f"zarr/grd/{name}"
            files.update(zarr_array(f"{node}/points", g["points"], g["spec"], chunk=10,
                                    compressor="blosc"))
            files[f"{node}/.zattrs"] = json.dumps(
                {"type": "mbm", "controlling_thi": g["thread"], "watching_thi": g["thread"],
                 "points": "./points"}).encode()
            files[f"{node}/points/.zattrs"] = json.dumps(
                {"points_by_gri": g.get("points_by_gri", {})}).encode()
            files[f"zarr/mbm/{name}/.zattrs"] = json.dumps(
                {"correcting_thi": list(g["corrects"]), "grid": name,
                 "used": g.get("used", [])}).encode()
        files["zarr/mbm/.zattrs"] = json.dumps({"version": "3"}).encode()
    elif beads:
        files.update(zarr_array("zarr/grd/mbm/points", beads["points"], beads["spec"],
                                chunk=10, compressor="blosc"))
        files["zarr/grd/mbm/points/.zattrs"] = json.dumps(
            {"points_by_gri": beads.get("points_by_gri", {})}).encode()
        files["zarr/mbm/.zattrs"] = json.dumps({"used": beads.get("used", [])}).encode()
    blob = archive(files)
    side = int(np.ceil(np.sqrt(len(blob))))
    tag = json.dumps({"type": "data", "label": label, "did": f"did-{label}"})
    return ("", (side, side, 1, 1), (1.0, 1.0, 1.0, 1.0), (0.0,) * 4, 0x01, blob,
            {"minflux": tag, "imspector": "<root/>"})


def image_stack(name: str, data: np.ndarray, pixel_m: float, origin_m=(0.0, 0.0)):
    """A 2D image (int16) as an OBF stack, placed in metres."""
    ny, nx = data.shape
    return (name, (nx, ny, 1, 1), (nx * pixel_m, ny * pixel_m, 1.0, 1.0),
            (origin_m[0], origin_m[1], -0.5, 0.0), 0x08,
            np.asarray(data, "<i2").tobytes(), {"imspector": "<root/>"})
