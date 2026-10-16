"""Just enough zarr v2 to read Imspector's MINFLUX archives, from any mapping
of path -> bytes.

Inside an `.msr`, a MINFLUX measurement is one OBF stack of bytes which,
inflated, is an **MFXDTA archive**: a 147-byte header (``MFXDTA\\0``, a
``u32`` format version -- 2 in m2205, 3 in m2410 -- a ``u32`` Unix time,
zeros), then entries -- ``u64``
length, a path, a ``u8`` kind (1 a directory; 0 a file, then a ``u64`` size and
the bytes) -- until a zero length.  The files are a complete zarr v2 tree, the
same one Imspector's *ZARR* export and specpy's ``unpack`` write to disk
(m2205 with Windows separators, m2410 with ``/``).  So the archive, an
unpacked folder and a zipped one are all the same thing to this module: a
mapping of path to bytes (`archive_files`, `folder_files`).

What is read of zarr: an array's ``.zarray`` -- the shape, the chunks, the
dtype including the nested structured dtypes Imspector writes, the fill value,
order C, and a compressor that is zlib, blosc (`io.blosc`) or none --
its chunks, and any ``.zattrs`` as JSON.  zarr-python is not needed, and zarr
v3, which Abberior says it does not write, is not read.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

ARCHIVE_MAGIC = b"MFXDTA"
ARCHIVE_HEADER = 147


def archive_files(blob: bytes) -> Dict[str, bytes]:
    """An MFXDTA archive's files, by their path with ``/`` separators."""
    if not blob.startswith(ARCHIVE_MAGIC):
        raise ValueError("not a MINFLUX data archive (no MFXDTA header)")
    out: Dict[str, bytes] = {}
    o, n = ARCHIVE_HEADER, len(blob)
    while o + 8 <= n:
        length, = struct.unpack_from("<Q", blob, o)
        if length == 0:
            break
        if length > 4096 or o + 8 + length + 1 > n:
            raise ValueError(f"damaged MINFLUX archive at byte {o}")
        name = blob[o + 8:o + 8 + length].decode("utf-8", "replace").replace("\\", "/")
        o += 8 + length
        kind = blob[o]
        o += 1
        if kind == 1:                       # a directory: nothing follows
            continue
        size, = struct.unpack_from("<Q", blob, o)
        o += 8
        out[name] = blob[o:o + size]
        o += size
    return out


def archive_time(blob: bytes) -> Optional[int]:
    """The Unix time in the archive's header, the measurement's start."""
    if not blob.startswith(ARCHIVE_MAGIC) or len(blob) < 15:
        return None
    t, = struct.unpack_from("<I", blob, 11)       # after "MFXDTA\0" and the version
    return int(t) or None


def folder_files(folder) -> Dict[str, bytes]:
    """An unpacked zarr tree (Imspector's export, specpy's ``unpack``)."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in folder.rglob("*") if p.is_file()}


def _dtype(spec) -> np.dtype:
    """zarr's dtype: a string, or a list of [name, type(, shape)] fields."""
    if isinstance(spec, str):
        return np.dtype(spec)
    fields = []
    for f in spec:
        sub = _dtype(f[1]) if isinstance(f[1], list) else np.dtype(f[1])
        fields.append((f[0], sub, tuple(f[2])) if len(f) > 2 else (f[0], sub))
    return np.dtype(fields)


def attrs(files: Mapping[str, bytes], node: str) -> dict:
    """A group's or array's ``.zattrs``; {} when there are none."""
    raw = files.get(f"{node}/.zattrs" if node else ".zattrs")
    return json.loads(raw) if raw else {}


def has_array(files: Mapping[str, bytes], node: str) -> bool:
    return f"{node}/.zarray" in files


def array_info(files: Mapping[str, bytes], node: str) -> dict:
    return json.loads(files[f"{node}/.zarray"])


def _decompress(raw: bytes, compressor: Optional[dict]) -> bytes:
    if not compressor:
        return raw
    kind = compressor.get("id")
    if kind == "zlib":
        return zlib.decompress(raw)
    if kind == "blosc":
        from .blosc import decode
        return decode(raw)
    raise ValueError(f"zarr compressor {kind!r} is not read")


def read_array(files: Mapping[str, bytes], node: str) -> np.ndarray:
    """A 1D zarr array, its chunks joined and the last one's padding cut.

    One dimension is all Imspector writes (a record per row); more would
    need the chunk grid, which is not worth guessing at without a file.
    """
    info = array_info(files, node)
    dtype = _dtype(info["dtype"])
    shape, chunks = info["shape"], info["chunks"]
    if len(shape) != 1:
        raise ValueError(f"{node}: a {len(shape)}D zarr array; only 1D ones are read")
    if info.get("order", "C") != "C" or info.get("filters"):
        raise ValueError(f"{node}: order {info.get('order')} or filters are not read")
    n, step = int(shape[0]), int(chunks[0])
    out = np.zeros(n, dtype)
    fill = info.get("fill_value")
    if isinstance(fill, str) and dtype.names:     # a structured fill, base64
        try:
            out[:] = np.frombuffer(base64.b64decode(fill), dtype, count=1)[0]
        except ValueError:
            pass
    elif fill is not None and not dtype.names:
        out[:] = fill
    for k in range((n + step - 1) // step if step else 0):
        raw = files.get(f"{node}/{k}")
        if raw is None:                            # never written: the fill
            continue
        data = np.frombuffer(_decompress(raw, info.get("compressor")), dtype)
        start = k * step
        take = min(len(data), n - start)
        out[start:start + take] = data[:take]
    return out
