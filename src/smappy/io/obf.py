"""Imspector's OBF container (`.msr`, `.obf`): the stacks a file holds, and
their data.

The format is Abberior's, documented in AbberiorInstruments/ImspectorDocs
(`docs/fileformat.rst`, MIT) and read here from that description, checked on
files from Imspector m2205 and m2410.  Everything is little-endian.

* A **file header** -- ``"OMAS_BF\\n\\xff\\xff"``, ``u32`` version, ``u64``
  position of the first stack, a description -- and from version 2 a
  ``u64`` pointer to a tag dictionary of the file.
* A **linked list of stacks**.  Each has a fixed header -- the magic
  ``"OMAS_BF_STACK\\n\\xff\\xff"``, version, rank, then for 15 dimensions the
  number of samples (`res`), the length and the offset in SI units (metres
  for x, y, z), the data type, the compression (0 or zlib), the name, a
  description, the data's length on disk and the next stack's position --
  then the data, then a **footer**.  The footer starts with its own size, so
  fields a newer Imspector adds are skipped by jumping that far; after it come
  the axis labels, optional per-axis positions and labels, flush points, the
  **tag dictionary** (string pairs, an empty key ending it: `imspector` is
  Imspector's parameter tree as XML, ~0.5 MB per image; `minflux` a small
  JSON naming a MINFLUX dataset) and, from version 6, chunk positions.

Listing a file reads the headers and footers only (`stacks`): a 2 GB file is
listed without reading its data.  The data are read on request
(`read_stack`), and only for unchunked stacks: Imspector writes chunked
stacks for very large acquisitions, of which no file was at hand, and an
unchecked guess at their layout would be wrong silently.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np

FILE_MAGIC = b"OMAS_BF\n\xff\xff"
STACK_MAGIC = b"OMAS_BF_STACK\n\xff\xff"
MAX_DIMS = 15

# OMAS_DT_*: the stack's sample type
DTYPES = {0x01: "u1", 0x02: "i1", 0x04: "<u2", 0x08: "<i2", 0x10: "<u4",
          0x20: "<i4", 0x40: "<f4", 0x80: "<f8", 0x400: "<u8", 0x800: "<i8"}


@dataclass
class StackInfo:
    """One stack, as its header and footer describe it."""
    index: int
    name: str
    description: str
    res: Tuple[int, ...]               # samples per dimension, x first
    length_m: Tuple[float, ...]        # per dimension, SI (metres for x, y, z)
    offset_m: Tuple[float, ...]
    dtype_code: int
    compression: int                   # 0 none, 1 zlib
    data_pos: int
    data_len: int                      # on disk
    labels: Tuple[str, ...] = ()
    tags: Dict[str, bytes] = field(default_factory=dict)
    samples_written: Optional[int] = None
    n_chunks: int = 0
    version: int = 0

    @property
    def dtype(self) -> Optional[np.dtype]:
        code = DTYPES.get(self.dtype_code)
        return np.dtype(code) if code else None

    @property
    def shape(self) -> Tuple[int, ...]:
        """numpy order, slowest first, with the trailing ones of `res` dropped
        (an image is ``(y, x)``, a z stack ``(z, y, x)``)."""
        res = list(self.res)
        while len(res) > 1 and res[-1] == 1:
            res.pop()
        return tuple(res[::-1])

    def tag(self, key: str) -> str:
        return self.tags.get(key, b"").decode("utf-8", "replace")


def _string(f: BinaryIO) -> bytes:
    n, = struct.unpack("<I", f.read(4))
    return f.read(n)


def _read_header(f: BinaryIO, path) -> Tuple[int, int]:
    magic = f.read(10)
    if magic != FILE_MAGIC:
        raise ValueError(f"{Path(path).name}: not an OBF/MSR file")
    version, first = struct.unpack("<IQ", f.read(12))
    return version, first


def is_obf(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(10) == FILE_MAGIC
    except OSError:
        return False


def _read_stack(f: BinaryIO, pos: int, index: int) -> Tuple[StackInfo, int]:
    f.seek(pos)
    if f.read(16) != STACK_MAGIC:
        raise ValueError(f"no stack at byte {pos}: the file is damaged or cut short")
    version, rank = struct.unpack("<II", f.read(8))
    res = struct.unpack(f"<{MAX_DIMS}I", f.read(4 * MAX_DIMS))[:rank]
    length = struct.unpack(f"<{MAX_DIMS}d", f.read(8 * MAX_DIMS))[:rank]
    offset = struct.unpack(f"<{MAX_DIMS}d", f.read(8 * MAX_DIMS))[:rank]
    (dtype, compression, _level, name_len, descr_len, _reserved, data_len,
     next_pos) = struct.unpack("<IIIIIQQQ", f.read(44))
    name = f.read(name_len).decode("utf-8", "replace")
    description = f.read(descr_len).decode("utf-8", "replace")
    data_pos = f.tell()
    info = StackInfo(index, name, description, tuple(res), tuple(length), tuple(offset),
                     dtype, compression, data_pos, data_len, version=version)
    if version < 1:
        return info, next_pos
    # the footer; its size is its first field, so what a newer version adds
    # is skipped by jumping that far
    footer = data_pos + data_len
    f.seek(footer)
    footer_size, = struct.unpack("<I", f.read(4))
    has_pos = struct.unpack(f"<{MAX_DIMS}I", f.read(4 * MAX_DIMS))
    has_labels = struct.unpack(f"<{MAX_DIMS}I", f.read(4 * MAX_DIMS))
    meta_len = 0
    flush_points = 0
    if version >= 2:
        meta_len, = struct.unpack("<I", f.read(4))
    if version >= 3:
        f.read(80 * 16)                                 # SI units: unused here
    if version >= 4:
        flush_points, _block, _tags_len = struct.unpack("<QQQ", f.read(24))
    if version >= 5:
        f.read(20)                                      # stack ends, min. version
    if version >= 6:
        info.samples_written, info.n_chunks = struct.unpack("<QQ", f.read(16))
    f.seek(footer + footer_size)
    info.labels = tuple(_string(f).decode("utf-8", "replace") for _ in range(rank))
    for dim in range(rank):
        if has_pos[dim]:
            f.read(8 * res[dim])
    for dim in range(rank):
        if has_labels[dim]:
            for _ in range(res[dim]):
                _string(f)
    f.read(meta_len)
    f.read(8 * flush_points)
    if version >= 4:
        while True:
            key = _string(f)
            if not key:
                break
            info.tags[key.decode("utf-8", "replace")] = _string(f)
    return info, next_pos


def stacks(path) -> List[StackInfo]:
    """Every stack of the file, from the headers and footers alone."""
    out: List[StackInfo] = []
    with open(path, "rb") as f:
        _version, pos = _read_header(f, path)
        seen = set()
        while pos and pos not in seen:          # a loop would be a damaged file
            seen.add(pos)
            info, pos = _read_stack(f, pos, len(out))
            out.append(info)
    return out


def read_bytes(path, info: StackInfo) -> bytes:
    """The stack's data, inflated."""
    if info.n_chunks:
        raise ValueError(f"{info.name or 'stack'} {info.index} is stored in chunks, "
                         "which this reader does not read yet")
    with open(path, "rb") as f:
        f.seek(info.data_pos)
        raw = f.read(info.data_len)
    if info.compression == 0:
        return raw
    if info.compression == 1:
        return zlib.decompress(raw)
    raise ValueError(f"unknown compression {info.compression} in stack {info.index}")


def read_stack(path, info: StackInfo) -> np.ndarray:
    """The stack as an array of `StackInfo.shape`; samples an aborted
    acquisition never wrote are zero."""
    dtype = info.dtype
    if dtype is None:
        raise ValueError(f"stack {info.index} has data type 0x{info.dtype_code:x}, "
                         "which is not read")
    raw = read_bytes(path, info)
    n = int(np.prod(info.res))
    data = np.frombuffer(raw, dtype, count=min(n, len(raw) // dtype.itemsize))
    if len(data) < n:
        data = np.concatenate([data, np.zeros(n - len(data), dtype)])
    return data.reshape(info.shape)
