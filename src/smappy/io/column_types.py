"""What a column is stored as once read: float32 by default, and the
exceptions.

A table holds its floating-point columns as float32 (`locs.py`, the
readers): half the memory, and a 24-bit mantissa resolves a position of
1 mm to 0.06 nm, far below any precision.  What float32 cannot hold is a
column whose values are large *and* whose small differences matter:

* **time.**  An hour in float32 resolves 0.24 ms, a day 8 ms -- coarse for
  MINFLUX, whose steps take milliseconds, and an absolute timestamp
  (seconds since 1970) only to 128 s.  So every column whose name says time
  (`is_time`) stays float64, and so do the few names that mean time without
  saying it (`KEEP_FLOAT64`).
* **a large offset.**  Any other float column whose values sit far from
  zero compared with their spread -- a hardware clock, stage coordinates
  added to positions -- loses its differences the same way.  Rather than a
  list that misses the next format's column, `narrow` measures it: a median
  magnitude more than `OFFSET_RATIO` times the 1st-99th percentile range
  keeps float64.
* **integers** stay integers, whatever their width.  A float column whose
  values are all whole numbers and whose name is an id or a frame -- MATLAB
  stores everything as double -- becomes an integer, never float32, which is
  exact only to 2^24 = 16.7 M.  A whole-numbered float with another name
  (photons on a simulated table) stays float: an integer column there would
  change what arithmetic on it means.

The readers call `narrow` on each column as they read it, so a file is never
in memory as float64 all at once: the 57 M localization `_sml.mat` peaked at
20 GB because every column was read as double before any was converted.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..ids import is_id

#: float columns that mean time without the word in their name: MINFLUX's
#: raw ``tim``, a hardware ``timestamp``
KEEP_FLOAT64 = ("tim", "timestamp", "t_s")

#: integer columns by name besides the ids (`ids.is_id`)
INTEGER_NAMES = ("frame", "frame_in_file", "channel", "camera_channel", "iterations",
                 "filenumber", "group_id", "n_in_group")

#: a median magnitude this many times the 1st-99th percentile range keeps a
#: column float64: float32's step at the values is then above 1e-4 of their
#: spread
OFFSET_RATIO = 1000.0


def is_time(name: str) -> bool:
    """Whether column ``name`` holds a time: ``time`` in its name (``time_s``,
    ``on_time_s``, ``dark_time_frames``) or a name in `KEEP_FLOAT64`."""
    return "time" in name.lower() or name in KEEP_FLOAT64


def keeps_float64(name: str, values=None) -> bool:
    """Whether a float column stays float64: by its name, or -- given its
    values -- by `offset_heavy`."""
    if is_time(name):
        return True
    return values is not None and offset_heavy(values)


def offset_heavy(values) -> bool:
    """Whether float32 would lose a column's differences: its values sit far
    from zero compared with their spread (`OFFSET_RATIO`), or are all the same
    large value float32 cannot hold exactly."""
    v = np.asarray(values)
    if v.dtype != np.float64 or v.size == 0:
        return False
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return False
    if finite.size > 200_000:              # the percentiles of a sample suffice
        finite = finite[:: finite.size // 100_000]
    lo, hi = np.percentile(finite, (1, 99))
    middle = float(np.median(np.abs(finite)))
    if hi > lo:
        return middle > OFFSET_RATIO * (hi - lo)
    # one value: harmless if float32 holds it exactly
    return bool(np.float32(middle) != middle) and middle > 2 ** 24


def _whole(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size == values.size and bool(np.all(finite == np.round(finite)))


def narrow(name: str, values) -> np.ndarray:
    """``values`` as column ``name`` is stored: see the module's docstring.

    Integers and booleans are returned as they are; a whole-numbered float id
    or frame becomes int32, or int64 if its values need it; any other float
    becomes float32 unless `keeps_float64` says otherwise.
    """
    v = np.asarray(values)
    kind = v.dtype.kind
    if kind not in "f":
        return v
    if (is_id(name) or name in INTEGER_NAMES) and v.size and _whole(v):
        big = max(abs(float(v.min())), abs(float(v.max())))
        return v.astype(np.int32 if big < 2 ** 31 else np.int64)
    if v.dtype == np.float32:
        return v
    if keeps_float64(name, v):
        return v.astype(np.float64, copy=False)
    return v.astype(np.float32)


def narrow_all(columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """`narrow` on every column of a dict, in place, returned for chaining."""
    for name in list(columns):
        columns[name] = narrow(name, columns[name])
    return columns


def stored_itemsize(name: str, dtype) -> int:
    """The bytes per row column ``name`` of ``dtype`` takes once read: what a
    file's header gives, for an estimate before reading (`io.formats`)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f" and dtype.itemsize > 4:
        # an offset-heavy column is not known before it is read: counted
        # narrow, as nearly every column is
        return 8 if is_time(name) else 4
    return dtype.itemsize
