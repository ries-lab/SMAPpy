"""Column names that changed, and the lateral precision derived from x and y.

Precisions follow one rule, ``<quantity>_err_<unit>``: ``x_err_nm``,
``z_err_nm``, ``photons_err``.  The lateral precision is ``xy_err_nm``, the RMS
of the two coordinate errors.  It used to be ``loc_precision_nm``, and the
axial one ``loc_precision_z_nm`` -- which SMAPpy's own 3D fit never wrote (it
writes ``z_err_nm``), so the renderer and Statistics, looking for the old
name, found no z precision after a SMAPpy fit.

Files, workspaces and chains saved before the rename still name the old
columns.  They are renamed here, and only here, as they are read in:
:func:`current` is applied to a table's columns and metadata, to the GUI
state and tool results stored with it, and to workspace, chain and batch
files.  Only the new names are ever written.

Kept free of heavy imports: the chain and workspace readers use it.
"""
from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np

#: old name -> new name.  The only place the old names are allowed to appear.
RENAMED: Dict[str, str] = {
    "loc_precision_nm": "xy_err_nm",
    "loc_precision_pix": "xy_err_pix",
    "loc_precision_z_nm": "z_err_nm",
}

# A whole word, or one with a per-channel suffix (``_ch1``): an old name
# inside an expression -- ``(loc_precision_nm < 25) & ...`` -- is renamed too,
# and nothing that merely contains one as a substring is.
_OLD = re.compile(r"\b(" + "|".join(sorted(RENAMED, key=len, reverse=True))
                  + r")(?=_ch\d+\b|\b)")


def current_name(text: str) -> str:
    """``text`` with every old column name in it replaced by the new one."""
    return _OLD.sub(lambda m: RENAMED[m.group(1)], text)


def current(obj: Any) -> Any:
    """``obj`` -- a dict, list, string or anything else -- with old names renamed.

    Walks nested dicts and lists, renaming keys and strings; arrays and
    numbers are returned as they are.  A dict that already has the new name
    keeps its own value rather than being overwritten by the old one.
    """
    if isinstance(obj, str):
        return current_name(obj)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            new = current_name(key) if isinstance(key, str) else key
            if new != key and new in obj:
                continue
            out[new] = current(value)
        return out
    if isinstance(obj, list):
        return [current(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(current(v) for v in obj)
    return obj


def xy_err(x_err, y_err) -> np.ndarray:
    """The lateral precision: the RMS of the two coordinate errors."""
    x_err = np.asarray(x_err, np.float64)
    y_err = np.asarray(y_err, np.float64)
    return np.sqrt((x_err ** 2 + y_err ** 2) / 2).astype(np.float32)


def add_xy_err(columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Set ``xy_err_<unit>`` from ``x_err_<unit>`` and ``y_err_<unit>``, in place.

    Wherever the two coordinate errors exist the lateral precision is derived
    from them rather than carried on its own, so the two can never disagree --
    after grouping (which combines each error by its own rule), after a
    conversion to nm with pixels that are not square.  A table with only a
    lateral precision (SMAP's ``locprecnm``) keeps the one it has.
    """
    for unit in ("nm", "pix"):
        x, y = columns.get(f"x_err_{unit}"), columns.get(f"y_err_{unit}")
        if x is not None and y is not None:
            columns[f"xy_err_{unit}"] = xy_err(x, y)
    return columns
