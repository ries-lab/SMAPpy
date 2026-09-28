"""The lateral precision, derived from the x and y errors.

Precisions follow one rule, ``<quantity>_err_<unit>``: ``x_err_nm``,
``z_err_nm``, ``photons_err``.  The lateral precision is ``xy_err_nm``, the RMS
of the two coordinate errors.

Kept free of heavy imports: the table and grouping use it.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


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
