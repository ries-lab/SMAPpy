"""Id columns: which localizations belong together, and the value for none.

An id column names what a localization belongs to -- a merged blink
(`group_id`), a cluster, a track, a bead, a pick, a MINFLUX trace (`tid`).
Every value is an id, 0 included, and **-1 is "in none"**: noise left out by
DBSCAN, a localization in no track, a row left out of merging.

Until 2026-11 it was 0 that meant none, and every reader of an id column
skipped it.  That is right for ids SMAPpy numbers from 1, and wrong for ids
that come from elsewhere: Imspector's MINFLUX traces and Picasso's picks are
numbered from 0, so the first trace and the first pick of every file were
never a region, never measured per trace, never in a summary.  With -1 as
none, a reader asks one question, `assigned`, whatever made the ids.  Ids
SMAPpy numbers still start at 1 where they always did (`group_id - 1` is the
row of the grouped table); 0 is simply allowed.

Files saved before the change carry 0 for none and are not converted (no
backwards compatibility was asked for): run the plugin that wrote the
column again.
"""
from __future__ import annotations

import numpy as np

#: the id of a localization that belongs to none
NONE = -1

#: id columns whose name does not end in ``_id``
OTHER_IDS = ("tid",)

#: what a file's own numbers are kept under once its ids are moved
#: (`file_ids`): ``tid_file``, ``track_id_file``
OWN = "_file"


def is_id(name: str) -> bool:
    """Whether column ``name`` is an id column by its name: ``*_id``,
    `OTHER_IDS`, and a file's own numbers of either (``tid_file``)."""
    base = name.removesuffix(OWN)
    return base.endswith("_id") or base in OTHER_IDS


def assigned(values) -> np.ndarray:
    """Which rows have an id: finite and not negative."""
    values = np.asarray(values)
    if values.dtype.kind == "f":
        return np.isfinite(values) & (values >= 0)
    if values.dtype.kind == "u":
        return np.ones(values.shape, bool)
    return values >= 0


def none_like(n: int, dtype=np.int64) -> np.ndarray:
    """``n`` rows in no id."""
    return np.full(n, NONE, dtype=dtype)
