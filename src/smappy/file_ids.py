"""Ids that a file brings, kept apart when files share one table.

A file's id columns -- MINFLUX's trace id `tid`, a `track_id` or
`cluster_id` saved with it -- number its own traces, tracks or clusters from
1 (`tid` from 0), so two files in one table share numbers.  Grouping keeps
them apart by file and channel (`group.by_trace`'s blocks), but a region of
an id column is its id alone (`region_runs.ColumnRegions`, *Id statistics*):
two files shown in one layer would merge their trace 7s into one region.
Keying every region on (file, id) would reach the region tables, the
results stored per id (`Session.id_regions`), ROIs made from an id column,
the values written back and the region filter; making the ids unique when a
file joins the table reaches none of them.  So a file added to a table whose
ids overlap its own has them moved past the largest there, as *Combine
files* and `region_runs.merge_ids` do with the ids SMAPpy writes.

The file's own numbers are not thrown away: they are what links a trace
back to the instrument's record (Imspector's `tid`, a track found here to
the raw data), so they are kept in ``<name>_file`` (`tid_file`) and the
shift in the file's metadata (``id_offsets``).  `own_ids` reads them back;
code that matches ids against a file's own metadata (the MBM beads'
`mbm_beads` entries) reads them through it.  The shifted column is the
default everywhere else.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .group import GROUP_COLUMNS
from .locs import Localizations

#: integer ids whose every value is an id -- 0 included -- rather than 0
#: meaning "in none" as for the ``*_id`` columns
ALL_VALUE_IDS = ("tid",)

#: an ROI's own number: the same ROI whichever file its rows came from
REGION_NUMBER = "roi_id"

#: where a file's own numbers are kept once they are moved: ``tid_file``
OWN = "_file"


def id_columns(locs: Localizations) -> List[str]:
    """The integer columns whose numbers are per file and must be kept apart."""
    written = set(locs.metadata.get("region_columns") or {})
    out = []
    for name in locs.columns:
        if name in GROUP_COLUMNS or name == REGION_NUMBER or name in written:
            continue
        if not (name.endswith("_id") or name in ALL_VALUE_IDS):
            continue
        if np.asarray(locs[name]).dtype.kind in "iu":
            out.append(name)
    return out


def own_ids(locs: Localizations, name: str) -> np.ndarray:
    """The file's own numbers of id column ``name``: ``<name>_file`` where
    the ids were moved, else the column itself."""
    own = name + OWN
    return np.asarray(locs[own] if own in locs else locs[name])


def _used(values: np.ndarray, name: str) -> np.ndarray:
    return values if name in ALL_VALUE_IDS else values[values > 0]


def apart(values: np.ndarray, files: np.ndarray, name: str) -> bool:
    """Whether no two files share a number of id column ``name``."""
    values, files = np.asarray(values, np.int64), np.asarray(files, np.int64)
    used = values >= 0 if name in ALL_VALUE_IDS else values > 0
    pairs = np.unique(np.column_stack([values[used], files[used]]), axis=0)
    return len(np.unique(pairs[:, 0])) == len(pairs)


def keep_apart(old: Localizations, new: Localizations
               ) -> Tuple[Localizations, Localizations, Dict[str, int]]:
    """``old`` and ``new`` with the ids ``new`` brings moved past ``old``'s
    where they overlap, the original numbers kept in ``<name>_file`` on both
    (``old``'s own are its ids where it had none kept); and the shift per
    column.  A column ``old`` lacks, or numbers that do not overlap, are left
    as they are."""
    offsets: Dict[str, int] = {}
    old_cols, new_cols = dict(old.columns), dict(new.columns)
    for name in id_columns(new):
        if name not in old:
            continue
        mine = np.asarray(new[name], np.int64)
        theirs = _used(np.asarray(old[name], np.int64), name)
        used = _used(mine, name)
        if not len(theirs) or not len(used) or used.min() > theirs.max():
            continue
        # as *Combine files* numbers on: past the largest, 0 counted where it
        # is an id, so a file's own number is the id less its offset
        offset = int(theirs.max()) + (1 if name in ALL_VALUE_IDS else 0)
        own = name + OWN
        if own not in old_cols:
            old_cols[own] = np.asarray(old[name]).copy()
        if own not in new_cols:
            new_cols[own] = np.asarray(new[name]).copy()
        moved = mine + offset
        if name not in ALL_VALUE_IDS:            # 0 is "in none" and stays so
            moved = np.where(mine > 0, moved, 0)
        dtype = np.asarray(new[name]).dtype      # kept, where the ids still fit
        fits = moved.max(initial=0) <= np.iinfo(dtype).max
        new_cols[name] = moved.astype(dtype if fits else np.int64)
        offsets[name] = offset
    if not offsets:
        return old, new, offsets
    return (Localizations(old_cols, dict(old.metadata)),
            Localizations(new_cols, dict(new.metadata)), offsets)
