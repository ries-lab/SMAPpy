"""Which column names the tracks: one rule for everything that reads them.

A track is the localizations sharing a value above 0 of an id column, and
which column that is depends on where the table came from: `track_id` from
Link tracks, `tid` from a MINFLUX file, `track_id_true` from a simulation.
So no plugin names one: each has a *track id* field whose default, *auto*
(stored as ""), takes the first of `TRACK_COLUMNS` the table has, and any
integer column can be chosen instead -- `group_id` to treat blinks as tracks,
`cluster_id`, a second linking.  `group_id` and `cluster_id` are never
*auto*: nearly every table has a `group_id`, so *auto* would always find
something even where there are no tracks.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# what *auto* looks for, in order: linked tracks first, then MINFLUX's
# traces, then a simulation's truth
TRACK_COLUMNS = ("track_id", "tid", "track_id_true")
AUTO = ""


def integer_columns(locs) -> List[str]:
    """The table's one-dimensional integer columns: what can name tracks."""
    return [n for n in locs.keys() if np.asarray(locs[n]).ndim == 1
            and np.asarray(locs[n]).dtype.kind in "iu"]


def auto_track_column(locs) -> str:
    """The first of `TRACK_COLUMNS` the table has, or "" for none."""
    return next((c for c in TRACK_COLUMNS if c in locs), "")


def track_column(locs, chosen: str = AUTO) -> str:
    """The column that names the tracks: ``chosen``, or on *auto* the first of
    `TRACK_COLUMNS` the table has.  Raises ValueError, naming the integer
    columns there are, when there is none."""
    column = (chosen or "").strip() or auto_track_column(locs)
    if column and column in locs:
        return column
    have = integer_columns(locs)
    if chosen:
        start = f"no column {chosen!r} to take the tracks from"
    else:
        start = ("no tracks: the table has none of "
                 + ", ".join(TRACK_COLUMNS))
    raise ValueError(start + ".  Link them (Analysis/Tracking/Link tracks), or "
                     "choose the track id among "
                     + (", ".join(have) if have else "(no integer column)"))


def track_column_choices(locs) -> List[Tuple[str, str]]:
    """The *track id* field's list: *auto* (saying what it resolves to),
    then every integer column."""
    found = auto_track_column(locs) if locs is not None else ""
    auto = f"auto ({found})" if found else "auto"
    return [(AUTO, auto)] + [(c, c) for c in (integer_columns(locs) if locs is not None
                                             else [])]
