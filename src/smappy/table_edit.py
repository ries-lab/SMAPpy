"""Columns renamed, removed or kept, and a table collapsed to one row per id.

What *Edit columns*, *Collapse by id* and *Save smappy HDF5*'s *columns*
do, on a table and without a session, so that a script gets the same
table the GUI does.

**What a column name is tied to.**  A column is read by name from many
places, and renaming or removing one is the user's business where the
failure is loud -- a saved setting, a chain step or a region table naming
the old column fails with that plugin's own error.  Three places would
fail silently or crash instead, and are handled here or by the session:

* the recipes (`mathparse`, ``metadata["derived"]``) name fields and read
  columns: a rename rewrites both; a removal forgets the field's own
  recipe and those that read it, which keep their values as plain data;
* the bounds the table carries (`KEPT_BOUNDS`) follow a rename and are
  dropped with a removed column -- a layer's filter skips a bound on a
  column it does not have without a word (`ViewState.filter_for`), so a
  removal says which bounds it took away;
* a layer's colour and weight column: the renderer draws without one the
  table does not have (`render.with_columns_of`), and the session moves
  them on a rename (`Result.renamed_columns`).

Some columns are not the user's to rename: the program finds them by name
to draw, link or split the table (`STRUCTURAL`).

**Collapsing** is `group.combine` keyed by an id column instead of by
linked blinks: SMAP's *GroupAny*, and *Connect2Unconnect*, which replaced a
channel's localizations by its merged blinks.
Its point is a smaller file, one row per track or cluster.  What a row
cannot hold is gone -- the members' frames but the first, ``logl`` -- and
the grouping columns are renamed (``n_in_group`` -> ``n_merged``) so that
merging the blinks of the collapsed table does not read them as its own.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .locs import KEPT_BOUNDS, Localizations
from .mathparse import (CONSTANTS, FUNCTIONS, forget, readers_of, recipes,
                        rename_recipes)

#: Columns the program finds by name to draw, link or split the table, and
#: why: renaming or removing one breaks what reads it, not a plugin's
#: setting.  `GROUP_COLUMNS` and `USE` are written and read by merging
#: blinks, `tid` makes it merge by trace, `region_filter` is the region
#: filter's flag, recomputed from the table's metadata.
STRUCTURAL: Dict[str, str] = {
    "x_nm": "a position", "y_nm": "a position", "z_nm": "a position",
    "xy_err_nm": "the precision the picture is drawn with",
    "frame": "the time axis: merging blinks, drift and tracks read it",
    "filenumber": "which file a localization came from",
    "channel": "which channel a localization came from",
    "group_id": "written by merging blinks", "n_in_group": "written by merging blinks",
    "group_use": "says which localizations merging blinks reads",
    "tid": "a trace id: blinks are merged by trace where it is there",
    "region_filter": "the region filter's flag, recomputed from its conditions",
}

#: What the grouping columns of a collapsed table are called
MERGED = "n_merged"


def editable(locs: Localizations) -> List[str]:
    """The columns of ``locs`` that may be renamed or removed."""
    return [n for n in locs.keys() if n not in STRUCTURAL]


def _refuse_structural(names: Sequence[str]) -> None:
    hit = [n for n in names if n in STRUCTURAL]
    if hit:
        raise ValueError("; ".join(f"{n} cannot be renamed or removed: {STRUCTURAL[n]}"
                                   for n in hit))


def check_name(name: str, locs: Localizations) -> str:
    """A new column name, refused now rather than half-written later."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name the new column")
    if not name.isidentifier():
        raise ValueError(f"{name!r} cannot be read in an expression: a column "
                         "name is a word of letters, digits and underscores, "
                         "not starting with a digit")
    if name in FUNCTIONS or name in CONSTANTS:
        raise ValueError(f"{name!r} is a function or a constant in an "
                         "expression; choose another")
    if name in STRUCTURAL:
        raise ValueError(f"{name!r} is {STRUCTURAL[name]}; choose another")
    if name in locs:
        raise ValueError(f"the table already has a column {name!r}")
    return name


def rename_column(locs: Localizations, old: str, new: str
                  ) -> Tuple[Localizations, List[str]]:
    """A copy of the table with column ``old`` called ``new``.

    Returns the table and what it changed besides the column, one line
    each: the recipes rewritten, the kept bounds moved.  The column keeps
    its place; the arrays are shared, so the copy costs a dictionary.
    """
    if old not in locs:
        raise ValueError(f"no column {old!r}; the table has: "
                         + ", ".join(locs.keys()))
    _refuse_structural([old])
    new = check_name(new, locs)
    columns = {(new if n == old else n): v for n, v in locs.columns.items()}
    out = Localizations(columns, dict(locs.metadata))
    notes = []
    changed = rename_recipes(out, old, new)
    if changed:
        notes.append(f"the recipes of {', '.join(changed)} now read {new}")
    kept = out.metadata.get(KEPT_BOUNDS) or {}
    if old in kept:
        out.metadata[KEPT_BOUNDS] = {(new if n == old else n): b
                                     for n, b in kept.items()}
        notes.append(f"the bound the file keeps on {old} is now on {new}")
    return out, notes


def remove_columns(locs: Localizations, names: Sequence[str]
                   ) -> Tuple[Localizations, List[str]]:
    """A copy of the table without ``names``.  Returns the table and the
    warnings: the recipes that read a removed column, now plain data, and
    the bounds the file kept on one, now gone."""
    names = list(dict.fromkeys(names))
    missing = [n for n in names if n not in locs]
    if missing:
        raise ValueError(f"no column {', '.join(missing)}; the table has: "
                         + ", ".join(locs.keys()))
    _refuse_structural(names)
    gone = set(names)
    out = Localizations({n: v for n, v in locs.columns.items() if n not in gone},
                        dict(locs.metadata))
    return out, _forget_columns(out, gone)


def _forget_columns(out: Localizations, gone) -> List[str]:
    """Drop the recipes and kept bounds of the columns in ``gone`` from
    ``out``'s metadata, in place; the warnings saying so."""
    notes = []
    readers = []
    for name in sorted(gone):
        readers += [r for r in readers_of(out, name) if r not in gone and r not in readers]
    for field in [r["field"] for r in recipes(out)]:
        if field in gone or field in readers:
            forget(out, field)
    if readers:
        notes.append(f"{', '.join(readers)} read{'s' * (len(readers) == 1)} a "
                     "removed column: kept as values, no longer recomputed "
                     "when blinks are merged")
    kept = dict(out.metadata.get(KEPT_BOUNDS) or {})
    dropped = [n for n in kept if n in gone]
    if dropped:
        for n in dropped:
            kept.pop(n)
        if kept:
            out.metadata[KEPT_BOUNDS] = kept
        else:
            out.metadata.pop(KEPT_BOUNDS, None)
        notes.append(f"the filter the file kept on {', '.join(dropped)} is gone "
                     "with the column")
    return notes


def keep_columns(locs: Localizations, names: Sequence[str]) -> Localizations:
    """The table with only ``names`` -- and always the positions and
    ``filenumber``, without which the file would not open as the same
    files.  The recipes and kept bounds of what is left out go too.
    Names the table does not have are passed over: a saved choice of
    columns meets tables with fewer."""
    wanted = set(names) | {"x_nm", "y_nm", "z_nm", "filenumber"}
    gone = {n for n in locs.keys() if n not in wanted}
    if not gone:
        return locs
    out = Localizations({n: v for n, v in locs.columns.items() if n not in gone},
                        dict(locs.metadata))
    _forget_columns(out, gone)
    return out


# -------------------------------------------------------------- collapsing

#: The id columns *Collapse by id* offers first, when the table has them
ID_COLUMNS = ("group_id", "track_id", "cluster_id", "roi_id", "tid")


def id_candidates(locs: Localizations) -> List[str]:
    """Whole-number columns a table can be collapsed by, the usual ids
    first.  Not ``frame`` or ``filenumber``: one row per frame is a time
    course, not a smaller file."""
    ints = [n for n in locs.keys()
            if np.asarray(locs[n]).ndim == 1 and np.asarray(locs[n]).dtype.kind in "iu"
            and n not in ("frame", "filenumber", "channel", "n_in_group")]
    return [n for n in ID_COLUMNS if n in ints] + [n for n in ints if n not in ID_COLUMNS]


def collapse_index(locs: Localizations, column: str, unassigned: str = "keep"
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """1-based group ids for `group.combine`, one per row, and the rows used.

    One group per distinct ``(filenumber, channel, id)``: an id numbered
    per file meets the same number in the next file, and a cluster that
    spans both channels becomes one row per channel rather than a row whose
    channel is 1.5.  A row whose id is 0 or below is in none: with
    ``unassigned`` ``"keep"`` it is a row of its own, with ``"remove"`` it
    is not in the result.
    """
    if column not in locs:
        raise ValueError(f"no column {column!r}; the table has: "
                         + ", ".join(locs.keys()))
    ids = np.asarray(locs[column])
    if ids.dtype.kind not in "iu":
        raise ValueError(f"{column} is not a column of whole numbers: ids are")
    ids = ids.astype(np.int64)
    n = len(ids)
    has = ids > 0
    rows = np.arange(n) if unassigned == "keep" else np.flatnonzero(has)
    keys = [ids]
    for name in ("filenumber", "channel"):
        if name in locs:
            keys.append(np.asarray(locs[name], np.int64))
    if unassigned == "keep":
        # each unassigned row its own key: a negative number no id has
        alone = np.where(has, 0, -1 - np.arange(n, dtype=np.int64))
        keys[0] = np.where(has, ids, alone)
    stacked = np.stack([k[rows] for k in keys], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64) + 1, rows


def collapse(locs: Localizations, column: str, unassigned: str = "keep",
             progress=None) -> Tuple[Localizations, List[str]]:
    """One row per id of ``column``, each column reduced by the rule merging
    blinks uses (`group.combine`).  Returns the table and what was changed
    besides, one line each.

    ``n_merged`` is how many rows each one was made from; the table's own
    ``group_id`` and ``n_in_group`` are not kept -- they described rows
    that are gone, and merging the blinks of this table writes its own.
    ``column`` keeps the id, exactly (a mean through float32 holds an id
    only up to 2**24).
    """
    from .group import GROUP_COLUMNS, USE, combine
    if unassigned not in ("keep", "remove"):
        raise ValueError(f"unassigned is 'keep' or 'remove', not {unassigned!r}")
    index, rows = collapse_index(locs, column, unassigned)
    source = locs if len(rows) == len(locs) else locs[rows]
    ids = np.asarray(source[column])
    notes = []
    # a recipe in the grouping columns would be recomputed from columns that
    # are not there any more, or mean something else in `n_merged`: kept as
    # values
    readers = []
    for name in GROUP_COLUMNS:
        readers += [r for r in readers_of(source, name) if r not in readers]
    out = combine(source, index, progress=progress)
    for field in readers:
        forget(out, field)
    if readers:
        notes.append(f"{', '.join(readers)} kept as values: "
                     "read the grouping columns, which are not kept")
    order = np.argsort(index, kind="stable")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(index[order])) + 1))
    first = order[starts]                       # a row of each group, in group order
    columns = dict(out.columns)
    columns[MERGED] = columns.pop("n_in_group")
    columns.pop("group_id", None)
    columns.pop(USE, None)
    # the id itself, exactly; an unassigned row keeps its own 0 or below
    columns[column] = ids[first]
    metadata = dict(out.metadata)
    metadata.pop("grouped", None)
    metadata["collapsed"] = {"by": column, "rows": int(len(locs)),
                             "unassigned": unassigned}
    if column == "group_id":
        # merged blinks: their ids name nothing once the localizations are
        # gone, and a column of that name is written by the next merge
        columns.pop(column)
    collapsed = Localizations(columns, metadata)
    dropped = sorted(set(locs.keys()) - set(collapsed.keys()) - set(GROUP_COLUMNS)
                     - {USE})
    if dropped:
        notes.append(f"not kept: {', '.join(dropped)}")
    return collapsed, notes
