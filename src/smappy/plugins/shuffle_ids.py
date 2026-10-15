"""Shuffle the ids of an integer column, so neighbours get colours apart.

SMAP's ``ShuffelClusterIndex``.  Coloured by `cluster_id`, the
clusters a clusterer numbered one after the other -- by size, by position,
by the order it found them -- come out in neighbouring colours of the LUT,
and two touching clusters cannot be told apart.  The same is true of an ROI
number, a MINFLUX trace (`tid`), a track or a frame.  Colouring by the ids
in a random order puts neighbours far apart on the LUT.

**A permutation of the ids there are.**  Each id is replaced by another id
of the same column, drawn without replacement: the values stay the ones the
column had, so the colour range set for it fits the shuffled column too,
and a filter on a range of ids still means about as many ids.  SMAP
permuted 1 to the largest id instead, which hands a column with gaps (frames,
ROI numbers after deletions) values it never had.  An id of 0 or below --
DBSCAN's noise, a localization in no ROI -- is kept as it is: it is "in
none", not an id (SMAP set a negative one to 0).

**A new column**, ``<column>_shuffled`` by default, rather than the column
rewritten: an id is a name other things refer to -- the region results are
kept by it, Regions to column writes by it -- and only its colour is meant
to change.  The column is a recipe with the rule ``max`` (`mathparse`), as
`cluster_id` is, so a merged blink keeps one value.  A new seed each run
unless one is given: run it again for another order.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from . import Context, FormState, Plugin, Result, param, register
from ..locs import Localizations


def shuffle_ids(ids: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """``ids`` with every id above 0 replaced through one random permutation
    of the distinct ids; 0 and below unchanged."""
    ids = np.asarray(ids)
    out = ids.copy()
    positive = ids > 0
    distinct, where = np.unique(ids[positive], return_inverse=True)
    rng = np.random.default_rng(seed)
    out[positive] = rng.permutation(distinct)[where]
    return out


def _integer_columns(locs) -> list:
    return [n for n in locs.keys() if np.asarray(locs[n]).ndim == 1
            and np.asarray(locs[n]).dtype.kind in "iu"]


def _no_column():
    return [("", "-")]


@dataclass
class ShuffleSettings:
    column: str = param("", label="column", choices=_no_column,
                        help="an integer column: cluster_id, roi_id, tid, frame, ...")
    name: str = param("", label="new column",
                      help="empty: the column's name with _shuffled")
    seed: Optional[int] = param(None, label="seed", advanced=True,
                                help="auto: a new order each run; a number "
                                     "gives the same order again")


@register("Analysis/Process/Shuffle ids")
class ShuffleIds(Plugin):
    """Write an integer column with its ids in a random order, so that
    neighbouring clusters, ROIs or tracks get colours far apart."""

    Settings = ShuffleSettings
    changes_locs = True
    # the ids are per localization; a merged blink takes the largest (`max`)
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: ShuffleSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            state.choices["column"] = [(n, n) for n in _integer_columns(ctx.locs)]
            if settings.column:
                state.hints["name"] = f"{settings.column}_shuffled"
        return state

    def run(self, ctx: Context, settings: ShuffleSettings) -> Result:
        from ..mathparse import remember
        locs = ctx.locs
        column = settings.column
        if not column:
            raise ValueError("choose a column")
        if column not in locs:
            raise ValueError(f"no column {column!r}; the integer ones are "
                             + ", ".join(_integer_columns(locs)))
        ids = np.asarray(locs[column])
        if ids.dtype.kind not in "iu":
            raise ValueError(f"{column} is not a column of whole numbers: ids are")
        seed = settings.seed if settings.seed is not None else \
            int(np.random.default_rng().integers(0, 2 ** 31))
        name = settings.name.strip() or f"{column}_shuffled"
        if name == column:
            raise ValueError("the shuffled ids go into a new column: an id is "
                             "what results and regions are kept by")
        out = Localizations({**locs.columns, name: shuffle_ids(ids, seed)},
                            dict(locs.metadata))
        remember(out, name, "", grouped="max")
        n = len(np.unique(ids[ids > 0]))
        return Result(locs=out, text=f"{n} ids of {column} shuffled into {name} "
                                     f"(seed {seed})",
                      settings=replace(settings, seed=seed),
                      data={"column": name})
