"""Statistics per id: the localizations of each trace, track or cluster at once.

A region of an id column (`docs/regions.md`) -- a MINFLUX trace (`tid`), a
linked track, a cluster -- is measured by a block one region at a time, which
is right for a measurement that needs the region's own context and slow for
a count or a spread: 10^5 traces through a block are minutes, through a
`bincount` per quantity well under a second.  This module is that pass, on
arrays, so a script, a test and the plugin (`plugins/id_statistics.py`) get
the same numbers; the plugin hands them back as a region table
(`region_runs.column_table`), where the summaries and the region filter
(`docs/region-filter.md`) find them.

What is measured, per region (an id above 0 among the rows kept):

* ``n_locs``, its localizations;
* ``centre_x_nm`` ... and ``std_x_nm`` ...: the mean and the sample standard
  deviation (ddof 1, NaN for one localization) of each coordinate, and
  ``std_xy_nm``, the RMS of the x and y ones -- how `xy_err_nm` pools the
  two.  The spread is the region's own and the localization precision
  together: nothing is subtracted, as in *Cluster shape*.  For a MINFLUX
  trace of one emitter it is the measured precision; a trace twice as wide
  as its neighbours is two emitters or a jump between sites.  For a moving
  track it measures the motion and is no quality measure;
* its time: ``start_s``, ``end_s`` and ``duration_s`` from ``time_s`` where
  the table has a clock (MINFLUX), else ``first_frame``, ``last_frame`` and
  ``frames`` (the span, last - first + 1);
* for each chosen column, the chosen statistics (mean, median, std, min,
  max) of its finite values -- a localization whose `cfr` was not measured
  (NaN) counts for neither -- named ``<quantity>_<stat>[_<unit>]``
  (``cfr_mean``, ``xy_err_median_nm``) and listed as a group, so a reader
  can show ``cfr › mean`` as a level of its own (`RegionTable.groups`).

The regions are found without a sort where the ids allow (`dense_codes`):
ids are small integers, so a lookup table maps each to its region; the rows
are put in region order once (not at all for a MINFLUX file, whose traces
follow each other), and every statistic but the median is a `reduceat` over
each region's run.  A median sorts once per column.  2 x 10^7
localizations in 10^6 traces, with three coordinates, the time and three
columns: see `tests/test_id_statistics.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: what can be asked of a chosen column, in the order they are written
STATISTICS = ("mean", "median", "std", "min", "max")

# the unit suffixes a statistic goes in front of: `xy_err_nm` -> `xy_err_mean_nm`
UNITS = ("nm", "nm2", "s", "ms", "pix", "um2_per_s", "per_um2")

# a lookup table this many times the rows at most; above, `np.unique` sorts
DENSE_FACTOR = 4


@dataclass
class IdStatistics:
    """The regions' ids, a value array per name (one entry per region) and
    the groups of statistics of one quantity (`RegionTable.groups`)."""
    ids: np.ndarray
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    groups: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)


def stat_name(column: str, stat: str) -> str:
    """``cfr`` -> ``cfr_mean``; ``xy_err_nm`` -> ``xy_err_mean_nm``: the
    statistic before the unit, so the name keeps `<quantity>_<unit>`."""
    for unit in sorted(UNITS, key=len, reverse=True):
        if column.endswith("_" + unit) and len(column) > len(unit) + 1:
            return f"{column[:-len(unit) - 1]}_{stat}_{unit}"
    return f"{column}_{stat}"


def dense_codes(ids: np.ndarray, keep: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """``(region ids, code per row)``: the distinct ids above 0 among the kept
    rows, ascending, and each row's index among them (-1 for a row in no
    region).  Ids that are small integers go through a lookup table, which is
    linear; others (large ids, floats) through `np.unique`."""
    ids = np.asarray(ids)
    if ids.dtype.kind == "f":
        good = np.isfinite(ids) & (ids == np.round(ids))
        ids = np.where(good, ids, 0).astype(np.int64)
    ids = ids.astype(np.int64, copy=False)
    member = ids > 0
    if keep is not None:
        member &= np.asarray(keep, dtype=bool)
    codes = np.full(len(ids), -1, np.int64)
    if not member.any():
        return np.zeros(0, np.int64), codes
    top = int(ids[member].max())
    if top <= DENSE_FACTOR * len(ids) + 1024:
        present = np.bincount(ids[member], minlength=top + 1) > 0
        lookup = np.cumsum(present) - 1
        codes[member] = lookup[ids[member]]
        return np.flatnonzero(present).astype(np.int64), codes
    found, inverse = np.unique(ids[member], return_inverse=True)
    codes[member] = inverse.reshape(-1)
    return found.astype(np.int64), codes


class _Regions:
    """The member rows ordered by region, once for every column: what the
    statistics other than sums need (`reduceat` over each region's run)."""

    def __init__(self, codes: np.ndarray, n: int):
        self.codes, self.n = codes, n
        rows = np.flatnonzero(codes >= 0)
        c = codes[rows]
        if len(c) and np.any(c[1:] < c[:-1]):
            # by region, stably: a radix sort, linear, where numpy's stable
            # sort of int64 is a comparison sort.  Skipped where the rows
            # already are -- a MINFLUX file, whose traces follow each other
            from .group import radix_argsort
            rows = rows[radix_argsort(c)]
        self.order = rows
        self.counts = np.bincount(codes[rows], minlength=n)
        # every region has a member row, so no run is empty
        self.begin = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)

    def statistics(self, values, stats: Sequence[str]) -> Dict[str, np.ndarray]:
        """The statistics of ``values`` per region, over their finite entries;
        NaN where a region has none (and for a std of one)."""
        out: Dict[str, np.ndarray] = {}
        if not self.n:
            return {s: np.zeros(0) for s in STATISTICS if s in stats}
        v = np.asarray(values, dtype=np.float64)[self.order]
        finite = np.isfinite(v)
        count = np.add.reduceat(finite.astype(np.int64), self.begin).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            if "mean" in stats or "std" in stats:
                mean = np.add.reduceat(np.where(finite, v, 0.0), self.begin) / count
                out["mean"] = mean
            if "std" in stats:
                # two passes: positions are 10^4 nm and their spread a few nm
                dev = np.where(finite, v - np.repeat(mean, self.counts), 0.0) ** 2
                out["std"] = np.where(count > 1, np.sqrt(
                    np.add.reduceat(dev, self.begin) / (count - 1)), np.nan)
            if "min" in stats:
                out["min"] = np.fmin.reduceat(v, self.begin)     # fmin skips NaN
            if "max" in stats:
                out["max"] = np.fmax.reduceat(v, self.begin)
        if "median" in stats:
            out["median"] = self._median(v, finite, count)
        return {s: out[s] for s in STATISTICS if s in stats}

    def _median(self, v, finite, count) -> np.ndarray:
        """Each region's values in order, by one sort of the key region +
        value scaled into [0, 0.5] (NaN at 0.75, after every value).  Two
        values of a region closer than 2^-52 x regions x their range may
        swap, which moves a median by no more than that; sorting by value
        and then stably by region is exact and was three times slower."""
        lo, hi = (float(v[finite].min()), float(v[finite].max())) if finite.any() \
            else (0.0, 1.0)
        span = hi - lo if hi > lo else 1.0
        key = np.where(finite, 0.5 * (v - lo) / span, 0.75)
        key += np.repeat(np.arange(self.n, dtype=np.float64), self.counts)
        ordered = v[np.argsort(key)]
        med = np.full(self.n, np.nan)
        k = count.astype(np.int64)
        has = k > 0
        a = self.begin[has] + (k[has] - 1) // 2
        b = self.begin[has] + k[has] // 2
        med[has] = 0.5 * (ordered[a] + ordered[b])
        return med


def id_statistics(ids, positions: Dict[str, np.ndarray], *, keep=None,
                  time_s=None, frame=None,
                  columns: Optional[Dict[str, np.ndarray]] = None,
                  stats: Sequence[str] = ("mean", "median", "max")) -> IdStatistics:
    """The statistics of every region of ``ids`` (see the module docstring).

    ``positions`` maps ``"x"``, ``"y"`` and optionally ``"z"`` to nm; ``keep``
    is the rows measured (a layer's filter), the rest are in no region and a
    region with none kept is left out; ``time_s`` or ``frame`` give the time;
    ``columns`` the other quantities, each summarised by ``stats``."""
    unknown = [s for s in stats if s not in STATISTICS]
    if unknown:
        raise ValueError(f"unknown statistics {unknown}: choose among {STATISTICS}")
    found, codes = dense_codes(ids, keep)
    n = len(found)
    out = IdStatistics(found)
    regions = _Regions(codes, n)
    out.values["n_locs"] = regions.counts.astype(np.int64)
    std = {}
    for axis in ("x", "y", "z"):
        if axis not in positions:
            continue
        s = regions.statistics(positions[axis], ("mean", "std"))
        out.values[f"centre_{axis}_nm"] = s["mean"]
        std[axis] = out.values[f"std_{axis}_nm"] = s["std"]
    if "x" in std and "y" in std:
        out.values["std_xy_nm"] = np.sqrt(0.5 * (std["x"] ** 2 + std["y"] ** 2))
    if std:
        out.groups["std"] = {a: f"std_{a}_nm" for a in ("x", "y", "z", "xy")
                             if f"std_{a}_nm" in out.values}
        out.groups["centre"] = {a: f"centre_{a}_nm" for a in ("x", "y", "z")
                                if f"centre_{a}_nm" in out.values}
    if time_s is not None:
        s = regions.statistics(time_s, ("min", "max"))
        out.values["start_s"], out.values["end_s"] = s["min"], s["max"]
        out.values["duration_s"] = s["max"] - s["min"]
    elif frame is not None:
        s = regions.statistics(frame, ("min", "max"))
        first, last = s["min"].astype(np.int64), s["max"].astype(np.int64)
        out.values["first_frame"], out.values["last_frame"] = first, last
        out.values["frames"] = last - first + 1
    for name, column in (columns or {}).items():
        s = regions.statistics(column, tuple(stats))
        out.groups[name] = {}
        for stat, value in s.items():
            key = stat_name(name, stat)
            out.values[key] = value
            out.groups[name][stat] = key
    return out
