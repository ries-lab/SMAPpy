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

What is measured, per region (an id, 0 or above, among the rows kept):

* ``n_locs``, its localizations;
* ``centre_x_nm`` ...: the mean of each coordinate, always -- where the
  region is, which *Centre ROIs*, *Write results* and the results window
  read -- and ``std_xy_nm``, the RMS of the x and y sample standard
  deviations (ddof 1, NaN for one localization) -- how `xy_err_nm` pools
  the two, the region's size;
* for each axis asked for (``axes``), its own ``std_x_nm`` ... and the
  chosen statistics but the mean, which is the centre (``x_median_nm``,
  ``z_p95_nm``).  The spread is the region's own and the localization precision
  together: nothing is subtracted, as in *Cluster shape*.  For a MINFLUX
  trace of one emitter it is the measured precision; a trace twice as wide
  as its neighbours is two emitters or a jump between sites.  For a moving
  track it measures the motion and is no quality measure;
* its time: ``start_s``, ``end_s`` and ``duration_s`` from ``time_s``,
  and ``first_frame``, ``last_frame`` and ``frames`` (the span, last - first
  + 1) from ``frame``, each where it is given;
* for each chosen column, the chosen statistics (mean, median, std, min,
  max, and percentiles, ``photons_p5``) of its finite values -- a localization whose `cfr` was not measured
  (NaN) counts for neither -- named ``<quantity>_<stat>[_<unit>]``
  (``cfr_mean``, ``xy_err_median_nm``) and listed as a group, so a reader
  can show ``cfr › mean`` as a level of its own (`RegionTable.groups`);
* on request (``shape``), *Cluster shape*'s numbers under its names
  (`region_shapes`): the convex hull's area and density, the radius of
  gyration, the standard deviations along the covariance's long and short
  axis, the long axis' angle (degrees, 0 to 180 from x, unoriented: a
  cluster has no direction in time, unlike `traces.axes`' 0-360 forward
  one), the elongation and the extents along both axes, and in 3D the z
  extent and the hull's volume.  *Cluster shape* calls the same function
  for its one region, so the numbers are the same to the last bit.

The regions are found without a sort where the ids allow (`dense_codes`):
ids are small integers, so a lookup table maps each to its region; the rows
are put in region order once (not at all for a MINFLUX file, whose traces
follow each other), and every statistic but the median is a `reduceat` over
each region's run.  A median or a percentile sorts once per column.  2 x 10^7
localizations in 10^6 traces, with three coordinates, the time and three
columns: see `tests/test_id_statistics.py`.

The shape is vectorised the same way -- the covariance's axes in closed
form from three sums per region, the extents an `fmax` / `fmin.reduceat` of
the rotated positions -- all but the hull, which qhull computes one region
at a time (`hull_measure`).  That loop is the cost of *shape*: 10^5
clusters of 5-60 localizations (3.2 x 10^6 in all) took 3.9 s in 2D and
7.9 s in 3D (the area's hull and the volume's) on an M-series Mac, ~35 us
a hull, almost all of it inside qhull, against 0.15 s for every statistic
without *shape*; a region of fewer than three points (four in 3D) has no
hull and is not passed to qhull.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: what can be asked of a chosen column, in the order they are written
STATISTICS = ("mean", "median", "std", "min", "max")

def percentile_key(q: float) -> str:
    """The statistic's name of a percentile: 5 -> ``p5``, 2.5 -> ``p2_5``
    -- no dot, so the column stays a name an expression can use."""
    text = f"{float(q):g}"
    return "p" + text.replace(".", "_")


def parse_percentiles(text) -> Tuple[float, ...]:
    """Typed percentiles, ``"5 95"`` or ``"2.5, 97.5"`` (or a list), each
    once, in order.  ValueError for one that is not a number in [0, 100]."""
    if isinstance(text, str):
        text = text.replace(",", " ").split()
    out = []
    for item in text or ():
        try:
            q = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"percentile {item!r} is not a number: type them "
                             "as 5 95") from None
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"percentile {q:g} is outside 0 to 100")
        if q not in out:
            out.append(q)
    return tuple(out)


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
    """``(region ids, code per row)``: the distinct ids (0 and up) among the kept
    rows, ascending, and each row's index among them (-1 for a row in no
    region).  Ids that are small integers go through a lookup table, which is
    linear; others (large ids, floats) through `np.unique`."""
    ids = np.asarray(ids)
    if ids.dtype.kind == "f":
        good = np.isfinite(ids) & (ids == np.round(ids))
        ids = np.where(good, ids, -1).astype(np.int64)
    ids = ids.astype(np.int64, copy=False)
    member = ids >= 0                      # -1 is "in none" (`ids.NONE`)
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

    def statistics(self, values, stats: Sequence[str],
                   percentiles: Sequence[float] = ()) -> Dict[str, np.ndarray]:
        """The statistics of ``values`` per region, over their finite entries,
        then each percentile under `percentile_key`; NaN where a region has
        none (and for a std of one)."""
        out: Dict[str, np.ndarray] = {}
        keys = [s for s in STATISTICS if s in stats] + [percentile_key(q)
                                                        for q in percentiles]
        if not self.n:
            return {k: np.zeros(0) for k in keys}
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
        if "median" in stats or len(percentiles):
            ordered = self._ordered(v, finite)
            k = count.astype(np.int64)
            has = k > 0
            if "median" in stats:
                med = np.full(self.n, np.nan)
                a = self.begin[has] + (k[has] - 1) // 2
                b = self.begin[has] + k[has] // 2
                med[has] = 0.5 * (ordered[a] + ordered[b])
                out["median"] = med
            for q in percentiles:
                out[percentile_key(q)] = self._percentile(ordered, k, has, q)
        return {key: out[key] for key in keys}

    def _ordered(self, v, finite) -> np.ndarray:
        """Each region's values in order, its finite ones first, by one sort
        of the key region + value scaled into [0, 0.5] (NaN at 0.75, after
        every value).  Two values of a region closer than 2^-52 x regions x
        their range may swap, which moves a median by no more than that;
        sorting by value and then stably by region is exact and was three
        times slower."""
        lo, hi = (float(v[finite].min()), float(v[finite].max())) if finite.any() \
            else (0.0, 1.0)
        span = hi - lo if hi > lo else 1.0
        key = np.where(finite, 0.5 * (v - lo) / span, 0.75)
        key += np.repeat(np.arange(self.n, dtype=np.float64), self.counts)
        return v[np.argsort(key)]

    def _percentile(self, ordered, k, has, q: float) -> np.ndarray:
        """The ``q``-th percentile of each region's finite values, linear
        between the two nearest as `np.percentile` is by default, so a
        script's ``np.nanpercentile`` of one region gives the same."""
        out = np.full(self.n, np.nan)
        h = (k[has] - 1) * (q / 100.0)
        low = np.floor(h).astype(np.int64)
        high = np.minimum(low + 1, k[has] - 1)
        b = self.begin[has]
        a, c = ordered[b + low], ordered[b + high]
        out[has] = a + (h - low) * (c - a)
        return out


# ------------------------------------------------------------------ shape

#: what `region_shapes` adds to a region table, *Cluster shape*'s names
SHAPE_2D = ("area_nm2", "density_per_um2", "rg_nm", "std_long_nm", "std_short_nm",
            "angle_deg", "elongation", "extent_long_nm", "extent_short_nm")
SHAPE_3D = ("z_extent_nm", "volume_nm3")


def hull_measure(points: np.ndarray) -> float:
    """The convex hull's area of (n, 2) points, or volume of (n, 3); 0 when
    the points span less than the dimension (too few, or on one line or
    plane)."""
    from scipy.spatial import ConvexHull, QhullError
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    if n <= d:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def covariance_axes(vx, vy, cxy):
    """``(std_long, std_short, angle)`` of covariances given by their
    entries (arrays, one per region): the square roots of the eigenvalues,
    in closed form, and the long axis' angle in radians in [0, pi) --
    unoriented, where `traces.axes` turns it forward in time.  A short axis
    below a millionth of the long one is 0: points on one line leave a
    rounding error across it that would make an elongation of 10^8."""
    vx, vy, cxy = (np.asarray(a, dtype=np.float64) for a in (vx, vy, cxy))
    half = 0.5 * (vx + vy)
    disc = np.sqrt(np.maximum((0.5 * (vx - vy)) ** 2 + cxy * cxy, 0.0))
    big = np.maximum(half + disc, 0.0)
    small = np.maximum(half - disc, 0.0)
    small = np.where(small > 1e-12 * big, small, 0.0)
    angle = np.mod(0.5 * np.arctan2(2.0 * cxy, vx - vy), np.pi)
    return np.sqrt(big), np.sqrt(small), angle


def region_shapes(codes: np.ndarray, n: int, x, y, z=None) -> Dict[str, np.ndarray]:
    """*Cluster shape*'s values of every region at once: ``codes`` is each
    row's region (0 .. n-1, -1 for none), ``x``, ``y`` (``z``) in nm.  Rows
    with a position that is not finite are left out, as *Cluster shape*
    leaves them out.  Returns ``centre_x_nm`` ... and the `SHAPE_2D` (and
    with ``z`` `SHAPE_3D`) names, an array of ``n`` each; NaN where a value
    needs more points than a region has (a standard deviation of one, a
    density without area), 0 for the hull of fewer than three (four) points
    or of points on a line (plane)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    use = (np.asarray(codes) >= 0) & np.isfinite(x) & np.isfinite(y)
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        use &= np.isfinite(z)
    regions = _Regions(np.where(use, codes, -1), n)
    counts = regions.counts
    present = counts > 0
    begin = regions.begin[present]
    m = counts.astype(np.float64)

    def per(op, v):
        out = np.full(n, np.nan)
        if begin.size:
            out[present] = op.reduceat(v, begin)
        return out

    xs, ys = x[regions.order], y[regions.order]
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        cx, cy = per(np.add, xs) / m, per(np.add, ys) / m
        # two passes: positions are 10^4 nm, their spread a few nm
        dx, dy = xs - np.repeat(cx, counts), ys - np.repeat(cy, counts)
        sxx, syy, sxy = per(np.add, dx * dx), per(np.add, dy * dy), per(np.add, dx * dy)
        two = m > 1
        long, short, angle = covariance_axes(sxx / (m - 1), syy / (m - 1), sxy / (m - 1))
        long, short, angle = (np.where(two, a, np.nan) for a in (long, short, angle))
        c, s = np.cos(angle), np.sin(angle)
        cr, sr = np.repeat(c, counts), np.repeat(s, counts)
        along, across = dx * cr + dy * sr, -dx * sr + dy * cr
        out["centre_x_nm"], out["centre_y_nm"] = cx, cy
        out["rg_nm"] = np.sqrt((sxx + syy) / m)
        out["std_long_nm"], out["std_short_nm"] = long, short
        out["angle_deg"] = np.degrees(angle)
        out["elongation"] = np.where(short > 0, long / short, np.nan)
        out["extent_long_nm"] = np.where(
            two, per(np.fmax, along) - per(np.fmin, along), np.nan)
        out["extent_short_nm"] = np.where(
            two, per(np.fmax, across) - per(np.fmin, across), np.nan)
        out["area_nm2"] = _hulls(np.column_stack((xs, ys)), regions, n)
        out["density_per_um2"] = np.where(out["area_nm2"] > 0,
                                          m / out["area_nm2"] * 1e6, np.nan)
        if z is not None:
            zs = z[regions.order]
            out["centre_z_nm"] = per(np.add, zs) / m
            out["z_extent_nm"] = per(np.fmax, zs) - per(np.fmin, zs)
            out["volume_nm3"] = _hulls(np.column_stack((xs, ys, zs)), regions, n)
    return out


def _hulls(points: np.ndarray, regions: _Regions, n: int) -> np.ndarray:
    """Each region's hull measure (`hull_measure`) of its run of
    ``points`` (rows in region order): the one loop over regions, skipping
    those too small for a hull (0) and the empty ones (NaN)."""
    d = points.shape[1]
    counts, begin = regions.counts, regions.begin
    out = np.where(counts > 0, 0.0, np.nan)
    for i in np.flatnonzero(counts > d):
        b = begin[i]
        out[i] = hull_measure(points[b:b + counts[i]])
    return out


def id_statistics(ids, positions: Dict[str, np.ndarray], *, keep=None,
                  time_s=None, frame=None,
                  columns: Optional[Dict[str, np.ndarray]] = None,
                  stats: Sequence[str] = ("mean", "median", "max"),
                  percentiles: Sequence[float] = (),
                  axes: Optional[Sequence[str]] = None,
                  shape: bool = False) -> IdStatistics:
    """The statistics of every region of ``ids`` (see the module docstring).

    ``positions`` maps ``"x"``, ``"y"`` and optionally ``"z"`` to nm, each
    measured for its centre; ``axes`` those also summarised -- their spread
    and ``stats`` -- all of them when None; ``keep`` is the rows measured (a
    layer's filter), the rest are in no region and a region with none kept
    is left out; ``time_s`` and ``frame`` give the time; ``columns`` the
    other quantities, each summarised by ``stats`` and ``percentiles``;
    ``shape`` adds `region_shapes`' values (3D ones where ``"z"`` is among
    ``axes``)."""
    _check(stats)
    found, codes = dense_codes(ids, keep)
    return _measure(IdStatistics(found), codes, positions, time_s, frame,
                    columns, stats, percentiles, axes, shape)


def member_statistics(members: Sequence[Tuple[Any, np.ndarray]],
                      positions: Dict[str, np.ndarray], *, keep=None,
                      time_s=None, frame=None,
                      columns: Optional[Dict[str, np.ndarray]] = None,
                      stats: Sequence[str] = ("mean", "median", "max"),
                      percentiles: Sequence[float] = (),
                      axes: Optional[Sequence[str]] = None,
                      shape: bool = False) -> IdStatistics:
    """`id_statistics` of regions given as their rows, ``[(id, rows), ...]``:
    the ROIs of the ROI manager, which may overlap, so no id column can name
    them.  The row lists are joined and every array is read through them --
    a localization in two ROIs is counted in both -- and then measured by the
    same pass as an id column's regions, so the numbers are the same as for
    an id column holding the same partition.  A region with no row ``keep``
    keeps is left out, as an id with none is."""
    _check(stats)
    keep = None if keep is None else np.asarray(keep, dtype=bool)
    ids, parts = [], []
    for region_id, rows in members:
        rows = np.asarray(rows, dtype=np.int64)
        if keep is not None:
            rows = rows[keep[rows]]
        if len(rows):
            ids.append(region_id)
            parts.append(rows)
    index = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    codes = np.repeat(np.arange(len(parts), dtype=np.int64),
                      [len(r) for r in parts])

    def take(v):
        return None if v is None else np.asarray(v)[index]
    return _measure(IdStatistics(np.array(ids, dtype=object)), codes,
                    {a: take(v) for a, v in positions.items()},
                    take(time_s), take(frame),
                    {n: take(v) for n, v in (columns or {}).items()}, stats,
                    percentiles, axes, shape)


def _check(stats: Sequence[str]) -> None:
    unknown = [s for s in stats if s not in STATISTICS]
    if unknown:
        raise ValueError(f"unknown statistics {unknown}: choose among {STATISTICS}")


def _measure(out: IdStatistics, codes: np.ndarray, positions, time_s, frame,
             columns, stats, percentiles=(), axes=None, shape=False) -> IdStatistics:
    """Every value of the regions ``codes`` names (0 .. len(out) - 1 per row,
    -1 for none, each region with a row), into ``out``."""
    n = len(out)
    regions = _Regions(codes, n)
    out.values["n_locs"] = regions.counts.astype(np.int64)
    axes = [a for a in ("x", "y", "z") if a in positions
            and (axes is None or a in axes)]
    extra = [s for s in stats if s not in ("mean", "std")]
    std = {}
    for axis in ("x", "y", "z"):
        if axis not in positions:
            continue
        # the centre always; x and y's spread for std_xy, the region's size
        wanted = axis in axes or axis in "xy"
        s = regions.statistics(positions[axis], ("mean", "std") if wanted else ("mean",))
        out.values[f"centre_{axis}_nm"] = s["mean"]
        if wanted:
            std[axis] = s["std"]
        if axis in axes:
            out.values[f"std_{axis}_nm"] = s["std"]
            more = regions.statistics(positions[axis], extra, percentiles)
            if more:
                out.groups[f"{axis}_nm"] = {}
            for stat, value in more.items():
                key = stat_name(f"{axis}_nm", stat)
                out.values[key] = value
                out.groups[f"{axis}_nm"][stat] = key
    if "x" in std and "y" in std:
        out.values["std_xy_nm"] = np.sqrt(0.5 * (std["x"] ** 2 + std["y"] ** 2))
    out.groups["std"] = {a: f"std_{a}_nm" for a in ("x", "y", "z", "xy")
                         if f"std_{a}_nm" in out.values}
    out.groups["centre"] = {a: f"centre_{a}_nm" for a in ("x", "y", "z")
                            if f"centre_{a}_nm" in out.values}
    for name in ("std", "centre"):
        if not out.groups[name]:
            del out.groups[name]
    if shape:
        found_shape = region_shapes(codes, n, positions["x"], positions["y"],
                                    positions.get("z") if "z" in axes else None)
        for name in SHAPE_2D + (SHAPE_3D if "z" in axes else ()):
            out.values[name] = found_shape[name]
    if time_s is not None:
        s = regions.statistics(time_s, ("min", "max"))
        out.values["start_s"], out.values["end_s"] = s["min"], s["max"]
        out.values["duration_s"] = s["max"] - s["min"]
    if frame is not None:
        s = regions.statistics(frame, ("min", "max"))
        first, last = s["min"].astype(np.int64), s["max"].astype(np.int64)
        out.values["first_frame"], out.values["last_frame"] = first, last
        out.values["frames"] = last - first + 1
    for name, column in (columns or {}).items():
        s = regions.statistics(column, tuple(stats), percentiles)
        out.groups[name] = {}
        for stat, value in s.items():
            key = stat_name(name, stat)
            out.values[key] = value
            out.groups[name][stat] = key
    return out
