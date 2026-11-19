"""Numbers per trace (or track): its geometry, its precision, and its start.

A MINFLUX trace (`tid`), a linked track (`track_id`) or any id column's
regions, measured all at once: the rows sorted once by id and time, every
quantity a ``reduceat`` or a ``bincount`` over the runs, no loop over
traces.  *Trace statistics* (`plugins/trace_statistics.py`) is the plugin;
*Steps* and the *Trace inspector* call the same functions.

**The track's axes.**  A trace's covariance (x, y) has a long and a short
axis: the standard deviations along them (``std_long_nm``,
``std_short_nm``), their ratio (``eccentricity``) and the long axis'
direction (``angle_deg``, 0-360, pointing the way the trace moved: from its
first localizations to its last).  `rotate` turns each trace onto them --
the long axis on x, forward in time, the centre at 0 -- as a motor's track
is analysed (MotorPAINT, StepsMINFLUX, the MFX scripts' `rotate_all_tracks`).

**The precision from consecutive differences**: std(diff) / sqrt(2) per
axis, the scatter between neighbouring localizations of a trace -- for a
still molecule the localization precision, for a moving one the precision
blurred by what it moved between two localizations.  On the track's axes
too (``precision_short_nm``), where a straight motor's short axis holds
only the scatter.  ``trend_nm`` = sqrt(max(0, std_short^2 -
precision_short^2)) is what the short axis' spread holds beyond that
scatter: 0 for a straight track, the wander of a curved one (the MFX
scripts' ``trace_trend_yrot_nm``).

**The start of a trace** -- a still molecule's first localizations creep
into place, 5 to 30 of them in 10-30 % of traces (NOTES.md) -- is labelled
by one of two rules (`tails`):

* *still*: the trace's settled position is the median of its localizations
  ``reference_from`` to ``reference_from + reference_window``; the tail is
  every localization before the first one within ``k`` times the robust
  spread (1.4826 x the median absolute distance) of that window's distances
  from it -- the MFX scripts' rule for static molecules, with SMAP's
  Tails_analyze's median as the position;
* *moving*: the rolling standard deviation of the steps between
  consecutive localizations (``rolling_window`` of them) settles below
  ``rolling_limit_nm``; the tail is everything before (the MFX scripts'
  kinesin rule, 10 and 7 nm).

`cfr_cut` cuts a trace after its first ``cfr`` above a limit (the MFX
scripts' post-hoc filter): the rows after it are flagged, not the start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: the median absolute deviation of a normal distribution, in sigmas
MAD = 1.4826


@dataclass
class Runs:
    """The rows of each trace, sorted by trace then time: ``order`` (row
    indices), ``ids`` (one per trace), ``start`` (each trace's first place
    in ``order``), ``n`` (its rows), ``code`` (each sorted row's trace,
    0..len(ids)-1) and ``rank`` (each sorted row's place in its trace)."""
    order: np.ndarray
    ids: np.ndarray
    start: np.ndarray
    n: np.ndarray
    code: np.ndarray
    rank: np.ndarray

    def sums(self, v: np.ndarray) -> np.ndarray:
        return np.add.reduceat(v, self.start) if len(self.start) else np.zeros(0)

    def first(self, v: np.ndarray) -> np.ndarray:
        return v[self.start]

    def last(self, v: np.ndarray) -> np.ndarray:
        return v[self.start + self.n - 1]

    def spread(self, per_trace: np.ndarray) -> np.ndarray:
        """A value per trace on each of its sorted rows."""
        return per_trace[self.code]


def runs(ids, time, keep=None) -> Runs:
    """The traces of ``ids`` among the rows ``keep`` (a mask; -1 is none,
    `ids.NONE`), each in the order of ``time``."""
    ids = np.asarray(ids, np.int64)
    time = np.asarray(time, np.float64)
    use = ids >= 0
    if keep is not None:
        use &= np.asarray(keep, bool)
    rows = np.flatnonzero(use)
    order = rows[np.lexsort((time[rows], ids[rows]))]
    sid = ids[order]
    start = (np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1] if len(order)
             else np.zeros(0, np.int64))
    n = np.diff(np.r_[start, len(order)]).astype(np.int64)
    code = np.repeat(np.arange(len(start)), n)
    rank = np.arange(len(order)) - np.repeat(start, n)
    return Runs(order, sid[start] if len(order) else np.zeros(0, np.int64), start, n,
                code, rank)


def _var(r: Runs, v: np.ndarray, use: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per trace, the mean and sample variance (ddof 1) of ``v`` (sorted
    rows) over the rows ``use``; NaN below two."""
    use = np.isfinite(v) if use is None else use & np.isfinite(v)
    w = use.astype(np.float64)
    x = np.where(use, v, 0.0)
    m = r.sums(w)
    s1, s2 = r.sums(x), r.sums(x * x)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / m
        var = (s2 - s1 * mean) / (m - 1)
    var = np.where(m > 1, np.maximum(var, 0.0), np.nan)
    return mean, var


def _cov(r: Runs, x: np.ndarray, y: np.ndarray, use: np.ndarray) -> np.ndarray:
    use = use & np.isfinite(x) & np.isfinite(y)
    w = use.astype(np.float64)
    m = r.sums(w)
    sx, sy = r.sums(np.where(use, x, 0.0)), r.sums(np.where(use, y, 0.0))
    sxy = r.sums(np.where(use, x * y, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > 1, (sxy - sx * sy / m) / (m - 1), np.nan)


def axes(r: Runs, x: np.ndarray, y: np.ndarray, use: Optional[np.ndarray] = None
         ) -> Dict[str, np.ndarray]:
    """Per trace, the long and short axis of its (x, y) covariance:
    ``std_long_nm``, ``std_short_nm``, ``eccentricity`` (their ratio) and
    ``angle_deg``, the long axis pointing from the trace's first quarter to
    its last (0 along +x, counter-clockwise, 0-360); ``x`` and ``y`` sorted
    as ``r``."""
    use = np.ones(len(x), bool) if use is None else use
    mx, vx = _var(r, x, use)
    my, vy = _var(r, y, use)
    cxy = _cov(r, x, y, use)
    tr, det = vx + vy, vx * vy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    big, small = tr / 2 + disc, np.maximum(tr / 2 - disc, 0.0)
    theta = 0.5 * np.arctan2(2 * cxy, vx - vy)
    # forward in time: the long axis turned to point from the start to the end
    q = np.maximum(r.n // 4, 1)
    first = _mean_of_ranks(r, x, y, lambda rank, n, k: rank < k, q)
    last = _mean_of_ranks(r, x, y, lambda rank, n, k: rank >= n - k, q)
    dx, dy = last[0] - first[0], last[1] - first[1]
    back = (np.cos(theta) * dx + np.sin(theta) * dy) < 0
    theta = np.where(back, theta + np.pi, theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"centre_x_nm": mx, "centre_y_nm": my,
                "std_long_nm": np.sqrt(big), "std_short_nm": np.sqrt(small),
                "eccentricity": np.sqrt(big) / np.sqrt(small),
                "angle_deg": np.degrees(theta) % 360.0}


def _mean_of_ranks(r: Runs, x, y, pick, k) -> Tuple[np.ndarray, np.ndarray]:
    sel = pick(r.rank, r.spread(r.n), r.spread(k)) & np.isfinite(x) & np.isfinite(y)
    m = np.maximum(r.sums(sel.astype(float)), 1)
    return r.sums(np.where(sel, x, 0)) / m, r.sums(np.where(sel, y, 0)) / m


def moving_direction(t, x, y, window: float, min_speed: float) -> Tuple[float, float]:
    """The direction a motor walks in, from the parts of its trace where it
    moves: ``(angle_deg, moving_share)``.

    The trace is cut into windows of ``window`` (in the units of ``t``) and
    each window's mean position taken -- a few tens of localizations, so a
    nanometre or two of noise; between neighbouring windows the motor moved
    ``d``, and a window pair moves when ``|d| / window >= min_speed``.  The
    direction is that of the sum of the moving pairs' ``d``: the stuck parts,
    where the covariance's long axis is pulled by noise and by any slow
    wander, count for nothing, and a motor that pauses for half its trace is
    still turned along where it walked.  NaN where no pair moves;
    ``moving_share`` the share of the pairs that did."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    if len(t) < 4 or not window > 0:
        return np.nan, np.nan
    k = np.floor((t - t.min()) / window).astype(np.int64)
    n = np.bincount(k)
    sx, sy = np.bincount(k, x), np.bincount(k, y)
    full = n >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sy / n
    pair = full[:-1] & full[1:]
    if not pair.any():
        return np.nan, np.nan
    dx, dy = np.diff(mx)[pair], np.diff(my)[pair]
    moving = np.hypot(dx, dy) / window >= min_speed
    share = float(moving.mean())
    if not moving.any():
        return np.nan, share
    vx, vy = dx[moving].sum(), dy[moving].sum()
    if vx == 0 and vy == 0:
        return np.nan, share
    return float(np.degrees(np.arctan2(vy, vx)) % 360.0), share


def rotate(r: Runs, x: np.ndarray, y: np.ndarray, a: Dict[str, np.ndarray]
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Each sorted row turned onto its trace's axes (`axes`): x along the
    long axis, forward in time, y across it, the centre at 0."""
    theta = np.radians(r.spread(a["angle_deg"]))
    dx, dy = x - r.spread(a["centre_x_nm"]), y - r.spread(a["centre_y_nm"])
    c, s = np.cos(theta), np.sin(theta)
    return c * dx + s * dy, -s * dx + c * dy


def consecutive_precision(r: Runs, v: np.ndarray, use: Optional[np.ndarray] = None
                          ) -> np.ndarray:
    """Per trace, std(diff(v)) / sqrt(2) over consecutive rows (both in
    ``use``); NaN with fewer than three rows."""
    use = np.ones(len(v), bool) if use is None else use
    d = np.r_[np.diff(v), np.nan]
    pair = use & np.r_[use[1:], False] & (np.r_[r.code[1:], -1] == r.code)
    _, var = _var(r, d, pair)
    return np.sqrt(var / 2.0)


def steps(r: Runs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each sorted row's distance from the trace's previous localization;
    NaN at a trace's first."""
    d = np.hypot(np.r_[np.nan, np.diff(x)], np.r_[np.nan, np.diff(y)])
    d[r.start] = np.nan
    return d


def rolling_std(r: Runs, v: np.ndarray, window: int) -> np.ndarray:
    """Each sorted row's standard deviation of ``v`` over the ``window`` rows
    of its trace centred on it (fewer at the ends; NaN entries left out, NaN
    below 3)."""
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    c0 = np.r_[0, np.cumsum(ok)]
    c1 = np.r_[0, np.cumsum(x)]
    c2 = np.r_[0, np.cumsum(x * x)]
    i = np.arange(len(v))
    lo = np.maximum(i - window // 2, np.repeat(r.start, r.n))
    hi = np.minimum(i + (window - window // 2), np.repeat(r.start + r.n, r.n))
    m = c0[hi] - c0[lo]
    s1, s2 = c1[hi] - c1[lo], c2[hi] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / m) / (m - 1)
    return np.where(m >= 3, np.sqrt(np.maximum(var, 0.0)), np.nan)


def _first_true(r: Runs, flag: np.ndarray) -> np.ndarray:
    """Per trace, the rank of its first row with ``flag``; its length where
    there is none."""
    rank = np.where(flag, r.rank, np.iinfo(np.int64).max)
    first = np.minimum.reduceat(rank, r.start) if len(r.start) else np.zeros(0, np.int64)
    return np.where(first == np.iinfo(np.int64).max, r.n, first)


@dataclass
class Tails:
    """Per sorted row ``in_tail``; per trace the tail's length in
    localizations (``n``), how far its first point was from the settled
    position (``reach_nm``), how long it lasted (``duration``, in the units
    of the time given) and the settled position (``x``, ``y``); and per
    sorted row its distance from that position (``distance``)."""
    in_tail: np.ndarray
    n: np.ndarray
    reach_nm: np.ndarray
    duration: np.ndarray
    x: np.ndarray
    y: np.ndarray
    distance: np.ndarray


def tails(r: Runs, x: np.ndarray, y: np.ndarray, time: np.ndarray, rule: str = "still",
          reference_from: int = 20, reference_window: int = 30, k: float = 3.0,
          rolling_window: int = 10, rolling_limit_nm: float = 7.0) -> Tails:
    """Each trace's start, by ``rule`` (see the module docstring); ``x``,
    ``y``, ``time`` sorted as ``r``.  A trace too short for its reference
    window has no tail."""
    n = len(x)
    ref = (r.rank >= reference_from) & (r.rank < reference_from + reference_window)
    ref &= np.isfinite(x) & np.isfinite(y)
    cx = _median_per(r, x, ref)
    cy = _median_per(r, y, ref)
    distance = np.hypot(x - r.spread(cx), y - r.spread(cy))
    if rule == "still":
        spread = MAD * _median_per(r, distance, ref)
        near = distance <= k * r.spread(spread)
        settled_at = _first_true(r, near & np.isfinite(r.spread(spread)))
        has_ref = np.isfinite(spread)
    elif rule == "moving":
        roll = rolling_std(r, steps(r, x, y), rolling_window)
        settled_at = _first_true(r, roll <= rolling_limit_nm)
        has_ref = settled_at < r.n
    elif rule == "none":
        settled_at = np.zeros(len(r.start), np.int64)
        has_ref = np.zeros(len(r.start), bool)
    else:
        raise ValueError(f"no tail rule {rule!r}: still, moving or none")
    settled_at = np.where(has_ref, settled_at, 0)
    in_tail = r.rank < r.spread(settled_at)
    reach = np.where(settled_at > 0, distance[r.start], 0.0)
    t = np.asarray(time, np.float64)
    end = r.start + np.maximum(settled_at, 1) - 1
    duration = np.where(settled_at > 0, t[np.minimum(end + 1, n - 1)] - t[r.start], 0.0) \
        if n else np.zeros(0)
    return Tails(in_tail, settled_at, reach, duration, cx, cy, distance)


def _median_per(r: Runs, v: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Per trace, the median of ``v`` over its rows ``use``; NaN where none."""
    out = np.full(len(r.start), np.nan)
    rows = np.flatnonzero(use & np.isfinite(v))
    if not len(rows):
        return out
    codes = r.code[rows]
    order = np.lexsort((v[rows], codes))
    codes, vals = codes[order], v[rows][order]
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(codes)])
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    out[codes[starts]] = 0.5 * (vals[lo] + vals[hi])
    return out


def nan_means(r: Runs, v: np.ndarray) -> np.ndarray:
    """Each run's mean of ``v`` (sorted as ``r.order``), NaN left out --
    what merging the trace gives; NaN where the run has none."""
    v = np.asarray(v, np.float64)
    finite = np.isfinite(v)
    count = r.sums(finite.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r.sums(np.where(finite, v, 0.0)) / count


def cfr_cut(r: Runs, cfr: np.ndarray, limit: float) -> np.ndarray:
    """Each sorted row after its trace's first ``cfr`` above ``limit`` (the
    row itself included): True.  NaN cfr (not measured) never cuts."""
    bad = np.nan_to_num(np.asarray(cfr, np.float64), nan=-np.inf) > limit
    first = _first_true(r, bad)
    return r.rank >= r.spread(first)


def statistics(r: Runs, x: np.ndarray, y: np.ndarray, time: np.ndarray,
               use: Optional[np.ndarray] = None, photons: Optional[np.ndarray] = None
               ) -> Dict[str, np.ndarray]:
    """Per trace, over its sorted rows ``use`` (default all): ``n_locs``,
    displacement from the first to the last (``displacement_nm``), duration,
    velocity (displacement over duration, nm per unit of ``time``), the
    track's axes (`axes`), the precision from consecutive differences in x
    and y and along the axes, ``trend_nm``, the median time between
    localizations (``dt_median``) and the photons' sum."""
    use = np.ones(len(x), bool) if use is None else use
    use = use & np.isfinite(x) & np.isfinite(y)
    n = r.sums(use.astype(float))
    first = _first_true(r, use)
    last = (r.n - 1) - _first_true_reversed(r, use)
    i0 = r.start + np.minimum(first, r.n - 1)
    i1 = r.start + np.maximum(last, 0)
    disp = np.hypot(x[i1] - x[i0], y[i1] - y[i0])
    t = np.asarray(time, np.float64)
    duration = t[i1] - t[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.where(duration > 0, disp / duration, np.nan)
    out = {"n_locs": n, "displacement_nm": np.where(n > 1, disp, np.nan),
           "duration": duration, "velocity": velocity}
    a = axes(r, x, y, use)
    out.update(a)
    xr, yr = rotate(r, x, y, a)
    out["precision_x_nm"] = consecutive_precision(r, x, use)
    out["precision_y_nm"] = consecutive_precision(r, y, use)
    out["precision_long_nm"] = consecutive_precision(r, xr, use)
    out["precision_short_nm"] = consecutive_precision(r, yr, use)
    out["trend_nm"] = np.sqrt(np.maximum(out["std_short_nm"] ** 2
                                         - out["precision_short_nm"] ** 2, 0.0))
    dt = np.r_[np.diff(t), np.nan]
    pair = use & np.r_[use[1:], False] & (np.r_[r.code[1:], -1] == r.code)
    out["dt_median"] = _median_per(r, dt, pair)
    if photons is not None:
        out["photons_sum"] = r.sums(np.where(use, np.nan_to_num(photons), 0.0))
    return out


def _first_true_reversed(r: Runs, flag: np.ndarray) -> np.ndarray:
    """Per trace, how many rows from its end the last ``flag`` is (0: the
    last row); its length where there is none."""
    back = r.spread(r.n) - 1 - r.rank
    rank = np.where(flag, back, np.iinfo(np.int64).max)
    first = np.minimum.reduceat(rank, r.start) if len(r.start) else np.zeros(0, np.int64)
    return np.where(first == np.iinfo(np.int64).max, r.n, first)


def convergence(r: Runs, distance: np.ndarray, up_to: int,
                photons: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Pooled over the traces, the distance of each trace's n-th
    localization from its settled position, n = 1 .. ``up_to``: mean and
    median, and the mean cumulative photons by then -- SMAP's Tails_analyze's
    convergence curve."""
    index = np.arange(1, up_to + 1)
    mean = np.full(up_to, np.nan)
    median = np.full(up_to, np.nan)
    cum = np.full(up_to, np.nan)
    if photons is not None:
        p = np.nan_to_num(np.asarray(photons, np.float64))
        c = np.cumsum(p)
        before = np.r_[0.0, c][r.start]
        cumulative = c - np.repeat(before, r.n)
    for j in range(up_to):
        sel = (r.rank == j) & np.isfinite(distance)
        if sel.any():
            mean[j] = distance[sel].mean()
            median[j] = np.median(distance[sel])
            if photons is not None:
                cum[j] = cumulative[sel].mean()
    return {"index": index, "mean": mean, "median": median, "photons": cum}
