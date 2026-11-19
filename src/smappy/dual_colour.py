"""Two-colour MINFLUX: which traces are one molecule in both colours, and
how far apart the colours are.

The colour of a localization is its laser (``channel``, `io.msr`), else
its thread (``thi``); colour 1 is moved onto colour 0.  Everything here
takes arrays, so a script gets the plugin's numbers without a session.

**Pairing** (`link_traces`).  A colour-0 trace and a colour-1 trace are a
candidate when they overlap in time.  Each colour-1 localization is matched
with the colour-0 one nearest in time (within ``max_dt``), and the pair is
accepted when enough localizations match (``min_matched``) and enough of
the matches lie within ``max_distance`` of each other once the colour
offset is taken off (``min_fraction``).  The fraction, not the median,
because two trajectories that cross are close for a moment: their median
distance can pass, most of their matches cannot.  Pairs are joined
greedily, the most matches within the distance first, into groups -- one
``pair_id`` each -- so that a 560 trace that stopped and started again on
the same kinesin joins the one 638 trace (and the other way round), as long
as two traces of one colour in a group never overlap in time: two at once
are two molecules.  A trace with no partner is a group of its own.

**The offset** is ``offset(t) = c + delta(t)``: a constant from beads or
from the pairs, plus the colours' drift against each other over time from
the MBM beads.

* From **beads** in a measurement of their own (`bead_offset`): the old
  pairing by time (`pairing.pair_traces`) and each pair's median offset
  (`pairing.pair_offsets`), their median.  One measurement before the
  tracking gives ``c``; one before and one after are interpolated in time
  (`offset_curve`), with the MBM drift as the shape between them and the
  part it does not explain spread linearly -- that part is the check.
* From the **pairs themselves** (`fit_track_offset`): each pair's median
  offset ``d`` is modelled as ``c + a u``, ``u`` the direction the pair
  moves in (zero for a pair that stays put), so the registration ``c`` and
  ``a``, the second colour's lead along the track (half a step for two
  heads of a kinesin, the label geometry for others), are told apart -- the
  more directions the tracks take, the better.  Robust (Tukey's biweight
  over the pairs), since a few chance pairs are always accepted.
* **MBM colour drift** (`mbm_colour_drift`): each MBM bead is recorded in
  both colours, a bias apart that differs from bead to bead (tens of nm in
  the file looked at, 2026-11) and changes slowly (1-2 nm per 100 s).  The
  bias is useless, the change is not: each bead's colour difference is a
  track with an offset of its own, and `beads.select_beads` fits one curve
  through them all, the beads that disagree left out.  Zero at its first
  point, held flat beyond its last.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: the robust fit's tuning constant (Tukey's biweight, in robust SDs)
TUKEY_C = 4.685
#: the robust spread is never taken below this (nm): beads and pairs can agree
#: to far better than they are known
MIN_SCALE_NM = 1.0


def colour_column(locs) -> str:
    """``channel`` where the table has two colours in it, else ``thi``."""
    if "channel" in locs and len(np.unique(np.asarray(locs["channel"]))) > 1:
        return "channel"
    return "thi"


def colour_names(locs, column: Optional[str] = None) -> Dict[int, str]:
    """What each colour is called: the laser (``638 nm``) where the reader
    wrote what each ``channel`` is (`io.msr`), else ``thi 0`` / ``channel 0``."""
    from .metadata import file_metadata
    column = column or colour_column(locs)
    out: Dict[int, str] = {}
    if column == "channel":
        files = (np.unique(np.asarray(locs["filenumber"])) if "filenumber" in locs
                 else np.zeros(1, np.int64))
        for k in files:
            mf = file_metadata(locs.metadata, int(k)).get("minflux")
            for m in (mf.get("measurements") or []) if isinstance(mf, dict) else []:
                for c in m.get("channels") or []:
                    out.setdefault(int(c["channel"]), str(c["name"]))
    if column in locs:
        for c in np.unique(np.asarray(locs[column], np.int64)):
            out.setdefault(int(c), f"{column} {int(c)}")
    return out


# ------------------------------------------------------------------- traces

@dataclass
class Traces:
    """The traces of two colours, each with its rows in time order."""
    file: np.ndarray            # per trace
    colour: np.ndarray
    tid: np.ndarray
    start: np.ndarray           # first and last time
    end: np.ndarray
    order: np.ndarray           # the rows, trace by trace, each in time order
    first: np.ndarray           # where each trace's rows begin in ``order``
    count: np.ndarray
    trace: np.ndarray           # per row: its trace, -1 for none

    def rows(self, k: int) -> np.ndarray:
        return self.order[self.first[k]:self.first[k] + self.count[k]]


def traces(tid, colour, time, x, y, file=None) -> Traces:
    """Every trace of colour 0 and 1 (per file), rows without a trace, a
    colour or a position left out."""
    tid = np.asarray(tid, np.int64)
    colour = np.asarray(colour, np.int64)
    t = np.asarray(time, np.float64)
    f = np.zeros(len(tid), np.int64) if file is None else np.asarray(file, np.int64)
    use = ((tid >= 0) & ((colour == 0) | (colour == 1))
           & np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float)))
    rows = np.flatnonzero(use)
    keys = np.column_stack([f[rows], colour[rows], tid[rows]])
    found, inverse = (np.unique(keys, axis=0, return_inverse=True) if len(rows)
                      else (np.zeros((0, 3), np.int64), np.zeros(0, np.int64)))
    inverse = np.asarray(inverse).reshape(-1)
    trace = np.full(len(tid), -1, np.int64)
    trace[rows] = inverse
    order = rows[np.lexsort((t[rows], inverse))]
    count = np.bincount(inverse, minlength=len(found)).astype(np.int64)
    first = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    start = t[order[first]] if len(found) else np.zeros(0)
    end = t[order[first + count - 1]] if len(found) else np.zeros(0)
    return Traces(found[:, 0], found[:, 1], found[:, 2], start, end, order, first, count,
                  trace)


def _nearest(t_ref: np.ndarray, t: np.ndarray) -> np.ndarray:
    """For each of ``t``, the index in the sorted ``t_ref`` nearest in time."""
    at = np.clip(np.searchsorted(t_ref, t), 1, max(len(t_ref) - 1, 1))
    before = at - 1
    after = np.minimum(at, len(t_ref) - 1)
    return np.where(np.abs(t_ref[before] - t) <= np.abs(t_ref[after] - t), before, after)


def match(ra: np.ndarray, rb: np.ndarray, t: np.ndarray, max_dt: float
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Each row of ``rb`` with the row of ``ra`` (time-sorted) nearest in
    time, where within ``max_dt``: the matched rows of both."""
    if not len(ra) or not len(rb):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    near = ra[_nearest(t[ra], t[rb])]
    keep = np.abs(t[near] - t[rb]) <= max_dt
    return near[keep], rb[keep]


# ------------------------------------------------------------------ pairing

@dataclass
class Links:
    """`link_traces`' answer: ``pair_id`` per row (-1 none) and, per pair,
    what it was judged on.  The per-pair arrays are in ``pair_id`` order,
    1, 2, ...; a pair in one colour has NaN where both are needed."""
    pair_id: np.ndarray
    ids: np.ndarray
    n0: np.ndarray                  # localizations per colour
    n1: np.ndarray
    n_traces: np.ndarray            # traces joined
    n_matched: np.ndarray           # colour-1 localizations matched in time
    within: np.ndarray              # the fraction of the matches within the distance
    separation_nm: np.ndarray       # their median distance
    dx: np.ndarray                  # the median offset, colour 1 - colour 0, as seen
    dy: np.ndarray
    time: np.ndarray                # the matches' mean time
    x: np.ndarray                   # colour 0's mean position
    y: np.ndarray
    ux: np.ndarray                  # the direction colour 0 moves in (unit)
    uy: np.ndarray
    displacement_nm: np.ndarray     # how far it moves along it
    velocity_nm_per_s: np.ndarray

    @property
    def both(self) -> np.ndarray:
        return (self.n0 > 0) & (self.n1 > 0)

    def values(self) -> Dict[str, np.ndarray]:
        """The per-pair numbers as region-table columns."""
        return {"both": self.both.astype(float), "n_traces": self.n_traces.astype(float),
                "n_0": self.n0.astype(float), "n_1": self.n1.astype(float),
                "n_matched": self.n_matched.astype(float), "within": self.within,
                "separation_nm": self.separation_nm, "dx_nm": self.dx, "dy_nm": self.dy,
                "displacement_nm": self.displacement_nm,
                "velocity_nm_per_s": self.velocity_nm_per_s}


def direction(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[float, float, float]:
    """The long axis of a track, turned forward in time, and how far it
    moves along it: the median of its last fifth less that of its first
    (the ends' noise averaged, not their extremes taken)."""
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    p = np.column_stack([x - x.mean(), y - y.mean()])
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    u = vt[0]
    s = p @ u
    order = np.argsort(t)
    k = max(1, len(s) // 5)
    moved = np.median(s[order[-k:]]) - np.median(s[order[:k]])
    if moved < 0:
        u, moved = -u, -moved
    return float(u[0]), float(u[1]), float(moved)


def link_traces(tid, colour, time, x, y, file=None, *, max_dt: float = 0.005,
                max_distance: float = 30.0, min_matched: int = 10,
                min_fraction: float = 0.5, min_overlap: float = 0.0,
                offset: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Links:
    """The traces of both colours joined into molecules (module docstring).
    ``offset(t)``: colour 1 - colour 0 (n, 2) nm, taken off colour 1 before
    distances are measured; None: none."""
    t = np.asarray(time, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    colour = np.asarray(colour, np.int64)
    tr = traces(tid, colour, t, x, y, file)
    xb, yb = x.copy(), y.copy()
    ones = colour == 1
    if offset is not None and ones.any():
        sh = np.asarray(offset(t[ones]), float).reshape(-1, 2)
        xb[ones] -= sh[:, 0]
        yb[ones] -= sh[:, 1]
    m = len(tr.tid)
    cands = []
    for f in np.unique(tr.file):
        a = np.flatnonzero((tr.file == f) & (tr.colour == 0))
        b = np.flatnonzero((tr.file == f) & (tr.colour == 1))
        if not len(a) or not len(b):
            continue
        b = b[np.argsort(tr.start[b])]
        for i in a:
            hi = np.searchsorted(tr.start[b], tr.end[i] + max_dt, side="right")
            for j in b[:hi][tr.end[b[:hi]] >= tr.start[i] - max_dt]:
                overlap = min(tr.end[i], tr.end[j]) - max(tr.start[i], tr.start[j])
                if overlap < min_overlap:
                    continue
                na, nb = match(tr.rows(i), tr.rows(j), t, max_dt)
                if len(nb) < max(min_matched, 1):
                    continue
                d = np.hypot(xb[nb] - x[na], yb[nb] - y[na])
                close = d <= max_distance
                if close.mean() < min_fraction:
                    continue
                cands.append((int(close.sum()), float(-np.median(d)), i, j))
    # greedy, the most close matches first; a group never holds two traces
    # of one colour at the same time
    parent = np.arange(m)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    members: Dict[int, List[int]] = {k: [k] for k in range(m)}
    for _, _, i, j in sorted(cands, key=lambda c: (-c[0], -c[1], c[2], c[3])):
        gi, gj = root(i), root(j)
        if gi == gj:
            continue
        clash = any(tr.colour[p] == tr.colour[q]
                    and max(tr.start[p], tr.start[q]) < min(tr.end[p], tr.end[q])
                    for p in members[gi] for q in members[gj])
        if clash:
            continue
        parent[gj] = gi
        members[gi] += members.pop(gj)
    groups = sorted(members.values(), key=lambda g: min(tr.start[k] for k in g))
    pid_of_trace = np.empty(m, np.int64)
    for n, g in enumerate(groups):
        pid_of_trace[g] = n + 1
    pair_id = np.full(len(t), -1, np.int64)
    has = tr.trace >= 0
    pair_id[has] = pid_of_trace[tr.trace[has]]

    n = len(groups)
    nan = lambda: np.full(n, np.nan)    # noqa: E731
    out = Links(pair_id, np.arange(1, n + 1), np.zeros(n, np.int64), np.zeros(n, np.int64),
                np.array([len(g) for g in groups], np.int64), np.zeros(n, np.int64),
                nan(), nan(), nan(), nan(), nan(), nan(), nan(), nan(), nan(), nan(), nan())
    for k, g in enumerate(groups):
        r0 = np.concatenate([tr.rows(q) for q in g if tr.colour[q] == 0] or [np.zeros(0, int)])
        r1 = np.concatenate([tr.rows(q) for q in g if tr.colour[q] == 1] or [np.zeros(0, int)])
        out.n0[k], out.n1[k] = len(r0), len(r1)
        if len(r0):
            r0 = r0[np.argsort(t[r0], kind="stable")]
            out.x[k], out.y[k] = x[r0].mean(), y[r0].mean()
            ux, uy, moved = direction(x[r0], y[r0], t[r0])
            out.ux[k], out.uy[k], out.displacement_nm[k] = ux, uy, moved
            span = t[r0[-1]] - t[r0[0]]
            if span > 0 and np.isfinite(moved):
                out.velocity_nm_per_s[k] = moved / span
        if not len(r0) or not len(r1):
            continue
        na, nb = match(r0, r1, t, max_dt)
        out.n_matched[k] = len(nb)
        if not len(nb):
            continue
        d = np.hypot(xb[nb] - x[na], yb[nb] - y[na])
        out.within[k] = float(np.mean(d <= max_distance))
        out.separation_nm[k] = float(np.median(d))
        out.dx[k] = float(np.median(x[nb] - x[na]))
        out.dy[k] = float(np.median(y[nb] - y[na]))
        out.time[k] = float(t[nb].mean())
    return out


# ------------------------------------------------------------------- offset

@dataclass
class TrackOffset:
    """`fit_track_offset`'s answer."""
    c: np.ndarray                   # the registration, colour 1 - colour 0 (nm)
    a: float                        # colour 1's lead along the direction of motion (nm)
    c_err: np.ndarray               # their standard errors
    a_err: float
    used: np.ndarray                # per pair given: kept by the robust fit
    residual: np.ndarray            # per pair (n, 2): d - c - a u
    n_moving: int
    n_still: int
    scale_nm: float                 # the robust spread of the residuals


def fit_track_offset(dx, dy, ux, uy, moving, iterations: int = 30) -> TrackOffset:
    """``d = c + a u`` over the pairs (module docstring): ``dx, dy`` each
    pair's offset (colour 1 - colour 0, with any drift over time already
    taken off), ``u`` its direction of motion, ``moving`` whether it moves
    (a still pair says nothing of ``a``).  Tukey's biweight on each pair's
    residual length, its scale the residuals' robust spread."""
    d = np.column_stack([np.asarray(dx, float), np.asarray(dy, float)])
    u = np.column_stack([np.asarray(ux, float), np.asarray(uy, float)])
    mv = np.asarray(moving, bool) & np.all(np.isfinite(u), axis=1)
    ok = np.all(np.isfinite(d), axis=1)
    n = len(d)
    if not ok.any():
        raise ValueError("no pair in both colours to measure the offset in")
    u = np.where(mv[:, None], u, 0.0)
    # a lead is only told from the registration when the tracks move in
    # more than one direction: u and -u, or two at an angle
    spread = (np.linalg.eigvalsh(np.cov(u[ok & mv].T)).max()
              if (ok & mv).sum() > 2 else 0.0)
    mean_u = np.abs(u[ok & mv].mean(axis=0)) if (ok & mv).any() else np.zeros(2)
    with_a = (ok & mv).sum() >= 3 and (spread > 0.05 or np.all(mean_u < 0.8))
    cols = 3 if with_a else 2
    w = ok.astype(float)
    beta = np.zeros(cols)
    scale = MIN_SCALE_NM
    for _ in range(iterations):
        A = np.zeros((2 * n, cols))
        A[0::2, 0] = 1.0
        A[1::2, 1] = 1.0
        if with_a:
            A[0::2, 2] = u[:, 0]
            A[1::2, 2] = u[:, 1]
        b = np.nan_to_num(d).reshape(-1)
        ww = np.repeat(w, 2)
        sw = np.sqrt(ww)
        new, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        res = (b - A @ new).reshape(-1, 2)
        r = np.hypot(res[:, 0], res[:, 1])
        scale = max(1.4826 * np.median(r[ok]), MIN_SCALE_NM)
        z = r / (TUKEY_C * scale)
        w = np.where(ok & (z < 1), (1 - z ** 2) ** 2, 0.0)
        done = np.allclose(new, beta, atol=1e-4)
        beta = new
        if done:
            break
    used = w > 0
    A = np.zeros((2 * n, cols))
    A[0::2, 0] = 1.0
    A[1::2, 1] = 1.0
    if with_a:
        A[0::2, 2], A[1::2, 2] = u[:, 0], u[:, 1]
    Au = A[np.repeat(used, 2)]
    try:
        cov = np.linalg.inv(Au.T @ Au) * scale ** 2
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(cols, np.nan)
    res = d - beta[0:2] - (beta[2] * u if with_a else 0.0)
    return TrackOffset(beta[0:2].copy(), float(beta[2]) if with_a else np.nan,
                       err[0:2].copy(), float(err[2]) if with_a else np.nan, used, res,
                       int((ok & mv).sum()), int((ok & ~mv).sum()), float(scale))


@dataclass
class BeadOffset:
    """The colours' offset in a measurement of still emitters (beads)."""
    offset: np.ndarray              # colour 1 - colour 0 (nm), the median over the pairs
    n_pairs: int
    n_matched: int
    time: float                     # the mean time it was measured at (the table's clock)
    per_pair: "object" = None       # `pairing.PairOffsets`


def bead_offset(tid, colour, time, x, y, max_dt: float = 0.005,
                min_matched: int = 10) -> BeadOffset:
    """The offset in a bead measurement: its traces paired by time, each
    pair's median offset, and their median (module docstring)."""
    from .pairing import pair_offsets, pair_traces
    pid = pair_traces(tid, colour, time)
    off = pair_offsets(pid, colour, time, x, y, max_dt)
    keep = off.n >= min_matched
    if not keep.any():
        raise ValueError(f"no emitter of the bead measurement was seen in both colours "
                         f"{min_matched} times or more")
    return BeadOffset(np.array([np.median(off.dx[keep]), np.median(off.dy[keep])]),
                      int(keep.sum()), int(off.n[keep].sum()),
                      float(np.average(off.time[keep], weights=off.n[keep])), off)


# --------------------------------------------------------------- MBM drift

@dataclass
class ColourDrift:
    """`mbm_colour_drift`'s answer: the colours' drift against each other."""
    beads: object                   # `beads.BeadSet` of the colour differences
    t: np.ndarray                   # the difference points: time,
    d: np.ndarray                   # colour 1 - colour 0 (n, 2) nm,
    bead: np.ndarray                # and which bead (index into ``names``)
    names: List[str]

    def __call__(self, t) -> np.ndarray:
        """The drift (n, 2) nm, zero at the beads' first point."""
        return self.beads.curve(np.asarray(t, float))[:, :2]

    @property
    def span(self) -> Tuple[float, float]:
        return float(self.t.min()), float(self.t.max())


def mbm_colour_drift(beads, info: Sequence[Dict], knot: float = 60.0,
                     max_dt: float = 2.0) -> Optional[ColourDrift]:
    """The colours' drift against each other from MBM beads recorded in
    both: ``beads`` a bead table (`io.msr.bead_table`: ``tid``, ``channel``,
    ``time_s``, ``x_nm``, ``y_nm``) and ``info`` its ``mbm_beads`` entries.
    A bead's two colours are its two entries of one slot (``gri``) and
    measurement; each colour-1 point is matched with the bead's colour-0
    point nearest in time (within ``max_dt`` s).  None where no bead was
    recorded in both colours."""
    from .beads import select_beads
    if "channel" not in beads:
        return None
    tid = np.asarray(beads["tid"], np.int64)
    t = np.asarray(beads["time_s"], float)
    x = np.asarray(beads["x_nm"], float)
    y = np.asarray(beads["y_nm"], float)
    by_slot: Dict[tuple, Dict[int, int]] = {}
    for b in info:
        if b.get("channel") in (0, 1):
            by_slot.setdefault((b.get("measurement"), b["gri"]), {})[b["channel"]] = b["tid"]
    ts, ds, ids, names = [], [], [], []
    for (meas, gri), pair in sorted(by_slot.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        if 0 not in pair or 1 not in pair:
            continue
        r0 = np.flatnonzero(tid == pair[0])
        r1 = np.flatnonzero(tid == pair[1])
        r0 = r0[np.argsort(t[r0])]
        na, nb = match(r0, r1, t, max_dt)
        if len(nb) < 2:
            continue
        ts.append(t[nb])
        ds.append(np.column_stack([x[nb] - x[na], y[nb] - y[na]]))
        ids.append(np.full(len(nb), len(names)))
        name = next((str(b.get("name") or gri) for b in info if b["tid"] == pair[0]), str(gri))
        names.append(name)
    if not names:
        return None
    t_all, d_all, bead = np.concatenate(ts), np.concatenate(ds), np.concatenate(ids)
    chosen = select_beads(t_all, d_all, bead, names, knot=knot, min_beads=1)
    return ColourDrift(chosen, t_all, d_all, bead, names)


# ------------------------------------------------------------ offset curve

@dataclass
class OffsetCurve:
    """``offset(t)``, colour 1 - colour 0 (n, 2) nm, on the target's clock:
    ``c + delta(t) - delta(t0) + slope (t - t0)`` -- a constant, the MBM
    drift's shape (zero where none), and the linear part a bead measurement
    after the tracking adds where the drift does not explain it."""
    c: np.ndarray
    drift: Optional[ColourDrift] = None
    t0: float = 0.0
    slope: np.ndarray = field(default_factory=lambda: np.zeros(2))
    note: str = ""

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        out = np.tile(self.c, (len(t), 1))
        if self.drift is not None:
            out += self.drift(t) - self.drift(np.array([self.t0]))[0]
        out += (t - self.t0)[:, None] * self.slope[None, :]
        return out


def offset_curve(before: Tuple[np.ndarray, float], after: Optional[Tuple[np.ndarray, float]] = None,
                 drift: Optional[ColourDrift] = None) -> OffsetCurve:
    """The offset from one bead measurement before the tracking, or one
    before and one after: ``(offset, time)`` each, the time on the
    target's clock.  With both, what the MBM drift explains between them is
    its, and the rest is spread linearly over the time between; that rest
    is the consistency check, in ``note``."""
    c, t0 = np.asarray(before[0], float), float(before[1])
    if after is None:
        return OffsetCurve(c, drift, t0)
    c1, t1 = np.asarray(after[0], float), float(after[1])
    if t1 <= t0:
        raise ValueError("the bead measurement 'after' was taken before the one 'before'")
    explained = (drift(np.array([t1]))[0] - drift(np.array([t0]))[0]
                 if drift is not None else np.zeros(2))
    rest = (c1 - c) - explained
    note = (f"beads after - before: ({c1[0] - c[0]:.1f}, {c1[1] - c[1]:.1f}) nm over "
            f"{t1 - t0:.0f} s; MBM explains ({explained[0]:.1f}, {explained[1]:.1f}), "
            f"rest ({rest[0]:.1f}, {rest[1]:.1f}) nm spread linearly")
    return OffsetCurve(c, drift, t0, rest / (t1 - t0), note)


@dataclass
class CoMovement:
    """How two colours' traces of one molecule move together, from windows
    of the time both are seen (`co_movement`): the share of the moving
    windows where they go the same way at the same speed, close together,
    and the medians of what was compared over the moving windows."""
    share: float = np.nan              # of the moving windows, those that match
    angle_deg: float = np.nan          # between the two directions
    speed_ratio: float = np.nan        # the faster over the slower, >= 1
    distance_nm: float = np.nan        # between the two lines at the window's centre
    windows: int = 0                   # moving windows compared


def _line(t, x, y, centre):
    """A straight line through (t, x), (t, y) by least squares: the
    velocity and the position at ``centre``."""
    dt = t - centre
    var = np.sum((dt - dt.mean()) ** 2)
    if var <= 0:
        return None
    vx = np.sum((dt - dt.mean()) * (x - x.mean())) / var
    vy = np.sum((dt - dt.mean()) * (y - y.mean())) / var
    return vx, vy, x.mean() - vx * dt.mean(), y.mean() - vy * dt.mean()


def co_movement(t, x, y, colour, window: float, min_speed: float, *,
                max_angle: float = 30.0, max_ratio: float = 1.5,
                max_distance: float = 50.0, min_travel: float = 30.0,
                min_points: int = 5) -> CoMovement:
    """Whether a molecule's two colours (``colour`` 0 and 1) move as one.

    The time both colours are seen is cut into windows of ``window`` (in the
    units of ``t``; 0 or more than the overlap: one window, the whole
    overlap), and in each a straight line x(t), y(t) is fitted to each
    colour -- for linear stepping over the whole overlap, exactly the two
    lines compared.  A window moves when the faster colour's speed is
    ``min_speed`` or more: a stuck molecule says nothing about whether its
    colours move together, so stuck windows are left out, and a molecule
    that never moves gets NaN.  A moving window matches when the directions
    are within ``max_angle`` degrees, the speeds within a factor
    ``max_ratio`` and the lines within ``max_distance`` at the window's
    centre.  Windows, not one line, so that a curved microtubule and a
    motor that stops and goes again are still compared where they move.

    A line through a few tens of milliseconds of a stepping motor is one
    plateau or one step, and its slope says nothing (a pair seen in its
    second colour for 40 ms, 2026-11, came out "five times slower, 136°
    off").  So a window is judged only when, at the faster colour's speed,
    each colour is seen over ``min_travel`` or more -- a few steps."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    colour = np.asarray(colour)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    a, b = ok & (colour == 0), ok & (colour == 1)
    if a.sum() < min_points or b.sum() < min_points:
        return CoMovement()
    start, stop = max(t[a].min(), t[b].min()), min(t[a].max(), t[b].max())
    if not stop > start:
        return CoMovement()
    n = 1 if not window > 0 else max(1, round((stop - start) / window))
    edges = np.linspace(start, stop, n + 1)
    angles, ratios, distances, matched = [], [], [], []
    for lo, hi in pairwise(edges):
        centre = 0.5 * (lo + hi)
        lines = []
        for m in (a, b):
            w = m & (t >= lo) & (t <= hi)
            lines.append((_line(t[w], x[w], y[w], centre), np.ptp(t[w]))
                         if w.sum() >= min_points else (None, 0.0))
        if lines[0][0] is None or lines[1][0] is None:
            continue
        (vx0, vy0, x0, y0), (vx1, vy1, x1, y1) = lines[0][0], lines[1][0]
        s0, s1 = np.hypot(vx0, vy0), np.hypot(vx1, vy1)
        if max(s0, s1) < min_speed:
            continue
        if max(s0, s1) * min(lines[0][1], lines[1][1]) < min_travel:
            continue
        cos = (vx0 * vx1 + vy0 * vy1) / (s0 * s1) if s0 > 0 and s1 > 0 else -1.0
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        ratio = float(max(s0, s1) / min(s0, s1)) if min(s0, s1) > 0 else np.inf
        distance = float(np.hypot(x1 - x0, y1 - y0))
        angles.append(angle)
        ratios.append(ratio)
        distances.append(distance)
        matched.append(angle <= max_angle and ratio <= max_ratio and distance <= max_distance)
    if not matched:
        return CoMovement()
    return CoMovement(float(np.mean(matched)), float(np.median(angles)),
                      float(np.median(ratios)), float(np.median(distances)), len(matched))
