"""Two colours by thread: a MINFLUX measurement of two threads -- two
sequences, interleaved, one per dye -- localizes a molecule in both when it
carries both labels, or a motor and its cargo at once.  Which trace of one
thread belongs with which of the other is not recorded: it is found here,
from when they were measured.

**Pairing** (`pair_traces`): a trace of thread 0 and one of thread 1 that
overlap in time are candidates; the pairs are taken one to one, the longest
overlap first (SMAP's Segment_cotracks and StepsMINFLUX_dc took the most
frequent partner within the reference trace's time; the MFX scripts'
`get_co_traces`, the first match).  Each pair gets one ``pair_id``, a
trace with no partner its own, so a region of ``pair_id`` is a track in one
colour or both.

**The offset between the colours** (`pair_offsets`): within a pair, each
localization of thread 1 is matched with the thread-0 localization nearest
in time (SMAP's DCoffset's `distancedc`), and the pair's offset is the
median of their differences.  Per pair, over time and over the field it is
what DCoffset_Beads_ROIs showed; pooled, a shift that puts thread 1 onto
thread 0, kept as a named transformation for *Transform*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Intervals:
    """Each trace's thread, id, file and time span, as `trace_intervals` finds them."""
    tid: np.ndarray
    thread: np.ndarray
    file: np.ndarray
    start: np.ndarray
    end: np.ndarray


def trace_intervals(tid, thi, time, file=None) -> Intervals:
    """One entry per trace (per file and thread): its first and last time."""
    tid = np.asarray(tid, np.int64)
    thi = np.asarray(thi, np.int64)
    t = np.asarray(time, np.float64)
    f = np.zeros(len(tid), np.int64) if file is None else np.asarray(file, np.int64)
    use = tid >= 0
    keys = np.column_stack([f[use], thi[use], tid[use]])
    found, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    start = np.full(len(found), np.inf)
    end = np.full(len(found), -np.inf)
    np.minimum.at(start, inverse, t[use])
    np.maximum.at(end, inverse, t[use])
    return Intervals(found[:, 2], found[:, 1], found[:, 0], start, end)


def pair_traces(tid, thi, time, file=None, min_overlap: float = 0.0) -> np.ndarray:
    """Each row's ``pair_id`` (see the module docstring): 1, 2, ... in the
    order the pairs (or lone traces) start; -1 where the row has no trace."""
    tid = np.asarray(tid, np.int64)
    thi = np.asarray(thi, np.int64)
    f = np.zeros(len(tid), np.int64) if file is None else np.asarray(file, np.int64)
    iv = trace_intervals(tid, thi, time, f)
    m = len(iv.tid)
    partner = np.full(m, -1, np.int64)
    for file_k in np.unique(iv.file):
        a = np.flatnonzero((iv.file == file_k) & (iv.thread == 0))
        b = np.flatnonzero((iv.file == file_k) & (iv.thread == 1))
        if not len(a) or not len(b):
            continue
        b = b[np.argsort(iv.start[b])]
        cands = []
        for i in a:
            hi = np.searchsorted(iv.start[b], iv.end[i], side="right")
            js = b[:hi][iv.end[b[:hi]] >= iv.start[i]]
            for j in js:
                overlap = min(iv.end[i], iv.end[j]) - max(iv.start[i], iv.start[j])
                if overlap >= min_overlap:
                    cands.append((overlap, i, j))
        for overlap, i, j in sorted(cands, key=lambda c: (-c[0], c[1], c[2])):
            if partner[i] < 0 and partner[j] < 0:
                partner[i], partner[j] = j, i
    # a pair, or a lone trace, is one group: numbered by when it starts
    group = np.where(partner >= 0, np.minimum(np.arange(m), partner), np.arange(m))
    first = np.full(m, np.inf)
    np.minimum.at(first, group, iv.start)
    roots = np.unique(group)
    rank = np.empty(m, np.int64)
    rank[roots[np.argsort(first[roots], kind="stable")]] = np.arange(1, len(roots) + 1)
    pid_of_trace = rank[group]
    keys = np.column_stack([iv.file, iv.thread, iv.tid])
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    out = np.full(len(tid), -1, np.int64)
    use = tid >= 0
    rows = np.column_stack([f[use], thi[use], tid[use]])
    at = _find_rows(keys[order], rows)
    out[use] = pid_of_trace[order][at]
    return out


def _find_rows(sorted_keys: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Where each of ``rows`` (n, 3) sits in ``sorted_keys`` (lexsorted)."""
    big = np.int64(1) << np.int64(20)
    code = lambda k: (k[:, 0] * big + k[:, 1]) * (big * big) + k[:, 2]   # noqa: E731
    return np.searchsorted(code(sorted_keys), code(rows))


@dataclass
class PairOffsets:
    """Per pair: its ``pair_id``, the median offset of thread 1 from thread 0
    (``dx``, ``dy``, nm), the localizations matched (``n``), the mean time
    and the thread-0 position (``x``, ``y``) it was measured at."""
    pair_id: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    n: np.ndarray
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def pooled(self) -> np.ndarray:
        """The median offset over the pairs (nm, dx and dy)."""
        ok = self.n > 0
        if not ok.any():
            return np.full(2, np.nan)
        return np.array([np.median(self.dx[ok]), np.median(self.dy[ok])])


def pair_offsets(pair_id, thi, time, x, y, max_dt: Optional[float] = None) -> PairOffsets:
    """The offset between the colours in every pair (see the module
    docstring).  ``max_dt``: a match further apart in time is left out."""
    pid = np.asarray(pair_id, np.int64)
    thi = np.asarray(thi, np.int64)
    t = np.asarray(time, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = (pid >= 0) & np.isfinite(x) & np.isfinite(y)
    ids, dx, dy, n, tm, px, py = [], [], [], [], [], [], []
    a_rows = np.flatnonzero(ok & (thi == 0))
    b_rows = np.flatnonzero(ok & (thi == 1))
    a_rows = a_rows[np.lexsort((t[a_rows], pid[a_rows]))]
    b_rows = b_rows[np.lexsort((t[b_rows], pid[b_rows]))]
    a_pid, b_pid = pid[a_rows], pid[b_rows]
    for p in np.intersect1d(np.unique(a_pid), np.unique(b_pid)):
        ra = a_rows[np.searchsorted(a_pid, p):np.searchsorted(a_pid, p, side="right")]
        rb = b_rows[np.searchsorted(b_pid, p):np.searchsorted(b_pid, p, side="right")]
        at = np.clip(np.searchsorted(t[ra], t[rb]), 1, max(len(ra) - 1, 1))
        before = ra[at - 1]
        after = ra[np.minimum(at, len(ra) - 1)]
        near = np.where(np.abs(t[before] - t[rb]) <= np.abs(t[after] - t[rb]), before, after)
        keep = np.ones(len(rb), bool) if max_dt is None else np.abs(t[near] - t[rb]) <= max_dt
        if not keep.any():
            continue
        ids.append(p)
        dx.append(np.median(x[rb][keep] - x[near][keep]))
        dy.append(np.median(y[rb][keep] - y[near][keep]))
        n.append(int(keep.sum()))
        tm.append(float(t[rb][keep].mean()))
        px.append(float(x[ra].mean()))
        py.append(float(y[ra].mean()))
    arr = lambda v, d=np.float64: np.asarray(v, d)   # noqa: E731
    return PairOffsets(arr(ids, np.int64), arr(dx), arr(dy), arr(n, np.int64), arr(tm),
                       arr(px), arr(py))
