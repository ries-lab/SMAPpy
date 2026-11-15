"""Steps in a one-dimensional trace: a motor's position along its track.

**AutoStepFinder** (Loeff, Kerssemakers et al. 2021), re-implemented from
the method -- its code carries no licence -- as SMAP's StepsMINFLUX uses
it (`shared/externaltools/AutoStepFinder_Ries`):

1. *Greedy fit* (Kerssemakers et al. 2006): start from one plateau, the
   mean; split, again and again, the plateau whose best split removes the
   most squared error.  A plateau of fewer than four points is not split.
2. *Counter-fit*: the plateaus' best splits not taken, as a step function
   of their own.  Its squared error over the fit's is the **S-score**: it
   peaks where the fit has found the real steps -- before, the counter-fit
   holds steps too; after, the fit only splits noise.
3. The S-curve minus one, a straight line from 0 to its end subtracted;
   its peak is the number of steps.  A round counts only where that peak
   is above the *threshold* (0.15).
4. A second round on the residual finds steps the first passed over; the
   final steps are the first round's and the second's new ones,
   ``overshoot`` times as many (SMAP's *coarseness*).

Where SMAP ranked a plateau by (avr - avl)^2 x w, this ranks it by the
squared error the split removes, (avr - avl)^2 x nl x nr / w, Kerssemakers'
criterion: a split at a plateau's very end no longer outranks one in its
middle.  The localstepmerge of AutoStepFinder and its bootstrap errors are
not done.

**Split and merge** (SMAP's `splitmergefit`), with an expected step size
s: two steps in a row whose sum is under 1.4 s and whose first is under
0.7 s become one; a step over 1.4 s gets a plateau inserted halfway; then
the boundaries are refined (`refine`); ten rounds.  **Refine** (SMAP's
`fitstepind`): each boundary moves by up to two points to where the points
lie closer to their own level, the levels recomputed, until nothing moves
(ten times at most).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

#: a plateau shorter than this is not split (AutoStepFinder's window)
MIN_SPLIT = 4


class _Sums:
    """Squared errors of segments from cumulative sums."""

    def __init__(self, x: np.ndarray):
        self.c1 = np.r_[0.0, np.cumsum(x)]
        self.c2 = np.r_[0.0, np.cumsum(x * x)]

    def ss(self, a, e):
        a, e = np.asarray(a), np.asarray(e)
        n = np.maximum(e - a, 1)
        s1 = self.c1[e] - self.c1[a]
        return self.c2[e] - self.c2[a] - s1 * s1 / n

    def mean(self, a, e):
        return (self.c1[e] - self.c1[a]) / np.maximum(e - a, 1)


def _best_split(s: _Sums, a: int, e: int) -> Tuple[int, float]:
    """The best split of [a, e): ``(k, gain)``, k the first point of the right
    part; gain the squared error removed (0 where it cannot be split)."""
    w = e - a
    if w < 2:
        return a, 0.0
    lo, hi = (a + 1, e - 1) if w < MIN_SPLIT else (a + 2, e - 2)
    ks = np.arange(lo, hi + 1)
    if not len(ks):
        return a + w // 2, 0.0
    gain = s.ss(a, e) - s.ss(a, ks) - s.ss(ks, e)
    j = int(np.argmax(gain))
    return int(ks[j]), float(gain[j])


def greedy(x: np.ndarray, n_max: int) -> Tuple[List[int], np.ndarray]:
    """Kerssemakers' greedy fit up to ``n_max`` steps: the boundaries in the
    order they were added, and the S-score for 0 .. n_max steps."""
    x = np.asarray(x, np.float64)
    n = len(x)
    s = _Sums(x)
    plateaus = [(0, n) + _best_split(s, 0, n)]          # (start, end, cand, gain)
    order: List[int] = []
    S = np.ones(n_max + 1)
    for c in range(n_max + 1):
        starts = np.array([p[0] for p in plateaus])
        ends = np.array([p[1] for p in plateaus])
        fit = float(s.ss(starts, ends).sum())
        cands = np.sort(np.array([p[2] for p in plateaus if p[1] - p[0] >= 2]))
        bounds = np.r_[0, cands, n]
        counter = float(s.ss(bounds[:-1], bounds[1:]).sum())
        S[c] = counter / fit if fit > 0 else 1.0
        if c == n_max:
            break
        ok = [i for i, p in enumerate(plateaus) if p[1] - p[0] >= MIN_SPLIT and p[3] > 0]
        if not ok:
            S = S[:c + 1]
            break
        i = max(ok, key=lambda j: plateaus[j][3])
        a, e, k, _ = plateaus[i]
        order.append(k)
        plateaus[i:i + 1] = [(a, k) + _best_split(s, a, k), (k, e) + _best_split(s, k, e)]
    return order, S


def s_peak(S: np.ndarray) -> Tuple[int, np.ndarray]:
    """The number of steps where the S-curve, minus one and minus the line
    from 0 to its end, peaks; and that curve."""
    s2 = np.maximum(S, 1.0) - 1.0
    s3 = s2 - np.linspace(0.0, s2[-1], len(s2))
    return int(np.argmax(s3)), s3


def fit_from(x: np.ndarray, boundaries, stat: str = "mean") -> np.ndarray:
    """The step function through ``x`` with plateaus starting at
    ``boundaries`` (sorted, after the first), each its points' mean or median."""
    x = np.asarray(x, np.float64)
    b = np.r_[0, np.asarray(boundaries, np.int64), len(x)]
    out = np.empty(len(x))
    for a, e in zip(b[:-1], b[1:]):
        if e > a:
            out[a:e] = np.median(x[a:e]) if stat == "median" else x[a:e].mean()
    return out


@dataclass
class Found:
    """What `autostepfinder` found: the boundaries (sorted plateau starts),
    the S-curves of the two rounds, the steps each round accepted."""
    boundaries: np.ndarray
    s_curves: List[np.ndarray]
    accepted: List[int]


def autostepfinder(x, threshold: float = 0.15, overshoot: float = 1.0,
                   max_fraction: float = 0.25, max_steps: int = 10000,
                   stat: str = "mean") -> Found:
    """AutoStepFinder's two rounds (see the module docstring)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    n_max = int(min(np.ceil(n * max_fraction), max_steps))
    residual = x.copy()
    log: List[Tuple[int, int]] = []                     # (boundary, round or 3)
    curves, accepted = [], []
    for round_no in (1, 2):
        if n_max < 1:
            break
        order, S = greedy(residual, n_max)
        best, s3 = s_peak(S)
        curves.append(s3)
        ok = s3.max(initial=0) > threshold
        accepted.append(best if ok else 0)
        marks = [(b, round_no if (ok and j < best) else 3) for j, b in enumerate(order)]
        if round_no == 1:
            log = marks
        else:
            found = {b for b, r in log if r < 3}
            log = [(b, r) for b, r in log if r < 3] + [(b, r) for b, r in marks
                                                       if b not in found]
        residual = residual - fit_from(residual, sorted(order[:best]), stat)
    n_accept = sum(1 for _, r in log if r < 3)
    pick = int(round(min(overshoot, 1.0) * n_accept)) if overshoot <= 1 else \
        int(round(overshoot * n_accept))
    chosen = np.unique(np.array([b for b, _ in log[:pick]], np.int64))
    return Found(chosen, curves, accepted)


def refine(x, boundaries, stat: str = "mean", iterations: int = 10) -> np.ndarray:
    """Each boundary moved by up to two points to where the points around it
    lie closer to their own level, the levels recomputed, until nothing moves."""
    x = np.asarray(x, np.float64)
    b = np.array(sorted(boundaries), np.int64)
    for _ in range(iterations):
        edges = np.r_[0, b, len(x)]
        lev = np.array([(np.median(x[a:e]) if stat == "median" else x[a:e].mean())
                        if e > a else np.nan for a, e in zip(edges[:-1], edges[1:])])
        moved = b.copy()
        for k in range(len(b)):
            left, right = lev[k], lev[k + 1]
            lo = max(edges[k] + 1, b[k] - 2)
            hi = min(edges[k + 2] - 1, b[k] + 2)
            best, cost_best = b[k], np.inf
            for cand in range(lo, hi + 1):
                seg = x[max(lo - 1, 0):hi + 1]
                pos = np.arange(max(lo - 1, 0), hi + 1)
                cost = np.sum(np.where(pos < cand, (seg - left) ** 2, (seg - right) ** 2))
                if cost < cost_best - 1e-12:
                    best, cost_best = cand, cost
            moved[k] = best
        moved = np.unique(moved)
        if np.array_equal(moved, b):
            break
        b = moved
    return b


def split_merge(x, boundaries, step: float, stat: str = "mean", rounds: int = 10
                ) -> np.ndarray:
    """SMAP's split and merge to an expected step size ``step`` (see the
    module docstring)."""
    x = np.asarray(x, np.float64)
    b = np.array(sorted(boundaries), np.int64)
    for _ in range(rounds):
        edges = np.r_[0, b, len(x)]
        lev = np.array([(np.median(x[a:e]) if stat == "median" else x[a:e].mean())
                        for a, e in zip(edges[:-1], edges[1:])])
        d = np.diff(lev)
        # merge: two steps in a row that add to less than 1.4 s, the first under 0.7 s
        drop = set()
        k = 0
        while k < len(d) - 1:
            if abs(d[k] + d[k + 1]) < 1.4 * step and abs(d[k]) < 0.7 * step:
                drop.add(k)
                k += 2
            else:
                k += 1
        new = []
        for k, v in enumerate(b):
            if k in drop:
                continue
            if k - 1 in drop:                         # the merged plateau's step: halfway
                v = (b[k - 1] + v) // 2
            new.append(v)
        b = np.array(sorted(set(new)), np.int64)
        # split: a step over 1.4 s gets a plateau halfway
        edges = np.r_[0, b, len(x)]
        lev = np.array([(np.median(x[a:e]) if stat == "median" else x[a:e].mean())
                        for a, e in zip(edges[:-1], edges[1:])])
        big = np.flatnonzero(np.abs(np.diff(lev)) > 1.4 * step)
        extra = []
        for k in big:
            v = b[k]
            if v - 1 > edges[k] and v + 1 < edges[k + 2]:
                extra += [v - 1, v + 1]
            else:
                extra.append(v)
        if len(big):
            b = np.array(sorted(set(b.tolist()) - set(b[big].tolist()) | set(extra)),
                         np.int64)
        b = refine(x, b, stat)
    return b


def step_table(x, t, boundaries, stat: str = "mean") -> Dict[str, np.ndarray]:
    """The plateaus and steps: per plateau ``level``, ``start_time``,
    ``dwell`` (from its first point to the next plateau's), ``n``, ``std``
    and the detrended std (std(diff)/sqrt(2)); per step (between plateau k
    and k+1) ``size`` and ``time``."""
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    edges = np.r_[0, np.asarray(boundaries, np.int64), len(x)]
    a, e = edges[:-1], edges[1:]
    keep = e > a
    a, e = a[keep], e[keep]
    lev = np.array([np.median(x[i:j]) if stat == "median" else x[i:j].mean()
                    for i, j in zip(a, e)])
    start = t[a]
    nxt = np.r_[t[a[1:]], t[-1]] if len(a) else np.zeros(0)
    std = np.array([x[i:j].std(ddof=1) if j - i > 1 else np.nan for i, j in zip(a, e)])
    dstd = np.array([np.diff(x[i:j]).std(ddof=1) / np.sqrt(2) if j - i > 2 else np.nan
                     for i, j in zip(a, e)])
    return {"level": lev, "start_time": start, "dwell": nxt - start, "n": e - a,
            "std": std, "std_detrended": dstd,
            "size": np.diff(lev), "time": start[1:]}


def erlang_shape(dwells) -> float:
    """The number of hidden sub-steps a dwell-time distribution suggests:
    mean^2 / variance, the shape of a gamma (Erlang) distribution by the
    method of moments -- 1 for one rate-limiting step, n for n equal ones."""
    d = np.asarray(dwells, np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 3:
        return float("nan")
    return float(d.mean() ** 2 / d.var(ddof=1))
