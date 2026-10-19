"""Beads: which ones to trust, and the drift the trusted ones agree on.

Every bead-based correction has the same two halves.  A bead sits still on
the coverslip, so where it appears to move, the sample moved; and a bead
that moves on its own -- it came loose, it was lost and found again in the
wrong place, it is dim and noisy -- moves the correction with it.
Imspector's MBM correction is the plain mean of its beads' displacements
(NOTES.md, "MINFLUX beam-line monitoring beads"): one bead among four that
wanders by 20 nm moves every localization by 5.

**The fit** (`fit_offsets_and_spline`) is the fiducial fit's
(`smappy.fiducials`), on a continuous time axis: point ``i`` of bead ``b``
at time ``t`` is at ``o_b + d(t_i) + noise``, each bead at its own fitted
offset and the drift a cubic B-spline with a coefficient every *knot
spacing*, all in one least-squares problem.  A bead that starts late or
stops early is placed by its offset over the times it was seen, never by
its first point or its mean (which is the mean of the drift over *its*
times; Soeller's MBM code subtracts it, and shifts that bead's piece of the
curve by the difference).  `fiducials.fit_drift` is this on frames.

**Judging a bead** (`select_beads`) is against the consensus of the
others: its residual from the curve the *other* used beads give, with its
own offset.  Leaving it out matters with few beads -- with four, a bead in
the fit pulls the curve a quarter of the way to itself and hides that much
of its own error.  From that residual, per bead (`bead_metrics`):

* *noise* -- the scatter from one point to the next, robustly (1.4826 times
  the median absolute deviation of the differences, over sqrt 2): its
  precision, whatever it does slowly;
* *wander* -- the slow part of its residual, ``sqrt(rms^2 - noise^2)``: a
  motion of its own;
* *jump* -- the largest step between the medians of five points before and
  five after: a bead lost and found again elsewhere;
* *coverage* -- the fraction of the beads' time span it was seen in.

**The rule** is the caller's (`Rule`); the default, `MadRule`, calls a
bead an outlier where a metric is above the median of the used beads by more
than *k* times their spread (1.4826 MAD) *and* by more than an absolute
margin in nm -- the margin because the spread of four beads that agree is
tiny, and a bead 0.3 nm noisier than the others is not a bad bead.  Rejecting
and fitting alternate until the set no longer changes, never below
*min beads*.  A choice made by hand (*in* or *out*, by bead name) is never
overruled.  The defaults are a guess, to be set from many files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# the second-difference penalty on the spline, relative to the mean data
# weight per coefficient (`fiducials`, where it was measured): it bridges a
# stretch no bead covers and barely bends one the beads do cover
PENALTY = 1e-3

# the metrics `bead_metrics` gives, in the order a bead list shows them
METRICS = ("noise_nm", "wander_nm", "jump_nm", "coverage", "n")


@dataclass
class SplineCurve:
    """The drift as a cubic B-spline: ``coefficients`` (n_coefficients,
    n_axes) on knots spread evenly over ``[lo, hi]``, held beyond them."""
    coefficients: np.ndarray
    lo: float
    hi: float

    def __call__(self, t) -> np.ndarray:
        from .drift import _spline_basis
        t = np.asarray(t, float)
        if not len(t):
            return np.zeros((0, self.coefficients.shape[1]))
        basis = _spline_basis(t, len(self.coefficients), self.lo, self.hi)
        return np.asarray(basis @ self.coefficients)


def n_coefficients(lo: float, hi: float, knot: float) -> int:
    return max(4, int(round((hi - lo) / max(knot, 1e-12))) + 3)


def fit_offsets_and_spline(t, positions, index, n_beads: int, lo: float, hi: float,
                           n_coeffs: int, reference: Optional[float] = None,
                           weights=None, penalty: float = PENALTY
                           ) -> Tuple[np.ndarray, SplineCurve]:
    """Every bead's offset and one spline through all of them, together.

    ``t`` (n,) the times, ``positions`` (n, n_axes) nm, ``index`` (n,) the
    bead of each point (0 .. n_beads - 1), ``weights`` (n,) optional.  The
    curve is zero at ``reference`` (default ``lo``): a constant can move
    from every offset into the curve, and a Lagrange multiplier fixes it.
    A bead with no points gets offset 0.
    """
    from scipy import sparse
    from .drift import _spline_basis

    t = np.asarray(t, float)
    p = np.asarray(positions, float)
    if p.ndim == 1:
        p = p[:, None]
    index = np.asarray(index, np.int64)
    if not len(t):
        raise ValueError("no bead points to fit the drift to")
    n_axes = p.shape[1]
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    basis = sparse.csr_matrix(_spline_basis(t, n_coeffs, lo, hi))
    offsets = sparse.csr_matrix((np.ones(len(t)), (np.arange(len(t)), index)),
                                shape=(len(t), n_beads))
    design = sparse.diags(sw) @ sparse.hstack([offsets, basis]).tocsr()
    normal = (design.T @ design).toarray()
    rhs = design.T @ (p * sw[:, None])
    # a bead with no points: its offset fixed at 0 rather than left free
    empty = np.flatnonzero(np.diag(normal)[:n_beads] == 0)
    normal[empty, empty] = 1.0

    d2 = np.diff(np.eye(n_coeffs), 2, axis=0)
    weight = penalty * float(np.trace(normal[n_beads:, n_beads:])) / n_coeffs
    normal[n_beads:, n_beads:] += weight * (d2.T @ d2)

    at = np.asarray(_spline_basis([lo if reference is None else reference],
                                  n_coeffs, lo, hi).toarray())[0]
    k = n_beads + n_coeffs
    system = np.zeros((k + 1, k + 1))
    system[:k, :k] = normal
    system[k, n_beads:k] = at
    system[n_beads:k, k] = at
    right = np.vstack([np.asarray(rhs), np.zeros((1, n_axes))])
    solution = np.linalg.lstsq(system, right, rcond=None)[0]
    return solution[:n_beads], SplineCurve(solution[n_beads:k], float(lo), float(hi))


def _step(r: np.ndarray, window: int = 5) -> float:
    """The largest step between the medians of ``window`` points before and
    after any point of a residual track (n, n_axes), in nm."""
    n = len(r)
    w = min(window, n // 2)
    if w < 2:
        return 0.0
    best = 0.0
    for i in range(w, n - w + 1):
        before = np.median(r[i - w:i], axis=0)
        after = np.median(r[i:i + w], axis=0)
        best = max(best, float(np.sqrt(np.sum((after - before) ** 2))))
    return best


def bead_metrics(t, residuals, index, n_beads: int) -> Dict[str, np.ndarray]:
    """Per bead, from its residuals about the consensus: *noise*, *wander*,
    *jump* (nm), *coverage* (of the beads' time span) and *n* points."""
    t = np.asarray(t, float)
    r = np.asarray(residuals, float)
    index = np.asarray(index, np.int64)
    span = float(np.ptp(t)) if len(t) else 0.0
    out = {name: np.full(n_beads, np.nan) for name in METRICS}
    for b in range(n_beads):
        rows = np.flatnonzero(index == b)
        rows = rows[np.argsort(t[rows], kind="stable")]
        out["n"][b] = len(rows)
        if not len(rows):
            out["coverage"][b] = 0.0
            continue
        rb = r[rows]
        out["coverage"][b] = float(np.ptp(t[rows])) / span if span > 0 else 1.0
        rms2 = float(np.mean(np.sum(rb ** 2, axis=1)))
        if len(rows) >= 3:
            d = np.diff(rb, axis=0)
            mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
            noise2 = float(np.sum((1.4826 * mad / np.sqrt(2.0)) ** 2))
        else:
            noise2 = np.nan
        out["noise_nm"][b] = np.sqrt(noise2)
        out["wander_nm"][b] = np.sqrt(max(rms2 - (noise2 if np.isfinite(noise2) else 0.0),
                                          0.0))
        out["jump_nm"][b] = _step(rb)
    return out


# --------------------------------------------------------------------- rules
# A rule takes the metrics and which beads are used now, and says which beads
# are outliers and why: (bool per bead, a reason per bead, "" for none).
Rule = Callable[[Dict[str, np.ndarray], np.ndarray], Tuple[np.ndarray, List[str]]]


@dataclass
class MadRule:
    """An outlier where a metric exceeds the used beads' median by more than
    ``k`` robust standard deviations and by more than ``margin_nm``; or seen
    in less than ``min_coverage`` of the time."""
    metrics: Sequence[str] = ("noise_nm", "wander_nm", "jump_nm")
    k: float = 3.0
    margin_nm: float = 2.0
    min_coverage: float = 0.0

    def __call__(self, metrics: Dict[str, np.ndarray], used: np.ndarray
                 ) -> Tuple[np.ndarray, List[str]]:
        n = len(used)
        out = np.zeros(n, bool)
        reasons = [[] for _ in range(n)]
        for name in self.metrics:
            values = np.asarray(metrics[name], float)
            base = values[used & np.isfinite(values)]
            if len(base) < 2:
                continue
            median = float(np.median(base))
            spread = 1.4826 * float(np.median(np.abs(base - median)))
            limit = median + max(self.k * spread, self.margin_nm)
            bad = np.isfinite(values) & (values > limit)
            for b in np.flatnonzero(bad):
                reasons[b].append(f"{name[:-3] if name.endswith('_nm') else name} "
                                  f"{values[b]:.1f} > {limit:.1f} nm")
            out |= bad
        if self.min_coverage > 0:
            low = np.asarray(metrics["coverage"], float) < self.min_coverage
            for b in np.flatnonzero(low):
                reasons[b].append(f"seen in {metrics['coverage'][b]:.0%} of the time")
            out |= low
        return out, ["; ".join(r) for r in reasons]


# -------------------------------------------------------------------- choose
@dataclass
class BeadSet:
    """The beads, the ones used, why, and the drift the used ones give.

    ``ids`` and ``names`` per bead; ``used`` what the drift is fitted from;
    ``outlier`` what the rule rejected and ``reasons`` its words, the
    ``overrides`` by name (True in, False out) never overruled; ``metrics``
    per bead (`METRICS`), each against the consensus of the *other* used
    beads.  ``t``, ``residuals`` and ``index`` are every point's time, its
    residual from the final curve and offset, and its bead, for figures.
    """
    ids: np.ndarray
    names: List[str]
    used: np.ndarray
    outlier: np.ndarray
    reasons: List[str]
    overrides: Dict[str, bool]
    metrics: Dict[str, np.ndarray]
    curve: SplineCurve
    offsets_nm: np.ndarray
    t: np.ndarray
    residuals: np.ndarray
    index: np.ndarray
    rounds: int = 1

    def __len__(self) -> int:
        return len(self.ids)

    def state(self, b: int) -> str:
        """What a bead list says of bead ``b``."""
        name = self.names[b]
        if name in self.overrides:
            return "in, by hand" if self.overrides[name] else "out, by hand"
        if self.outlier[b]:
            return f"out: {self.reasons[b]}"
        return "used" if self.used[b] else "not used"

    def drift_nm(self, times_s) -> np.ndarray:
        """The curve at ``times_s``, as (n, 3) nm (z 0 where not fitted)."""
        curve = self.curve(times_s)
        out = np.zeros((len(curve), 3))
        out[:, :curve.shape[1]] = curve
        return out

    def drift(self, step_s: Optional[float] = None, **kwargs):
        """The curve as a `drift.Drift` on a time axis, sampled every
        ``step_s`` (default an eighth of the knot spacing, at most 0.5 s) over
        the beads' span; linear between samples is then far below the
        precision of any bead."""
        from .drift import Drift
        lo, hi = self.curve.lo, self.curve.hi
        knot = (hi - lo) / max(len(self.curve.coefficients) - 3, 1)
        step = step_s or min(knot / 8.0, 0.5)
        times = np.arange(lo, hi + 0.5 * step, step) if hi > lo else np.array([lo])
        kwargs.setdefault("n_used", int(self.used.sum()))
        return Drift(self.drift_nm(times), times_s=times, **kwargs)


def _offset(p: np.ndarray, curve: np.ndarray) -> np.ndarray:
    return np.mean(p - curve, axis=0) if len(p) else np.zeros(p.shape[1])


def select_beads(t, positions, bead, names: Optional[Sequence[str]] = None, *,
                 knot: float = 10.0, rule: Optional[Rule] = None,
                 overrides: Optional[Dict[str, bool]] = None, min_beads: int = 2,
                 max_rounds: int = 10, candidates=None) -> BeadSet:
    """Choose the beads and fit the drift they agree on.

    ``t`` (n,) times (s, or frames), ``positions`` (n, n_axes) nm, ``bead``
    (n,) an id per point (ids below 1 are no bead).  ``names`` per distinct
    id in sorted order (default the ids); ``knot`` the spline's spacing in
    the units of ``t``; ``rule`` default `MadRule()`; ``overrides`` by name,
    True to use a bead whatever the rule says, False never to; ``candidates``
    (bool per bead) the beads the rule may use at all, default every one.
    """
    t = np.asarray(t, float)
    p = np.asarray(positions, float)
    if p.ndim == 1:
        p = p[:, None]
    bead = np.asarray(bead, np.int64)
    keep = bead > 0
    t, p, bead = t[keep], p[keep], bead[keep]
    if not len(t):
        raise ValueError("no beads: no point has a bead id above 0")
    ids, index = np.unique(bead, return_inverse=True)
    n = len(ids)
    names = [str(i) for i in ids] if names is None else [str(x) for x in names]
    if len(names) != n:
        raise ValueError(f"{len(names)} names for {n} beads")
    rule = rule or MadRule()
    overrides = {k: bool(v) for k, v in (overrides or {}).items() if k in names}
    forced_in = np.array([overrides.get(x) is True for x in names])
    forced_out = np.array([overrides.get(x) is False for x in names])
    allowed = np.ones(n, bool) if candidates is None else np.asarray(candidates, bool)
    lo, hi = float(t.min()), float(t.max())
    k = n_coefficients(lo, hi, knot)

    def fit(use):
        rows = use[index]
        if not rows.any():
            raise ValueError("no bead is used: tick at least one in")
        offsets, curve = fit_offsets_and_spline(t[rows], p[rows], index[rows], n,
                                                lo, hi, k)
        return offsets, curve

    def judge(use, offsets, curve):
        """Residuals of every bead from the consensus of the other used beads."""
        r = np.zeros_like(p)
        full = curve(t)
        for b in range(n):
            rows = index == b
            if use[b] and use.sum() > 1:
                _, other = fit(use & (np.arange(n) != b))
                c = other(t[rows])
            else:
                c = full[rows]
            r[rows] = p[rows] - c - _offset(p[rows], c)
        return bead_metrics(t, r, index, n)

    used = (allowed | forced_in) & ~forced_out
    outlier = np.zeros(n, bool)
    reasons = [""] * n
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        offsets, curve = fit(used)
        metrics = judge(used, offsets, curve)
        outlier, reasons = rule(metrics, used & ~forced_in)
        outlier = outlier & ~forced_in & ~forced_out
        new = (allowed | forced_in) & ~forced_out & ~outlier
        if new.sum() < min_beads:
            # too few left: the least deviant of the rest come back
            spare = np.flatnonzero((allowed & ~forced_out) & ~new)
            score = np.nan_to_num(metrics["wander_nm"][spare] + metrics["jump_nm"][spare],
                                  nan=np.inf)
            back = spare[np.argsort(score)][:max(min_beads - int(new.sum()), 0)]
            new[back] = True
            outlier[back] = False
        if np.array_equal(new, used):
            break
        used = new
    offsets, curve = fit(used)
    full = curve(t)
    # an unused bead gets its own offset about the curve, for its figure
    for b in np.flatnonzero(~used):
        rows = index == b
        offsets[b] = _offset(p[rows], full[rows])
    residuals = p - offsets[index] - full
    return BeadSet(ids, names, used, outlier, list(reasons), overrides, metrics, curve,
                   offsets, t, residuals, index, rounds)


# -------------------------------------------------------------- for a window
@dataclass
class BeadView:
    """What a bead list shows, from whichever plugin chose the beads.

    The bead window (`gui.bead_window`) knows nothing of MBM or fiducials: a
    plugin hands it this (`bead_view`), and takes back the choices made by
    hand (`with_overrides`).  ``sources`` are what the beads can come from
    -- a measurement each -- and ``source`` the one shown; ``axes`` names
    the columns of the positions (x, y, z); ``columns`` are more per-bead
    columns for the table (Imspector's own use of a bead); ``where_nm`` (n,
    2) where each bead is, for the view to be centred on it; ``prefix`` is
    what this source's bead names carry in the plugin's settings.
    """
    beads: BeadSet
    axes: Tuple[str, ...]
    sources: List[str]
    source: int = 0
    columns: Dict[str, List] = field(default_factory=dict)
    where_nm: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None       # the names as the list shows them
    title: str = ""


def parse_overrides(text: str) -> Dict[str, bool]:
    """``"R0 out, 2:R3 in"`` as ``{"R0": False, "2:R3": True}``."""
    out: Dict[str, bool] = {}
    for part in (text or "").split(","):
        words = part.strip().rsplit(None, 1)
        if len(words) == 2 and words[1].lower() in ("in", "out"):
            out[words[0]] = words[1].lower() == "in"
        elif part.strip():
            raise ValueError(f"{part.strip()!r}: a bead by hand is 'name in' or "
                             "'name out', separated by commas")
    return out


def format_overrides(overrides: Dict[str, bool]) -> str:
    return ", ".join(f"{name} {'in' if v else 'out'}" for name, v in overrides.items())


def draw_bead_tracks(figure, beads: BeadSet, axes: Sequence[str] = ("x", "y", "z"),
                     selected: Optional[int] = None, labels: Optional[Sequence[str]] = None,
                     time_unit: str = "s"):
    """Every bead's track lined up on the consensus (left), and its
    deviation from it (right), a row per axis.  Used beads grey, beads left
    out red and dashed, ``selected`` drawn over them in colour.  Each bead's
    lines carry its index as their gid and are pickable, for a window to
    select a bead by clicking it.  Returns the axes, (rows, 2)."""
    n_axes = beads.residuals.shape[1]
    grid = np.atleast_2d(figure.subplots(n_axes, 2, sharex=True, squeeze=False))
    order = np.argsort(beads.t, kind="stable")
    t_all = beads.t[order]
    curve = beads.curve(t_all)
    labels = list(labels) if labels is not None else beads.names
    # the selected bead last, so that it is drawn over the others
    for b in [b for b in range(len(beads)) if b != selected] + \
            ([selected] if selected is not None else []):
        rows = order[beads.index[order] == b]
        if not len(rows):
            continue
        t = beads.t[rows]
        c = beads.curve(t)
        if b == selected:
            style = dict(color="C1", lw=1.8, zorder=5, label=labels[b])
        elif beads.used[b]:
            style = dict(color="0.55", lw=0.8, zorder=2)
        else:
            style = dict(color="C3", lw=0.8, ls="--", alpha=0.7, zorder=3)
        for a in range(n_axes):
            for col, values in ((0, c[:, a] + beads.residuals[rows, a]),
                                (1, beads.residuals[rows, a])):
                line, = grid[a, col].plot(t, values, picker=5, **style)
                line.set_gid(str(b))
    for a in range(n_axes):
        grid[a, 0].plot(t_all, curve[:, a], color="k", lw=1.2, zorder=4)
        grid[a, 1].axhline(0.0, color="k", lw=0.8, zorder=4)
        name = axes[a] if a < len(axes) else str(a)
        grid[a, 0].set_ylabel(f"{name} (nm)")
        grid[a, 1].set_ylabel(f"{name} - drift (nm)")
    grid[0, 0].set_title("beads on the drift (black)", fontsize="medium")
    grid[0, 1].set_title("each bead's deviation", fontsize="medium")
    for col in range(2):
        grid[-1, col].set_xlabel(f"time ({time_unit})")
    if selected is not None:
        grid[0, 0].legend(loc="best", fontsize="small")
    return grid
