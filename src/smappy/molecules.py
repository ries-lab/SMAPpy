"""From clusters to molecules: RESI's centres and G5M's mixtures.

What *Molecular mapping* (`plugins/molecular_mapping.py`) computes, as plain
functions on arrays that a script and a test call alike.  Both are ported
from Picasso (jungmannlab/picasso 0.11.3, MIT; `docs/dna-paint.md`).

**Centres** (`centres`; Picasso's `clusterer.find_cluster_centers`, the
step RESI is made of -- Reinhardt et al. 2023): each cluster is one
molecule at the mean of its localizations, with ``std / sqrt(n)`` as the
precision of that mean.  Right when a cluster holds exactly one binding site,
as *Find clusters*' *local maxima* aims for.

**G5M** (`g5m`; Picasso's `g5m.py`, Kowalewski et al. 2026): a Gaussian
mixture fitted in each cluster, the number of molecules chosen by the
Bayesian information criterion (BIC).  Three things make a mixture a model of
molecules rather than of blobs, and all three are Picasso's:

* **the width is held to the localization precision**: in each M-step a
  component's standard deviation is clipped to 0.8 - 1.5 times the
  responsibility-weighted mean precision of the localizations it explains
  (`SIGMA_BOUNDS`).  A component cannot grow to cover two sites, nor shrink
  to fit one localization's scatter;
* **the Sparrow test**: a model is refused when, along the line between any
  two of its components, their summed density has no dip -- two molecules
  closer than that cannot be told apart and are one (`resolvable`);
* **at least ``min_locs`` localizations per molecule** (`MIN_LOCS`): a
  component with fewer is background and is not kept.

The search adds components one at a time from one, and stops after
`MAX_ROUNDS_WITHOUT_BEST` rounds without a better BIC, or at ``n / min_locs``
or `MAX_COMPONENTS`.  Each number of components is fitted from ``max(k, 3)``
k-means++ starts (seeded, so a run is reproducible), at most 100 EM
iterations each, stopping when the mean log-likelihood changes by less than
1e-3; the best start that passes the Sparrow test is kept.

Per molecule (`molecule_values`): its position and the standard error of
it (``sigma / sqrt(n)``, Picasso's approximate SEM -- its bootstrap is not
ported), the fitted width absolute and relative to the precision, its
localizations, its **binding events** (the cluster's localizations split at
gaps of more than `EVENT_GAP` frames, each event's centre given to the
nearest molecule), the weighted mean and spread of its frames, and a
**p-value** that its localizations' mean log-likelihood is what a Gaussian
of its fitted width gives: low for a molecule that is two, or a cloud that
is no molecule.  Picasso's default filter (`keep`) drops molecules whose
frames spread over less than a tenth of the acquisition (sticky), with
``p <= 0.015``, or with 3 binding events or fewer.

Where this departs from Picasso:

* **numpy, not numba**: the EM runs all of a component count's starts at
  once, as one array with the starts along its first axis, so the loop in
  Python is over iterations only;
* **3D** has one width for x and y and another for z, each held to its own
  precision (`xy_err_nm`, `z_err_nm`).  Picasso's 3D model takes the shape
  of an astigmatic spot from its z calibration, which SMAPpy's spline and
  pupil fits have no use for: their precisions already say how wide the
  cloud is along each axis;
* the Sparrow test counts a flat-bottomed dip: two equal components put
  their minimum between two equal samples, which Picasso's strict test
  misses;
* the localization precision of a localization is `xy_err_nm` (Picasso:
  the mean of ``lpx`` and ``lpy``), and positions are in nanometres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

#: Picasso's G5M constants (`g5m.py`)
MIN_LOCS = 10
SIGMA_BOUNDS = (0.8, 1.5)
MAX_ROUNDS_WITHOUT_BEST = 3
MAX_COMPONENTS = 100
MAX_ITERATIONS = 100
TOLERANCE = 1e-3
REG_COVAR = 1e-6
SEED = 42
#: samples along the line between two components in the Sparrow test
SPARROW_SAMPLES = 40
#: frames without a localization inside one binding event
EVENT_GAP = 3
#: Picasso's post-filter: a molecule's frames spread over more than this
#: fraction of the acquisition, its p-value above, its events more than
MIN_STD_FRAME_FRACTION = 0.1
MIN_P_VALUE = 0.015
MIN_EVENTS = 3


# ------------------------------------------------------------------ centres

def events_in(frame: np.ndarray, gap: int = EVENT_GAP) -> np.ndarray:
    """Which binding event each localization of one group belongs to (0,
    1, ...), its frames sorted or not: a new event after more than ``gap``
    frames without a localization -- Picasso's split in `g5m` and
    `clusterer`."""
    frame = np.asarray(frame, np.int64)
    order = np.argsort(frame, kind="stable")
    new = np.r_[False, np.diff(frame[order]) > gap]
    out = np.empty(len(frame), np.int64)
    out[order] = np.cumsum(new)
    return out


def centres(points: np.ndarray, frame: np.ndarray, labels: np.ndarray,
            gap: int = EVENT_GAP) -> Dict[str, np.ndarray]:
    """One molecule per cluster (``labels`` 0 and up; -1 none) at the mean of its
    localizations: RESI's centres (Picasso's `find_cluster_centers`).

    ``points`` (n, 2) or (n, 3) nm.  Each value an array over the clusters
    in increasing label: ``label``, ``centre`` (k, d), ``std`` (k, d, ddof 1
    as pandas', Picasso's), ``sem`` (k, d) = std / sqrt(n), ``n_locs``,
    ``n_events`` (gaps of more than ``gap`` frames), ``frame`` (mean) and
    ``std_frame`` (ddof 1).  All clusters at once, by sorting, not a loop.
    """
    points = np.asarray(points, float)
    frame = np.asarray(frame, np.int64)
    labels = np.asarray(labels, np.int64)
    use = np.flatnonzero(labels >= 0)
    order = use[np.lexsort((frame[use], labels[use]))]
    lab = labels[order]
    ids, starts, n = np.unique(lab, return_index=True, return_counts=True)
    p = points[order]
    sums = np.add.reduceat(p, starts, axis=0) if len(order) else np.zeros((0, p.shape[1]))
    centre = sums / n[:, None] if len(order) else sums
    dev = p - np.repeat(centre, n, axis=0)
    ss = np.add.reduceat(dev ** 2, starts, axis=0) if len(order) else sums
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(ss / (n - 1)[:, None])
        sem = std / np.sqrt(n)[:, None]
    f = frame[order].astype(float)
    fmean = np.add.reduceat(f, starts) / n if len(order) else f
    fdev = f - np.repeat(fmean, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        fstd = np.sqrt((np.add.reduceat(fdev ** 2, starts) if len(order) else f) / (n - 1))
    # events: sorted by label then frame, a new one at a label change or a gap
    if len(order):
        step = np.r_[True, (np.diff(lab) != 0) | (np.diff(frame[order]) > gap)]
        n_events = np.add.reduceat(step.astype(np.int64), starts)
    else:
        n_events = np.zeros(0, np.int64)
    return {"label": ids, "centre": centre, "std": std, "sem": sem,
            "n_locs": n, "n_events": n_events, "frame": fmean, "std_frame": fstd}


# ---------------------------------------------------------------------- G5M

@dataclass
class Mixture:
    """A fitted G5M: the valid components only (at least ``min_locs``).

    ``var`` is (k, d): the variance along each axis, x and y always equal
    (one width in the plane), z its own in 3D.  ``weights`` are the valid
    components' weights as fitted, not normalised again over them --
    Picasso's, whose BIC and p-value use them so."""
    weights: np.ndarray       # (k,)
    means: np.ndarray         # (k, d)
    var: np.ndarray           # (k, d)
    n_locs: np.ndarray        # (k,) round(weight * n)
    log_likelihood: float     # mean over the localizations, all components

    def __len__(self) -> int:
        return len(self.weights)

    def weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        """(n, k): log(w_k N(x; mu_k, var_k))."""
        return _log_gauss(X[None], self.means[None], self.var[None])[0] \
            + np.log(self.weights)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Each localization's log-likelihood under the mixture."""
        return _logsumexp(self.weighted_log_prob(X), axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.weighted_log_prob(X).argmax(axis=1)

    def n_parameters(self) -> int:
        """Means, widths (one in 2D, two in 3D) and weights."""
        k, d = self.means.shape
        return int(k * d + k * (1 if d == 2 else 2) + k - 1)

    def bic(self, X: np.ndarray) -> float:
        return float(self.n_parameters() * np.log(len(X))
                     - 2 * self.score(X).sum())


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.log(np.sum(np.exp(a - top), axis=axis)) + np.squeeze(top, axis)


def _log_gauss(X: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(s, n, k) log densities of axis-aligned Gaussians: X (s or 1, n, d),
    means and var (s, k, d).  The squared distance expanded into matrix
    products, ``x^2/v - 2 x m/v + m^2/v``, so no (s, n, k, d) array is
    made: three times faster on a cluster of a few hundred."""
    inv = 1.0 / var                                                    # (s, k, d)
    d2 = (np.matmul(X ** 2, inv.transpose(0, 2, 1))
          - 2.0 * np.matmul(X, (means * inv).transpose(0, 2, 1))
          + (means ** 2 * inv).sum(-1)[:, None, :])
    return -0.5 * (np.maximum(d2, 0.0) + np.log(2 * np.pi * var).sum(-1)[:, None, :])


def kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``k`` starting points by k-means++ with ``2 + log k``
    local trials -- scikit-learn's `_kmeans_plusplus`, which Picasso's G5M
    starts from."""
    n = len(X)
    trials = 2 + int(np.log(k))
    chosen = np.empty(k, np.int64)
    chosen[0] = rng.integers(n)
    closest = ((X - X[chosen[0]]) ** 2).sum(1)
    for c in range(1, k):
        pot = closest.sum()
        candidates = np.searchsorted(np.cumsum(closest), rng.random(trials) * pot)
        candidates = np.minimum(candidates, n - 1)
        dist = np.minimum(closest, ((X[None] - X[candidates][:, None]) ** 2).sum(-1))
        best = int(np.argmin(dist.sum(1)))
        chosen[c] = candidates[best]
        closest = dist[best]
    return chosen


def _shape(var_axes: np.ndarray, d: int) -> np.ndarray:
    """Per-axis variances (.., k, d) made one width in x and y."""
    out = var_axes.copy()
    xy = var_axes[..., :2].mean(-1)
    out[..., 0] = xy
    out[..., 1] = xy
    return out


def _m_step(X: np.ndarray, resp: np.ndarray, lp: np.ndarray,
            bounds: Tuple[float, float]):
    """Weights, means and variances from responsibilities (s, n, k), each
    component's widths clipped to ``bounds`` times its responsibility-
    weighted mean precision per axis (Picasso's `_m_step_2D`, *local*).
    ``X`` is centred (`g5m`), so the variance as ``E[x^2] - m^2`` loses
    nothing to cancellation."""
    rt = resp.transpose(0, 2, 1)                                       # (s, k, n)
    nk = resp.sum(1) + 10 * np.finfo(float).eps                       # (s, k)
    means = np.matmul(rt, X) / nk[..., None]
    var = np.matmul(rt, X ** 2) / nk[..., None] - means ** 2
    var = _shape(np.maximum(var, 0.0) + REG_COVAR, X.shape[1])
    mean_lp = np.matmul(rt, lp) / nk[..., None]                       # precision
    lo, hi = bounds
    var = np.clip(var, (lo * mean_lp) ** 2, (hi * mean_lp) ** 2)
    weights = nk / nk.sum(-1, keepdims=True)
    return weights, means, var


def resolvable(means: np.ndarray, weights: np.ndarray, var: np.ndarray) -> bool:
    """Picasso's Sparrow test (`_check_G5M_resolution_2D`): every pair of
    components has a dip in their summed density along the line between
    them, sampled at `SPARROW_SAMPLES` points.  No components fails; one
    passes.  All pairs at once."""
    k = len(means)
    if k == 0:
        return False
    if k == 1:
        return True
    i, j = np.triu_indices(k, 1)
    t = np.linspace(0.0, 1.0, SPARROW_SAMPLES)[None, :, None]
    line = means[i][:, None] + (means[j] - means[i])[:, None] * t     # (p, s, d)

    def density(c):
        d2 = ((line - means[c][:, None]) ** 2 / var[c][:, None]).sum(-1)
        return np.exp(-0.5 * d2 - 0.5 * np.log(2 * np.pi * var[c]).sum(-1)[:, None]
                      + np.log(weights[c])[:, None])

    pdf = density(i) + density(j)
    # a dip: lower than the sample before and not higher than the one after.
    # Picasso asks for lower than both, which two equal components miss: with
    # an even number of samples their minimum falls between two equal ones
    dip = (pdf[:, 1:-1] < pdf[:, :-2]) & (pdf[:, 1:-1] <= pdf[:, 2:])
    return bool(dip.any(1).all())


def fit_mixture(X: np.ndarray, lp: np.ndarray, k: int, min_locs: int = MIN_LOCS,
                bounds: Tuple[float, float] = SIGMA_BOUNDS, seed: int = SEED
                ) -> Optional[Mixture]:
    """``k`` components from ``max(k, 3)`` k-means++ starts, all fitted at
    once; the start with the highest final log-likelihood among those whose
    valid components pass `resolvable` (Picasso's `_fit_G5M`), or None.

    ``X`` (n, d) nm and ``lp`` (n, d) each localization's precision along
    each axis."""
    n, d = X.shape
    starts = max(k, 3)
    resp = np.zeros((starts, n, k))
    for s in range(starts):
        idx = kmeans_pp(X, k, np.random.default_rng(seed + s))
        resp[s, idx, np.arange(k)] = 1.0
    # the starts' first M-step, unclipped as Picasso's initialisation
    nk = resp.sum(1) + 10 * np.finfo(float).eps
    means = np.matmul(resp.transpose(0, 2, 1), X) / nk[..., None]
    var = np.full((starts, k, d), REG_COVAR)
    weights = nk / n
    previous = np.full(starts, -np.inf)
    active = np.ones(starts, bool)
    bound = np.full(starts, -np.inf)
    for _ in range(MAX_ITERATIONS):
        a = np.flatnonzero(active)
        wlp = _log_gauss(X[None], means[a], var[a]) + np.log(weights[a])[:, None, :]
        norm = _logsumexp(wlp, axis=-1)                                  # (s, n)
        bound[a] = norm.mean(1)
        r = np.exp(wlp - norm[..., None])
        weights[a], means[a], var[a] = _m_step(X, r, lp, bounds)
        done = np.abs(bound[a] - previous[a]) < TOLERANCE
        previous[a] = bound[a]
        active[a[done]] = False
        if not active.any():
            break
    # the best start that passes the Sparrow test: the starts tried best
    # first, so the test runs once or twice rather than on every start
    for s in np.argsort(-bound, kind="stable"):
        counts = np.round(weights[s] * n).astype(np.int64)
        valid = np.flatnonzero(counts >= min_locs)
        if resolvable(means[s][valid], weights[s][valid], var[s][valid]):
            return Mixture(weights[s][valid], means[s][valid], var[s][valid],
                           counts[valid], float(bound[s]))
    return None


def g5m(X: np.ndarray, lp: np.ndarray, min_locs: int = MIN_LOCS,
        bounds: Tuple[float, float] = SIGMA_BOUNDS,
        max_rounds: int = MAX_ROUNDS_WITHOUT_BEST, seed: int = SEED
        ) -> Optional[Mixture]:
    """The molecules of one cluster: the mixture with the lowest BIC as
    components are added one by one (Picasso's `_find_optimal_G5M_2D`), or
    None when no number of components fits.  ``lp`` (n,) one precision per
    localization (2D) or (n, d)."""
    X = np.asarray(X, float)
    n, d = X.shape
    lp = np.asarray(lp, float)
    if lp.ndim == 1:
        lp = np.repeat(lp[:, None], d, axis=1)
    if n < min_locs:
        return None
    # centred: the variances and the BIC do not care, the arithmetic does
    origin = X.mean(0)
    Xc = X - origin
    most = min(MAX_COMPONENTS, n // max(min_locs, 1))
    best, best_bic, without = None, np.inf, 0
    k = 1
    while k <= most and without < max_rounds:
        model = fit_mixture(Xc, lp, k, min_locs, bounds, seed)
        if model is None or not len(model):
            without += 1
        else:
            bic = model.bic(Xc)
            if bic < best_bic:
                best, best_bic, without = model, bic, 0
            else:
                without += 1
        k += 1
    if best is not None:
        best.means = best.means + origin
    return best


def molecule_values(model: Mixture, X: np.ndarray, lp: np.ndarray, frame: np.ndarray,
                    gap: int = EVENT_GAP) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """What is measured per molecule of one cluster's fit, and each
    localization's molecule (0, 1, ...): Picasso's `_convert_G5M_results`.

    ``p_val``: a molecule's localizations' responsibility-weighted mean
    log-likelihood against its expectation for draws from that component,
    ``log(w / ((2 pi)^(d/2) sqrt(det C))) - d/2``, with the standard
    deviation of a mean of ``n w`` such terms, ``sqrt(d / 2 / (n w))``, as a
    normal cumulative probability.
    """
    from scipy.special import erf
    X = np.asarray(X, float)
    n, d = X.shape
    lp = np.asarray(lp, float)
    if lp.ndim == 1:
        lp = np.repeat(lp[:, None], d, axis=1)
    frame = np.asarray(frame, float)
    wlp = model.weighted_log_prob(X)
    norm = _logsumexp(wlp, axis=-1)
    resp = np.exp(wlp - norm[:, None])
    rsum = resp.sum(0)
    mol_ll = (resp * wlp).sum(0) / rsum
    sqrt_det = np.sqrt(model.var.prod(1))
    expected = np.log(model.weights / ((2 * np.pi) ** (d / 2) * sqrt_det)) - d / 2
    spread = np.sqrt(d * 0.5 / (n * model.weights))
    p_val = 0.5 * (1 + erf((mol_ll - expected) / (spread * np.sqrt(2))))
    sigma = np.sqrt(model.var)                                   # (k, d)
    sem = np.sqrt(model.var / (n * model.weights)[:, None])
    weighted_lp = (resp.T @ lp) / rsum[:, None]
    fmean = (resp * frame[:, None]).sum(0) / rsum
    with np.errstate(invalid="ignore", divide="ignore"):
        fstd = np.sqrt((resp * (frame[:, None] - fmean) ** 2).sum(0)
                       / ((n - 1) * rsum / n))
    labels = model.predict(X)
    event = events_in(frame.astype(np.int64), gap)
    m = int(event.max()) + 1 if n else 0
    counts = np.bincount(event, minlength=m)
    event_centres = np.stack([np.bincount(event, X[:, j], m) / counts
                              for j in range(d)], axis=1)
    n_events = np.bincount(model.predict(event_centres), minlength=len(model))
    values = {"centre": model.means, "sem": sem, "sigma": sigma,
              "sigma_rel": sigma / weighted_lp, "n_locs": model.n_locs,
              "n_events": n_events, "frame": fmean, "std_frame": fstd,
              "p_val": p_val, "log_likelihood": mol_ll}
    return values, labels


def keep(values: Dict[str, np.ndarray], n_frames: float,
         min_std_frame: float = MIN_STD_FRAME_FRACTION, min_p: float = MIN_P_VALUE,
         min_events: int = MIN_EVENTS) -> np.ndarray:
    """Picasso's post-filter of G5M's molecules (`g5m`, *postprocess*):
    frames spread over more than ``min_std_frame`` of the ``n_frames``,
    ``p_val`` above ``min_p``, more than ``min_events`` binding events."""
    return ((np.asarray(values["std_frame"]) > min_std_frame * n_frames)
            & (np.asarray(values["p_val"]) > min_p)
            & (np.asarray(values["n_events"]) > min_events))


def g5m_clusters(points: np.ndarray, lp: np.ndarray, frame: np.ndarray,
                 labels: np.ndarray, min_locs: int = MIN_LOCS,
                 bounds: Tuple[float, float] = SIGMA_BOUNDS,
                 max_rounds: int = MAX_ROUNDS_WITHOUT_BEST, seed: int = SEED,
                 max_locs: Optional[int] = None,
                 progress: Optional[Callable[[float], None]] = None
                 ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """G5M in every cluster (``labels`` 0 and up; -1 none): the molecules'
    values, concatenated over the clusters with ``label`` their cluster, and
    each localization's molecule, 1, 2, ... over the whole table (-1: none --
    in no cluster, or in one where nothing was fitted).

    A cluster with more than ``max_locs`` localizations is passed over
    (Picasso's *max locs per cluster*: a fiducial)."""
    points = np.asarray(points, float)
    labels = np.asarray(labels, np.int64)
    d = points.shape[1]
    lp = np.asarray(lp, float)
    if lp.ndim == 1:
        lp = np.repeat(lp[:, None], d, axis=1)
    use = np.flatnonzero(labels >= 0)
    order = use[np.argsort(labels[use], kind="stable")]
    ids, starts = np.unique(labels[order], return_index=True)
    bounds_ = np.r_[starts, len(order)]
    parts = []
    molecule = np.full(len(points), -1, np.int64)
    next_id = 1
    for c, cluster in enumerate(ids):
        rows = order[bounds_[c]:bounds_[c + 1]]
        if progress is not None:
            progress(c / max(len(ids), 1))
        if max_locs is not None and len(rows) > max_locs:
            continue
        model = g5m(points[rows], lp[rows], min_locs, bounds, max_rounds, seed)
        if model is None or not len(model):
            continue
        values, which = molecule_values(model, points[rows], lp[rows], frame[rows])
        values["label"] = np.full(len(model), cluster)
        parts.append(values)
        molecule[rows] = which + next_id
        next_id += len(model)
    if progress is not None:
        progress(1.0)
    if not parts:
        empty = {"centre": np.zeros((0, d)), "sem": np.zeros((0, d)),
                 "sigma": np.zeros((0, d)), "sigma_rel": np.zeros((0, d))}
        for key in ("n_locs", "n_events", "frame", "std_frame", "p_val",
                    "log_likelihood", "label"):
            empty[key] = np.zeros(0)
        return empty, molecule
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return out, molecule


def renumber(molecule: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Molecule numbers (1, 2, ...; -1 none) with those not ``kept`` (a mask
    over 1..m) set to -1 and the rest numbered again in order."""
    molecule = np.asarray(molecule, np.int64)
    kept = np.r_[False, np.asarray(kept, bool)]
    new = np.full(len(kept), -1, np.int64)
    new[kept] = np.arange(1, int(kept.sum()) + 1)
    return np.where(molecule >= 0, new[np.maximum(molecule, 0)], -1)

