"""The labelling efficiency of nuclear pores: a model for the corners a pore
shows and the localizations it has, fitted to all pores together.

Corner counting (Thevathasan et al. 2019) asks of each of a pore's 8 corners
whether a localization lies in its segment.  Counted on every localization,
the answer depends on everything but the labelling: imprecise localizations
spill into empty neighbours and open them, more blinks give more of them, and
a filter on precision loses the corners whose only blinks were dim.  The study
in ``studies/npc_le`` measured both on simulations: plain counting is off by
up to 10 points at 5000 photons per blink and fails at 500.

What is fitted instead, over all pores together, is the joint distribution of
two numbers per pore:

* ``k``, the corners seen with the localizations better than a cutoff (20 nm)
  and at least 40 nm from the centre -- tight enough that few spill, but not
  so tight that most corners go unseen; and
* ``N``, its localizations better than a looser cutoff (30 nm) within 100 nm
  of the centre.

They come from one model of the pore.  A pore has 32 copies, 4 to a corner;
each is labelled with probability LE and blinks a geometric number of times,
mean 1/p (imaging until every fluorophore has bleached).  A blink becomes a
localization of one of three classes, with probabilities measured on the
precisions of every localization of the field (`localization_classes`): good
enough for the corners (``a``), for N only (``b``), or neither.  A corner
localization lands in a given neighbour's segment with ``eps``, from its
precision, its distance from the centre and the margin to the border.  The
ring couples neighbouring corners through those spills, and around the ring
the distribution of (k, N) is exact: a corner's state is (seen so far, spills
right), the next corner finalises it, so the ring is a 4 x 4 transfer matrix,
evaluated on roots of unity in k and N and turned back by a 2D FFT
(`joint_pmf`).

Pores with few corners are the ones a segmentation loses, so the fit is the
likelihood conditioned on ``k >= k_min`` (5): a pore lost below that biases
neither histogram, and junk, which rarely shows 5 corners, mostly drops out.

N carries how often a fluorophore blinks (its mean and its spread: binomial
for one blink, wider for many), and k how many copies were labelled, so the
joint fit gives both.  The efficiency it gives is the *labelled* one: the
fluorophores whose blinks were all too dim to pass are extrapolated from the
bright ones, which is what the blinks are needed for.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

COPIES, CORNERS, PER_CORNER = 32, 8, 4
SPOKE_NM = 5.6        # a corner's two copies of one ring, either side of its centre
EXTRA_NM = 4.0        # tilt, the label and the rotation fit, beyond the precision


def localization_classes(sigma, radius: float, band: Tuple[float, float],
                         cutoff: float, n_cutoff: float, window: float,
                         corners: int = CORNERS) -> Tuple[float, float, float]:
    """``(a, b, eps)`` from the precisions of every localization of the field.

    ``a``: the chance that a blink is better than `cutoff` and lands in the
    ring `band` (a radial Gaussian about the ring); ``b``: better than
    `n_cutoff` and within `window`, but not ``a``.  ``eps``: the chance that
    an ``a`` localization lands in a given neighbour's segment -- its angular
    margin pi/8 -+ the copies' offset against its angular error, at its own
    distance from the centre, so one nearer the centre spills more.
    """
    from scipy.stats import norm
    s = np.asarray(sigma, float)
    s = s[np.isfinite(s) & (s > 0)]
    if not len(s):
        raise ValueError("no localization precisions to measure the field on")
    lo, hi = band
    inside_band = norm.cdf((hi - radius) / s) - norm.cdf((lo - radius) / s)
    weight = (s < cutoff) * inside_band
    a = float(np.mean(weight))
    b = max(float(np.mean((s < n_cutoff) * norm.cdf((window - radius) / s))) - a, 0.0)
    eps = 0.0
    if a > 0:
        rho = np.linspace(lo, hi, 13)[None, :]
        density = norm.pdf((rho - radius) / s[:, None])
        width = np.sqrt(s ** 2 + EXTRA_NM ** 2)[:, None]
        delta = SPOKE_NM / radius
        half = np.pi / corners
        e = 0.5 * (norm.sf((half - delta) * rho / width) + norm.sf((half + delta) * rho / width))
        e = np.sum(e * density, axis=1) / np.maximum(density.sum(axis=1), 1e-300)
        eps = float(np.sum(weight * e) / np.sum(weight))
    return a, b, eps


def _copy_pgf(u, le, p):
    """E[u^R] for the blinks R of one copy: none if unlabelled, else geometric."""
    return 1 - le + le * p * u / (1 - (1 - p) * u)


def _corner_weights(z, le, p, a, b, eps, per_corner):
    """For one corner, E[z^N 1{(stay, left, right) indicators = t}], t = 0..7
    (bit 0: a localization stays, 1: one spills left, 2: one spills right), by
    inclusion and exclusion over which channels are empty."""
    share = np.array([1 - 2 * eps, eps, eps])
    base = 1 - a - b + b * z
    cache: Dict[tuple, Any] = {}

    def empty(channels):
        w = 1 - share[list(channels)].sum() if channels else 1.0
        return _copy_pgf(base + a * z * w, le, p) ** per_corner
    out = np.zeros((8,) + np.shape(z), complex)
    for t in range(8):
        on = [c for c in range(3) if t >> c & 1]
        off = [c for c in range(3) if not t >> c & 1]
        for n in range(len(on) + 1):
            for subset in itertools.combinations(on, n):
                key = tuple(sorted(off + list(subset)))
                if key not in cache:
                    cache[key] = empty(key)
                out[t] += (-1) ** n * cache[key]
    return out


def joint_pmf(le: float, p: float, a: float, b: float, eps: float, n_max: int,
              corners: int = CORNERS, per_corner: int = PER_CORNER) -> np.ndarray:
    """P(k, N): ``(corners + 1, n_max)``, k the segments seen, N wrapped at
    ``n_max`` (choose it well above the largest N)."""
    xs = np.exp(2j * np.pi * np.arange(corners + 1) / (corners + 1))
    zs = np.exp(2j * np.pi * np.arange(n_max) / n_max)
    w = _corner_weights(zs, le, p, a, b, eps, per_corner)
    t = np.arange(8)
    stay, left, right = t & 1, t >> 1 & 1, t >> 2 & 1
    m = np.zeros((len(xs), n_max, 4, 4), complex)
    for seen, spills in itertools.product((0, 1), repeat=2):    # this corner
        for state in range(8):                                   # the next one
            done = seen | left[state]
            new = (stay[state] | spills) * 2 + right[state]
            m[:, :, seen * 2 + spills, new] += xs[:, None] ** done * w[state][None, :]
    f = np.trace(np.linalg.matrix_power(m, corners), axis1=2, axis2=3)
    # the inverse transform; numpy's forward FFT has the e^{-i} it needs
    return np.clip(np.real(np.fft.fft2(f)) / f.size, 0, None)


def _n_max(n) -> int:
    return int(2 ** np.ceil(np.log2(max(64, 2 * int(np.max(n)) + 1))))


def fit(k, n, a: float, b: float, eps: float, k_min: int = 5,
        corners: int = CORNERS, per_corner: int = PER_CORNER,
        start: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """LE and p by maximum likelihood of (k, N) over the pores with k >= k_min.

    ``{"efficiency", "p", "blinks", "error", "blinks_error", "pores", "pmf"}``;
    the errors are from the curvature of the likelihood at its maximum.
    """
    from scipy.optimize import minimize
    k = np.asarray(k, int)
    n = np.asarray(n, int)
    use = k >= k_min
    k, n = k[use], n[use]
    if len(k) < 5:
        raise ValueError(f"{len(k)} pores with {k_min} corners or more: at least "
                         f"5 are needed")
    n_max = _n_max(n)

    def nll(le, p):
        pmf = joint_pmf(le, p, a, b, eps, n_max, corners, per_corner)
        selected = pmf[k_min:].sum()
        return -float(np.sum(np.log(np.maximum(pmf[k, n], 1e-300)))
                      - len(k) * np.log(max(selected, 1e-300)))

    def unpack(v):
        return 1 / (1 + np.exp(-v[0])), 1 / (1 + np.exp(-v[1]))
    if start is None:
        le0 = 0.5
        p0 = float(np.clip(per_corner * corners * le0 * (a + b) / max(n.mean(), 1), 0.05, 0.95))
        start = (le0, p0)
    v0 = np.log(np.asarray(start) / (1 - np.asarray(start)))
    tries = [v0, [v0[0], -1.5], [v0[0], 1.5]]
    best = min((minimize(lambda v: nll(*unpack(v)), x0, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
                for x0 in tries), key=lambda r: r.fun)
    le, p = unpack(best.x)
    error, blinks_error = _errors(nll, le, p)
    return {"efficiency": le, "p": p, "blinks": 1 / p, "error": error,
            "blinks_error": blinks_error, "pores": int(len(k)), "k_min": k_min,
            "pmf": joint_pmf(le, p, a, b, eps, n_max, corners, per_corner)}


def _errors(nll, le, p, h=1e-3):
    """Standard errors of LE and of 1/p from a finite-difference Hessian."""
    def f(x, y):
        return nll(min(max(x, 1e-6), 1 - 1e-6), min(max(y, 1e-6), 1.0))
    hess = np.empty((2, 2))
    f0 = f(le, p)
    hess[0, 0] = (f(le + h, p) - 2 * f0 + f(le - h, p)) / h ** 2
    hess[1, 1] = (f(le, p + h) - 2 * f0 + f(le, p - h)) / h ** 2
    hess[0, 1] = hess[1, 0] = (f(le + h, p + h) - f(le + h, p - h)
                               - f(le - h, p + h) + f(le - h, p - h)) / (4 * h ** 2)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    var_le, var_p = cov[0, 0], cov[1, 1]
    if not (var_le > 0 and var_p > 0):
        return float("nan"), float("nan")
    return float(np.sqrt(var_le)), float(np.sqrt(var_p) / p ** 2)


def marginals(pmf: np.ndarray, k_min: int) -> Tuple[np.ndarray, np.ndarray]:
    """The model's corner and localization histograms over the pores with
    k >= k_min, each summing to 1."""
    part = pmf[k_min:]
    total = max(part.sum(), 1e-300)
    corners = np.zeros(pmf.shape[0])
    corners[k_min:] = part.sum(axis=1) / total
    return corners, part.sum(axis=0) / total
