"""The joint distribution of (corners seen, localizations) of a pore, and its
fit conditioned on at least 4 corners -- so that whatever the segmentation
loses below 4 corners matters to neither histogram.

Per blink a localization is, with the probabilities measured on the whole
field (`classes`):

* ``a``: better than the corner cutoff and in the ring band -- it counts for
  the corners (and for N);
* ``b``: better than the N cutoff and inside the N window, but not ``a`` --
  it counts for N only;
* neither.

A copy is labelled with LE and blinks a geometric number of times (mean
1/p), and an ``a`` localization spills into a given neighbour's segment with
eps (`spill.spill_eps`).  Per corner, ``E[z^N 1{channels S empty}]`` is the
copy's generating function at ``u_S(z) = 1 - a - b + b z + a z w_S``, to the
4th power; inclusion and exclusion give the 8 (stay, left, right) indicator
weights.  Around the ring a corner's state is (seen so far, spills right):
the next corner finalises it -- seen if it was, or the next spills left --
so the ring is a 4 x 4 transfer matrix, and ``trace(M(x, z)^8)`` on the
9th roots of unity in x and Z-th roots in z gives P(k, N) by a 2D FFT.

    python studies/npc_le/joint.py         # writes joint.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import bench
import spill

HERE = Path(__file__).resolve().parent
COPIES, CORNERS, PER_CORNER = 32, 8, 4
K_MIN = 4
CUTOFF, N_CUTOFF, N_WINDOW = 20.0, 30.0, 100.0
POOL, REPLICATES = 7, 5


# ------------------------------------------------------------- the model

def classes(sigma_all, radius, cutoff=CUTOFF, n_cutoff=N_CUTOFF, window=N_WINDOW):
    """``(a, b, eps)`` from the precisions of every localization of the field."""
    s = np.asarray(sigma_all, float)
    band = norm.cdf((bench.R + bench.DR - radius) / s) - norm.cdf((bench.R - bench.DR - radius) / s)
    inside = norm.cdf((window - radius) / s)
    wa = (s < cutoff) * band
    a = float(np.mean(wa))
    b = float(np.mean((s < n_cutoff) * inside)) - a
    eps = 0.0
    if a > 0:
        # Each localization at its own distance rho from the centre: the
        # angular margin to the border is pi/8 -+ the copies' offset, and its
        # angular error is width / rho, so one nearer the centre spills more.
        # Averaged over where it lands in the band (a radial Gaussian about the
        # ring), weighted as the counted ones are.
        lo, hi = bench.R - bench.DR, bench.R + bench.DR
        rho = np.linspace(lo, hi, 13)[None, :]
        density = norm.pdf((rho - radius) / s[:, None])
        width = np.sqrt(s ** 2 + spill.EXTRA_NM ** 2)[:, None]
        delta = spill.SPOKE_NM / radius
        e = 0.5 * (norm.sf((np.pi / CORNERS - delta) * rho / width)
                   + norm.sf((np.pi / CORNERS + delta) * rho / width))
        e = np.sum(e * density, axis=1) / np.maximum(density.sum(axis=1), 1e-300)
        eps = float(np.sum(wa * e) / np.sum(wa))
    return a, max(b, 0.0), eps


def copy_pgf(u, le, p):
    return 1 - le + le * p * u / (1 - (1 - p) * u)


def corner_weights(z, le, p, a, b, eps):
    """The 8 indicator weights of one corner as functions of z: (8, len(z))."""
    share = np.array([1 - 2 * eps, eps, eps])
    base = 1 - a - b + b * z

    def empty(channels):
        w = 1 - share[list(channels)].sum() if channels else 1.0
        return copy_pgf(base + a * z * w, le, p) ** PER_CORNER
    cache = {}
    out = np.zeros((8,) + np.shape(z), complex)
    for t in range(8):
        on = [c for c in range(3) if t >> c & 1]
        off = [c for c in range(3) if not t >> c & 1]
        for k in range(len(on) + 1):
            for subset in itertools.combinations(on, k):
                key = tuple(sorted(off + list(subset)))
                if key not in cache:
                    cache[key] = empty(key)
                out[t] += (-1) ** k * cache[key]
    return out


def joint_pmf(le, p, a, b, eps, n_max):
    """P(k = 0..8, N = 0..n_max - 1), N wrapped beyond n_max (choose it large)."""
    xs = np.exp(2j * np.pi * np.arange(CORNERS + 1) / (CORNERS + 1))
    zs = np.exp(2j * np.pi * np.arange(n_max) / n_max)
    w = corner_weights(zs, le, p, a, b, eps)                 # (8, Z)
    stay, left, right = (np.arange(8) & 1), (np.arange(8) >> 1 & 1), (np.arange(8) >> 2 & 1)
    m = np.zeros((len(xs), n_max, 4, 4), complex)
    for s_, r_ in itertools.product((0, 1), repeat=2):        # current (seen, spills right)
        for t in range(8):
            seen = s_ | left[t]
            new = (stay[t] | r_) * 2 + right[t]
            m[:, :, s_ * 2 + r_, new] += xs[:, None] ** seen * w[t][None, :]
    power = m
    for _ in range(3):                                        # M^8
        power = power @ power
    f = np.trace(power, axis1=2, axis2=3)                     # (9, Z)
    # inverse transform: P(k, n) = 1/(9 Z) sum_j,m f(x_j, z_m) x_j^-k z_m^-n
    pmf = np.real(np.fft.fft2(f)) / f.size                    # fft2 uses e^{-i}: the inverse here
    return np.clip(pmf, 0, None)


def fit(k, n, sigma_all, radius, k_min=K_MIN):
    """LE and p by maximum likelihood of (k, N) over the pores with k >= k_min."""
    k = np.asarray(k)
    n = np.asarray(n)
    use = k >= k_min
    k, n = k[use], n[use]
    a, b, eps = classes(sigma_all, radius)
    n_max = int(2 ** np.ceil(np.log2(max(64, 2 * n.max() + 1))))

    def cost(v):
        le, p = 1 / (1 + np.exp(-v[0])), 1 / (1 + np.exp(-v[1]))
        pmf = joint_pmf(le, p, a, b, eps, n_max)
        sel = pmf[k_min:].sum()
        return -float(np.sum(np.log(np.maximum(pmf[k, n], 1e-300))) - len(k) * np.log(sel))
    best = min((minimize(cost, [0.0, b0], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
                for b0 in (-1.5, 0.5, 2.0)), key=lambda r: r.fun)
    le, p = 1 / (1 + np.exp(-best.x[0])), 1 / (1 + np.exp(-best.x[1]))
    return le, p, int(use.sum()), (a, b, eps)


# -------------------------------------------------------------- the data

def pore_counts(pores):
    """(k at the corner cutoff, N) for each pore, and the ring's radius."""
    k = [bench.hard(p.theta[p.sigma < CUTOFF], p.s[p.sigma < CUTOFF],
                    np.ones(int(np.sum(p.sigma < CUTOFF)), bool), p.phase_fit) for p in pores]
    n = [int(np.sum((p.rho_near < N_WINDOW) & (p.sigma_near < N_CUTOFF))) for p in pores]
    rho = np.concatenate([p.sigma / p.s for p in pores])
    precise = np.concatenate([p.sigma for p in pores]) < 10
    radius = float(np.median(rho[precise])) if precise.any() else 53.7
    return np.asarray(k), np.asarray(n), radius
