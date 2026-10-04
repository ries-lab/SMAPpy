"""Histogram models for the corners and the localizations of a pore, with
spillage between corners, and their joint fit over all pores.

The blinking model is the one of `pooled.py`: a pore has 32 copies, 4 per
corner; each is labelled with LE and shows a geometric number of blinks of
mean 1/p; a blink passes a precision cutoff with q (the fraction of all
localizations of the field that pass it).  Thinned that way, the passing
blinks of one copy are a zero-modified geometric.

**Spillage.**  A counted localization of corner k lands in the segment of a
given neighbour with probability eps, from its precision and the margin to
the border: the half-segment arc pi r / 8, less the +-5.6 nm offset of the
corner's two copies along the ring, with an extra spread for tilt and the
rotation fit.  Localizations spill independently, so a corner's localizations
split into stay / left / right with (1 - 2 eps, eps, eps).  A segment is seen
when its own corner keeps one or a neighbour spills one into it.  Around the
ring of 8 the distribution of the segments seen is exact through a 64 x 64
transfer matrix over the (stay, left, right) indicators of consecutive
corners, evaluated at the 9th roots of unity.

    python studies/npc_le/spill.py         # writes spill.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, norm

import bench

HERE = Path(__file__).resolve().parent
COPIES, CORNERS, PER_CORNER = 32, 8, 4
SPOKE_NM = 5.6                   # a corner's two copies, either side of its centre
EXTRA_NM = 4.0                   # tilt and rotation, added to the precision
POOL, REPLICATES = 7, 5


# ------------------------------------------------------------- the copies

def copy_pgf(z, le, p, q):
    """E[z^R] for the passing blinks R of one copy (unlabelled: R = 0)."""
    u = 1 - q + q * z
    return 1 - le + le * p * u / (1 - (1 - p) * u)


def copy_pmf(n_max, le, p, q):
    """P(R = 0..n_max) for one copy: 1 - LE, or a thinned geometric."""
    a = 1 - (1 - p) * (1 - q)
    rho = (1 - p) * q / a
    r = np.arange(n_max + 1)
    pmf = p * (1 - q) / a * rho ** r
    pmf[1:] += p * q / a * rho ** (r[1:] - 1)
    out = le * pmf
    out[0] += 1 - le
    return out


def n_pmf(n_max, le, p, q):
    """P(N = 0..n_max), N the passing localizations of a pore of 32 copies."""
    one = copy_pmf(n_max, le, p, q)
    total = np.zeros(n_max + 1)
    total[0] = 1.0
    for _ in range(COPIES):
        total = np.convolve(total, one)[:n_max + 1]
    return total


# ------------------------------------------------------------ the corners

def indicator_probabilities(le, p, q, eps):
    """P(stay>0, left>0, right>0) for one corner, as a vector over the 8
    combinations (bit 0 stay, bit 1 left, bit 2 right), by inclusion and
    exclusion over which channels are empty."""
    share = np.array([1 - 2 * eps, eps, eps])

    def empty(channels):            # P(every channel in the set gets none)
        z = 1 - share[list(channels)].sum() if channels else 1.0
        return copy_pgf(z, le, p, q) ** PER_CORNER
    out = np.zeros(8)
    for t in range(8):
        on = [c for c in range(3) if t >> c & 1]
        off = [c for c in range(3) if not t >> c & 1]
        # P(on all > 0, off all = 0) = sum over subsets S of on of (-1)^|S| P(S + off empty)
        total = 0.0
        for k in range(len(on) + 1):
            for subset in itertools.combinations(on, k):
                total += (-1) ** k * empty(tuple(off) + subset)
        out[t] = total
    return np.clip(out, 0, None)


_A, _B, _C = np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij")
# the middle corner b is seen when it keeps one, or a spills right, or c left
SEEN = ((_B & 1) | (_A >> 2 & 1) | (_C >> 1 & 1)).astype(int)
ROOTS = np.exp(2j * np.pi * np.arange(CORNERS + 1) / (CORNERS + 1))


def corner_pmf(le, p, q, eps):
    """P(segments seen = 0..8)."""
    prob = indicator_probabilities(le, p, q, eps)
    out = np.zeros(CORNERS + 1)
    values = []
    for x in ROOTS:
        # M[(a, b), (b, c)] = P(c) x^seen(a, b, c)
        m = np.zeros((64, 64), complex)
        for a, b, c in itertools.product(range(8), repeat=3):
            m[a * 8 + b, b * 8 + c] = prob[c] * x ** SEEN[a, b, c]
        values.append(np.trace(np.linalg.matrix_power(m, CORNERS)))
    values = np.asarray(values)
    for n in range(CORNERS + 1):
        out[n] = np.real(np.sum(values * ROOTS ** (-n))) / (CORNERS + 1)
    return np.clip(out, 0, None)


def spill_eps(sigma, radius):
    """The chance that a localization of this precision lands in a given
    neighbour's segment, averaged over the corner's two copies."""
    margin = np.pi * radius / CORNERS
    width = np.sqrt(np.asarray(sigma) ** 2 + EXTRA_NM ** 2)
    return float(np.mean(0.5 * (norm.sf((margin - SPOKE_NM) / width)
                                + norm.sf((margin + SPOKE_NM) / width))))


# ------------------------------------------------------------ the fitting

def corner_ll(hist, pmf, lo):
    k = np.arange(lo, CORNERS + 1)
    model = np.maximum(pmf[k] / max(pmf[k].sum(), 1e-300), 1e-300)
    return float(np.sum(hist[k] * np.log(model)))


def n_ll(hist, le, p, q):
    pmf = np.maximum(n_pmf(len(hist) - 1, le, p, q), 1e-300)
    return float(np.sum(hist * np.log(pmf)))


def fit(cost, start=(0.5, 0.5)):
    def unpack(v):
        return 1 / (1 + np.exp(-v[0])), 1 / (1 + np.exp(-v[1]))
    v0 = np.log(np.asarray(start) / (1 - np.asarray(start)))
    best = min((minimize(lambda v: cost(*unpack(v)), [v0[0], b], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 1000})
                for b in (v0[1], -2.0, 2.0)), key=lambda r: r.fun)
    return unpack(best.x)


def histograms(pores, sigma_all, cutoff, n_cutoff, n_window):
    """The corners seen at `cutoff`, the N histogram (localizations better
    than `n_cutoff` within `n_window` of the centre), both q's and eps."""
    q = float(np.mean(sigma_all < cutoff))
    q_n = float(np.mean(sigma_all < n_cutoff))
    k = []
    used = []
    for p in pores:
        use = p.sigma < cutoff
        k.append(bench.hard(p.theta[use], p.s[use], np.ones(int(use.sum()), bool),
                            p.phase_fit))
        used.append(p.sigma[use])
    n = [int(np.sum((p.rho_near < n_window) & (p.sigma_near < n_cutoff))) for p in pores]
    # the ring's radius: the median distance of the precise ring localizations
    rho = np.concatenate([p.sigma / p.s for p in pores])
    precise = np.concatenate([p.sigma for p in pores]) < 10
    radius = float(np.median(rho[precise])) if precise.any() else 53.7
    eps = spill_eps(np.concatenate(used), radius)
    return (np.bincount(k, minlength=CORNERS + 1), np.bincount(n), q, q_n, eps)


def estimate(pores, sigma_all, cutoff, n_cutoff, n_window, lo, spill=True):
    k_hist, n_hist, q, q_n, eps = histograms(pores, sigma_all, cutoff, n_cutoff, n_window)
    e = eps if spill else 0.0

    def cost(le, p):
        return -n_ll(n_hist, le, p, q_n) - corner_ll(k_hist, corner_pmf(le, p, q, e), lo)
    le, p = fit(cost)
    return le, p, eps


VARIANTS = {
    # name: cutoff, N cutoff, N window, fit from, spill model
    "tight 6, N all": (6.0, 1e9, 110.0, 0, False),
    "tight 6, N<30": (6.0, 30.0, 100.0, 0, False),
    "15 no spill, N<30": (15.0, 30.0, 100.0, 0, False),
    "15 spill, N<30": (15.0, 30.0, 100.0, 0, True),
    "15 spill, N<30, >=4": (15.0, 30.0, 100.0, 4, True),
    "15 spill, N<20": (15.0, 20.0, 100.0, 0, True),
}


def main():
    rows = []
    for efficiency, photons, blinks in itertools.product((0.35, 0.5, 0.7), (500, 5000),
                                                         (1, 3, 10)):
        for replicate in range(REPLICATES):
            pores, sigma_all = [], []
            for seed in range(1 + replicate * POOL, 1 + (replicate + 1) * POOL):
                pores += bench.pores_of(bench.settings_for(efficiency, photons, blinks, seed))
                sigma_all.append(bench.pores_of.sigma_all)
            sigma_all = np.concatenate(sigma_all)
            labelled = np.bincount([p.labelled for p in pores], minlength=CORNERS + 1)
            from scipy.optimize import minimize_scalar
            from smappy.plugins.npc import corner_model
            truth = minimize_scalar(
                lambda e: -np.sum(labelled * np.log(np.maximum(
                    corner_model(e, CORNERS, PER_CORNER), 1e-300))),
                bounds=(1e-3, 1 - 1e-3), method="bounded").x
            row = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
                   "replicate": replicate, "pores": len(pores), "ele_labelled": truth}
            for name, args in VARIANTS.items():
                le, p, eps = estimate(pores, sigma_all, *args)
                row[name] = le
                row["p " + name] = p
                row["eps " + name] = eps
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                   if not k.startswith(("p ", "eps "))}, flush=True)
    with open(HERE / "spill.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
