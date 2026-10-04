"""All pores together: histogram models of the corners and the localizations,
fitted jointly, on about 1000 pores.

Each estimate pools 7 simulations (about 1000 pores) and is repeated on 5
independent pools, so the spread between them is the precision at that size.
Every estimate is a maximum likelihood over histograms of all pores:

* ``counting``: the corners seen with every localization (the plugin's rule),
  the binomial corner model fitted from 4 corners up -- pores with fewer are
  the ones a segmentation misses.
* ``cutoff``: the corners seen with only the localizations better than 6, 8
  and 10 nm, each histogram with the corner model at ``LE d(q, p)``, and p
  tied to LE by the mean localizations per pore, ``p = 32 LE / mean(N)``.
* ``N``: the histogram of the localizations per pore, N, alone: the labelled
  copies of a pore are binomial in LE, and each shows a geometric number of
  blinks (mean 1/p), so N given M labelled copies is M plus a negative
  binomial.
* ``joint``: the N histogram and the cutoff histograms together, LE and p
  both free.
* ``joint_all``: the N histogram and the plain corner histogram (4..8)
  together, the corners at ``LE d(q, p)`` with q the fraction better than the
  plugin's 15.7 nm.

The corner histograms at a cutoff are fitted over 0..8: which pores are in
the sample is decided by all their localizations, not by the bright ones,
and at a tight cutoff most pores show fewer than 4 corners.

    python studies/npc_le/pooled.py        # writes pooled.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, nbinom

import bench
from smappy.plugins.npc import corner_model

HERE = Path(__file__).resolve().parent
COPIES, CORNERS, PER_CORNER = 32, 8, 4
CUTOFFS = (6.0, 8.0, 10.0)
POOL, REPLICATES = 7, 5


def seen(q, p):
    """A fluorophore with a geometric number of blinks (mean 1/p) has one
    that passes, when each passes with q."""
    return q / (q + p * (1 - q))


def corner_ll(hist, ele, lo=0, hi=CORNERS):
    k = np.arange(lo, hi + 1)
    model = corner_model(min(max(ele, 1e-9), 1 - 1e-9), CORNERS, PER_CORNER)[k]
    model = np.maximum(model / model.sum(), 1e-300)
    return float(np.sum(hist[k] * np.log(model)))


def n_pmf(n_max, le, p):
    """P(N = 0..n_max): M ~ Binomial(32, LE) labelled copies, each with a
    geometric number of blinks of mean 1/p."""
    n = np.arange(n_max + 1)
    out = np.zeros(n_max + 1)
    for m in range(COPIES + 1):
        w = binom.pmf(m, COPIES, le)
        if m == 0:
            out[0] += w
        elif p >= 1:
            if m <= n_max:
                out[m] += w
        else:
            out += w * nbinom.pmf(n - m, m, p)          # failures beyond m
    return out


def n_ll(n_hist, le, p):
    pmf = np.maximum(n_pmf(len(n_hist) - 1, le, p), 1e-300)
    return float(np.sum(n_hist * np.log(pmf)))


def fit(cost, start):
    """Minimise over (LE, p) in (0, 1) x (0, 1], through a logistic map."""
    def unpack(v):
        return 1 / (1 + np.exp(-v[0])), 1 / (1 + np.exp(-v[1]))
    v0 = [np.log(start[0] / (1 - start[0])), np.log(start[1] / (1 - start[1]))]
    best = min((minimize(lambda v: cost(*unpack(v)), [v0[0], b], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
                for b in (v0[1], -2.0, 2.0)), key=lambda r: r.fun)
    return unpack(best.x)


def estimates(pores):
    n = np.array([len(p.sigma_window) for p in pores])
    n_hist = np.bincount(n)
    sigma = np.concatenate([p.sigma_window for p in pores])
    q = {c: float(np.mean(sigma < c)) for c in CUTOFFS}
    cut_hist = {}
    for c in CUTOFFS:
        k = [bench.hard(p.theta[p.sigma < c], p.s[p.sigma < c],
                        np.ones(int(np.sum(p.sigma < c)), bool), p.phase_fit) for p in pores]
        cut_hist[c] = np.bincount(k, minlength=CORNERS + 1)
    counted = np.bincount([bench.hard(p.theta, p.s, p.kept, p.phase_fit) for p in pores],
                          minlength=CORNERS + 1)
    out = {}

    from scipy.optimize import minimize_scalar
    out["counting"] = minimize_scalar(lambda e: -corner_ll(counted, e, 4),
                                      bounds=(1e-3, 1 - 1e-3), method="bounded").x

    def cutoff_cost(le):
        p = min(COPIES * le / n.mean(), 1.0)
        return -sum(corner_ll(cut_hist[c], le * seen(q[c], p)) for c in CUTOFFS)
    out["cutoff"] = minimize_scalar(cutoff_cost, bounds=(1e-3, 1 - 1e-3),
                                    method="bounded").x

    start = (min(n.mean() / COPIES, 0.9) * 0.5, 0.5)
    le, p = fit(lambda le, p: -n_ll(n_hist, le, p), start)
    out["N"], out["p_N"] = le, p
    le, p = fit(lambda le, p: -n_ll(n_hist, le, p)
                - sum(corner_ll(cut_hist[c], le * seen(q[c], p)) for c in CUTOFFS), start)
    out["joint"], out["p_joint"] = le, p
    # the plain corner histogram (every localization better than the
    # plugin's 15.7 nm, from 4 corners up) with the N histogram
    q_all = float(np.mean(sigma < bench.PRECISION_OF_ARC * bench.STEP * bench.R))
    le, p = fit(lambda le, p: -n_ll(n_hist, le, p)
                - corner_ll(counted, le * seen(q_all, p), 4), start)
    out["joint_all"], out["p_joint_all"] = le, p
    return out


def main():
    rows = []
    for efficiency, photons, blinks in itertools.product((0.35, 0.5, 0.7), (500, 5000),
                                                         (1, 3, 10)):
        for replicate in range(REPLICATES):
            pores = []
            for seed in range(1 + replicate * POOL, 1 + (replicate + 1) * POOL):
                pores += bench.pores_of(bench.settings_for(efficiency, photons, blinks, seed))
            labelled = np.bincount([p.labelled for p in pores], minlength=CORNERS + 1)
            from scipy.optimize import minimize_scalar
            truth = minimize_scalar(lambda e: -corner_ll(labelled, e),
                                    bounds=(1e-3, 1 - 1e-3), method="bounded").x
            rows.append({"efficiency": efficiency, "photons": photons, "blinks": blinks,
                         "replicate": replicate, "pores": len(pores),
                         "ele_labelled": truth, **estimates(pores)})
            print(rows[-1], flush=True)
    with open(HERE / "pooled.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
