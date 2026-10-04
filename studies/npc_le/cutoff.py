"""The ELE of the bright localizations, and extrapolated to all of them.

Only localizations more precise than a cutoff are counted.  A fluorophore is
seen when one of its blinks passes; with q the fraction of grouped
localizations that pass and a geometric number of blinks of mean 1/p,

    d(q) = q / (q + p (1 - q)),

and the measured ELE is LE d(q).  Fitting LE and p to the measured ELE over
several cutoffs extrapolates to d = 1.

    python studies/npc_le/cutoff.py        # writes cutoff.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import bench
from smappy.plugins.npc import LabelingEfficiencySettings, labeling_efficiency

HERE = Path(__file__).resolve().parent
CUTOFFS = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
FIT = LabelingEfficiencySettings(bootstrap=0, fit_min=1)


def ele_at(pores, cutoff):
    n = []
    for p in pores:
        use = p.sigma < cutoff
        n.append(bench.hard(p.theta[use], p.s[use], np.ones(use.sum(), bool), p.phase_fit))
    try:
        return labeling_efficiency(n, FIT)["efficiency"]
    except ValueError:
        return float("nan")


def seen(q, p):
    return q / (q + p * (1 - q))


def extrapolate(q, e):
    """LE and p from the measured ELE at each pass fraction q."""
    ok = np.isfinite(e)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    out = least_squares(lambda v: v[0] * seen(q[ok], v[1]) - e[ok], [e[ok].max(), 0.5],
                        bounds=([0, 1e-3], [1, 1]))
    return float(out.x[0]), float(out.x[1])


def main():
    rows = []
    for efficiency, photons, blinks in itertools.product((0.35, 0.5, 0.7), (500, 5000),
                                                         (1, 3, 10)):
        for seed in (1, 2, 3):
            pores = bench.pores_of(bench.settings_for(efficiency, photons, blinks, seed))
            sigma = np.concatenate([p.sigma_window for p in pores])
            q = np.array([np.mean(sigma < c) for c in CUTOFFS])
            e = np.array([ele_at(pores, c) for c in CUTOFFS])
            detected = bench.ele(np.array([p.detected for p in pores]))
            labelled = bench.ele(np.array([p.labelled for p in pores]))
            for upto in (12.0, 15.0, 25.0):
                use = np.asarray(CUTOFFS) <= upto
                le, p_bleach = extrapolate(q[use], e[use])
                rows.append({"efficiency": efficiency, "photons": photons, "blinks": blinks,
                             "seed": seed, "fit_upto": upto, "le_extrapolated": le,
                             "p_fitted": p_bleach,
                             "ele_detected": detected, "ele_labelled": labelled,
                             **{f"q{c:g}": v for c, v in zip(CUTOFFS, q)},
                             **{f"ele{c:g}": v for c, v in zip(CUTOFFS, e)}})
            print(efficiency, photons, blinks, seed, flush=True)
    with open(HERE / "cutoff.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
