"""The ELE from the bright localizations, calibrated by the pores themselves.

At a tight precision cutoff c the corners are counted without strays, but
only the fluorophores with a blink that passes are seen: with q the fraction
of grouped localizations that pass and a geometric number of blinks of mean
1/p, the measured ELE is

    E = LE q / (q + p (1 - q)).

The blinks need not be known: a pore of 32 copies holds on average
N = 32 LE / p grouped localizations.  The two equations give

    p = E q / (N q / 32 - E (1 - q)),    LE = N p / 32 .

The localizations per pore also vary between pores, and by how much says
how many times a fluorophore blinks: with a geometric number of blinks of
mean b, var(N) / mean(N) = 2b - 1 - LE b, so the first two moments of N give
b and LE without any cutoff (`moments`).

The integrated version fits one LE to the ELE at every cutoff up to a
maximum, with p tied to it by p = 32 LE / N: more of the data, one parameter.

    python studies/npc_le/selfcal.py       # writes selfcal.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np

import bench
from cutoff import ele_at

HERE = Path(__file__).resolve().parent
COPIES = 32


def self_calibrated(pores, cutoff: float):
    """``(LE, p, E, q, N)`` from the pores' localizations alone."""
    sigma = np.concatenate([p.sigma_window for p in pores])
    n = float(np.mean([len(p.sigma_window) for p in pores]))
    q = float(np.mean(sigma < cutoff))
    e = ele_at(pores, cutoff)
    p = e * q / (n * q / COPIES - e * (1 - q))
    return n * p / COPIES, p, e, q, n


def integrated(pores, cutoffs) -> float:
    """One LE for the ELE at all these cutoffs, p = 32 LE / N."""
    from scipy.optimize import minimize_scalar
    sigma = np.concatenate([p.sigma_window for p in pores])
    n = float(np.mean([len(p.sigma_window) for p in pores]))
    q = np.array([np.mean(sigma < c) for c in cutoffs])
    e = np.array([ele_at(pores, c) for c in cutoffs])
    ok = np.isfinite(e)

    def cost(le):
        p = min(COPIES * le / n, 1.0)
        return float(np.sum((le * q[ok] / (q[ok] + p * (1 - q[ok])) - e[ok]) ** 2))
    return float(minimize_scalar(cost, bounds=(1e-3, 1.0), method="bounded").x)


def moments(pores):
    """``(LE, b)`` from the mean and the variance of the localizations per pore."""
    n = np.array([len(p.sigma_window) for p in pores], float)
    m = n.mean() / COPIES
    b = max((n.var(ddof=1) / n.mean() + 1 + m) / 2, 1.0)
    return m / b, b


def main():
    rows = []
    for efficiency, photons, blinks in itertools.product((0.35, 0.5, 0.7), (500, 5000),
                                                         (1, 3, 10)):
        for seed in (1, 2, 3):
            pores = bench.pores_of(bench.settings_for(efficiency, photons, blinks, seed))
            row = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
                   "seed": seed,
                   "ele_labelled": bench.ele(np.array([p.labelled for p in pores])),
                   "ele_detected": bench.ele(np.array([p.detected for p in pores])),
                   "ele_counted": bench.ele(np.array([bench.hard(p.theta, p.s, p.kept,
                                                                 p.phase_fit)
                                                      for p in pores]))}
            for cutoff in (8.0, 10.0, 12.0):
                le, p, e, q, n = self_calibrated(pores, cutoff)
                row.update({f"le@{cutoff:g}": le, f"p@{cutoff:g}": p,
                            f"ele@{cutoff:g}": e, f"q@{cutoff:g}": q})
            row["rows_per_pore"] = n
            row["le_moments"], row["b_moments"] = moments(pores)
            for upto in (10.0, 15.0):
                use = [c for c in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0) if c <= upto]
                row[f"le_integrated<={upto:g}"] = integrated(pores, use)
            rows.append(row)
            print(efficiency, photons, blinks, seed, flush=True)
    with open(HERE / "selfcal.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
