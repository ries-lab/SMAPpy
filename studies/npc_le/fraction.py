"""The apparent efficiency against the fraction of the data used.

Each pore keeps a random fraction f of its grouped rows -- blinks, which are
interchangeable, unlike frame windows that also carry activation and
bleaching -- and its corners are counted again with the plugin's rule.  True
corners are found and then saturate; strays keep adding corners in
proportion to the data.  Does the curve's shape tell the two apart?

    python studies/npc_le/fraction.py      # writes fraction.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np

import bench

HERE = Path(__file__).resolve().parent
FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
REPEATS = 5


def curve(pores, rng):
    out = []
    for f in FRACTIONS:
        values = []
        for _ in range(REPEATS if f < 1 else 1):
            n = []
            for p in pores:
                keep = rng.random(len(p.theta)) < f
                n.append(bench.hard(p.theta[keep], p.s[keep], p.kept[keep], p.phase_fit))
            values.append(bench.ele(np.asarray(n)))
        out.append(float(np.nanmean(values)))
    return out


def main():
    rows = []
    rng = np.random.default_rng(0)
    for efficiency, photons, blinks in itertools.product((0.35, 0.5), (500, 5000), (1, 3, 10)):
        for seed in (1, 2):
            pores = bench.pores_of(bench.settings_for(efficiency, photons, blinks, seed))
            detected = bench.ele(np.array([p.detected for p in pores]))
            for f, e in zip(FRACTIONS, curve(pores, rng)):
                rows.append({"efficiency": efficiency, "photons": photons, "blinks": blinks,
                             "seed": seed, "fraction": f, "ele": e, "ele_detected": detected})
            print(efficiency, photons, blinks, seed, flush=True)
    with open(HERE / "fraction.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
