"""The whole pipeline as a user runs it: the NPC segmenter with its
defaults, every site it keeps (junk included), and the estimates compared
with the efficiency the simulation was run with -- so that the pores the
segmentation misses, which are the sparsely labelled ones, count.

    python studies/npc_le/pipeline.py      # writes pipeline.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import bench
import spill
from smappy.plugins.npc import NPCSegmentSettings, corner_model

HERE = Path(__file__).resolve().parent
POOL, REPLICATES = 7, 5


def counting(pores, lo=4):
    hist = np.bincount([bench.hard(p.theta, p.s, p.kept, p.phase_fit) for p in pores],
                       minlength=9)
    k = np.arange(lo, 9)

    def cost(e):
        m = corner_model(e, 8, 4)[k]
        return -float(np.sum(hist[k] * np.log(np.maximum(m / m.sum(), 1e-300))))
    return minimize_scalar(cost, bounds=(1e-3, 1 - 1e-3), method="bounded").x


def one_condition(condition):
    efficiency, photons, blinks = condition
    rows = []
    for replicate in range(REPLICATES):
        pores, sigma_all, found, junk, copies = [], [], 0, 0, 0
        for seed in range(1 + replicate * POOL, 1 + (replicate + 1) * POOL):
            st = bench.settings_for(efficiency, photons, blinks, seed)
            pores += bench.pores_of(st, NPCSegmentSettings(), truth_only=False)
            sigma_all.append(bench.pores_of.sigma_all)
            found += bench.pores_of.found
            junk += bench.pores_of.junk
        sigma_all = np.concatenate(sigma_all)
        row = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
               "replicate": replicate, "sites": len(pores), "found": found,
               "junk": junk, "counting": counting(pores)}
        for name, lo in (("spill", 0), ("spill >=4", 4)):
            le, p, eps = spill.estimate(pores, sigma_all, 20.0, 30.0, 100.0, lo, True)
            row[name], row["p " + name] = le, p
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    from multiprocessing import Pool
    conditions = list(itertools.product((0.35, 0.5, 0.7), (500, 5000), (1, 3, 10)))
    with Pool() as pool:
        rows = [r for part in pool.map(one_condition, conditions) for r in part]
    with open(HERE / "pipeline.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
