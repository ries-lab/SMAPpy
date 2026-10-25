"""The pipeline end to end, with junk, and four ways of segmenting.

Every estimate is the joint fit of (corners at 20 nm, localizations per pore)
conditioned on at least 4 corners (`joint.fit`), against the efficiency the
simulation was run with.  The simulation has junk that is not a ring
(`npc_junk.yaml`).  The segmentations:

* ``default``: *Segment regions*' NPC preset (its *keep*);
* ``adaptive``: the segmenter's candidates with only ``n_ring >= 4``, judged again
  on the localizations better than the corner cutoff (so that imprecise ones
  do not count as outside), each fraction by a binomial test of its count
  (so that a pore with five localizations is not rejected for two outside),
  and the radius fitted on the same;
* ``none``: every candidate; only the 4 corners of the fit select.

None of the checks knows the blinking: they are fractions and a radius.

    python studies/npc_le/selection.py     # writes selection.csv
"""
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import binom

import bench
import joint
from pipeline import counting
from smappy.plugins.npc import fit_circle
from smappy.segmentation import VALUES, find_circles, judge

HERE = Path(__file__).resolve().parent
POOL, REPLICATES = 7, 5
ALPHA = 0.05                     # a fraction fails when its count is this unlikely
# the ring the NPC preset looks for, and the radii a pore may fit
R, DR, WINDOW_NM, RADII = 55.0, 15.0, 100.0, (40.0, 70.0)
RADIAL_EXTRA_NM = 5.0            # tilt and the label, beyond the precision
FAR_FRACTION = 0.01              # of |z| > 3 a pore may have
# 4 localizations, the fewest a pore showing 4 corners can have: no more, or
# the sparse pores that still show 4 corners are lost
MIN_LOCS = 4
OPEN = f"n_ring >= {MIN_LOCS}"


def candidates(grouped, keep):
    sites = find_circles(grouped, "ring", R, R, DR)
    judge(sites, keep, VALUES["ring"])
    return [site for site in sites if site["use"]]


def adaptive(grouped, cutoff=joint.CUTOFF):
    """The segmenter's candidates, judged on their precise localizations."""
    x = np.asarray(grouped["x_nm"], float)
    y = np.asarray(grouped["y_nm"], float)
    s = np.asarray(grouped["xy_err_nm"], float)
    precise = s < cutoff
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack((x[precise], y[precise])))
    xp, yp, sp = x[precise], y[precise], s[precise]
    dR = DR
    out = []
    for site in candidates(grouped, OPEN):
        near = np.asarray(tree.query_ball_point(site["center"], WINDOW_NM), int)
        n = len(near)
        if n < 3:
            continue
        c = np.asarray(site["center"], float)
        # centred on the precise localizations, the ring's radius fixed
        c = np.asarray(fit_circle(xp[near], yp[near], c, radius=R, scale=dR)[:2])
        r = np.hypot(xp[near] - c[0], yp[near] - c[1])
        radius = fit_circle(xp[near], yp[near], c, scale=dR)[2]
        if not RADII[0] <= radius <= RADII[1]:
            continue
        # Far from the ring for its own precision?  On a pore |z| > 3 happens
        # 0.3 % of the time; a filled structure, centred or seen from beside
        # it as an arc, has many.  Judged by count, so a sparse pore is not
        # rejected for one stray, and scaled by each precision, so the test is
        # the same at 500 photons as at 5000.
        z = (r - radius) / np.sqrt(sp[near] ** 2 + RADIAL_EXTRA_NM ** 2)
        far = int(np.sum(np.abs(z) > 3))
        if binom.sf(far - 1, n, FAR_FRACTION) < ALPHA:
            continue
        out.append(site["center"])
    return out


SEGMENTATIONS = {
    "default": bench.segment,
    "adaptive": adaptive,
    "none": lambda g: [q["center"] for q in candidates(g, OPEN)],
}


import os
JUNK = os.environ.get("NPC_JUNK", "1") == "1"
ONLY = [v for v in os.environ.get("NPC_SEGMENTATIONS", "").split(",") if v]
K_MINS = [int(v) for v in os.environ.get("NPC_K_MINS", "4").split(",")]


def one_condition(condition):
    efficiency, photons, blinks = condition
    segmentations = {n: f for n, f in SEGMENTATIONS.items() if not ONLY or n in ONLY}
    rows = []
    for replicate in range(REPLICATES):
        pores = {name: [] for name in segmentations}
        stats = {name: {"found": 0, "junk": 0} for name in segmentations}
        sigma_all = []
        for seed in range(1 + replicate * POOL, 1 + (replicate + 1) * POOL):
            ctx = bench.prepare(bench.settings_for(efficiency, photons, blinks, seed,
                                                   junk=JUNK))
            sigma_all.append(ctx["sigma"])
            for name, segment in segmentations.items():
                pores[name] += bench.pores_from_sites(ctx, segment(ctx["grouped"]),
                                                      truth_only=False)
                stats[name]["found"] += bench.pores_from_sites.found
                stats[name]["junk"] += bench.pores_from_sites.junk
        sigma_all = np.concatenate(sigma_all)
        row = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
               "replicate": replicate, "pores_total": POOL * 162}
        for name in segmentations:
            k, n, radius = joint.pore_counts(pores[name])
            row.update({f"{name} found": stats[name]["found"],
                        f"{name} junk": stats[name]["junk"],
                        f"{name} counting": counting(pores[name])})
            for k_min in K_MINS:
                le, p, used, _ = joint.fit(k, n, sigma_all, radius, k_min=k_min)
                tag = name if k_min == 4 else f"{name} k>={k_min}"
                row.update({f"{tag} fitted": used, f"{tag} le": le, f"{tag} p": p})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    return rows


def main():
    from multiprocessing import Pool
    conditions = list(itertools.product((0.35, 0.5, 0.7), (500, 5000), (1, 3, 10)))
    if os.environ.get("NPC_CONDITIONS"):     # "0.5/5000/1;0.7/500/3", to finish a run
        conditions = [tuple(float(v) if "." in v else int(v) for v in c.split("/"))
                      for c in os.environ["NPC_CONDITIONS"].split(";")]
    with Pool() as pool:
        rows = [r for part in pool.map(one_condition, conditions) for r in part]
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "selection.csv"
    with open(HERE / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
