"""The pipeline end to end, with junk, and four ways of segmenting.

Every estimate is the joint fit of (corners at 20 nm, localizations per pore)
conditioned on at least 4 corners (`joint.fit`), against the efficiency the
simulation was run with.  The simulation has junk that is not a ring
(`npc_junk.yaml`).  The segmentations:

* ``default``: the NPC segmenter with its defaults;
* ``lenient``: *min localizations* 3, its checks as they are;
* ``adaptive``: the segmenter's candidates with its checks off, judged again
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
from smappy.plugins.npc import NPCSegmentSettings, fit_circle, segment_npcs

HERE = Path(__file__).resolve().parent
POOL, REPLICATES = 7, 5
ALPHA = 0.05                     # a fraction fails when its count is this unlikely
# on the precise localizations a pore has a few percent inside and outside, so
# the limits can be tighter than the segmenter's, which count every one
CHECKS = NPCSegmentSettings(max_inside=0.1, max_outside=0.2)
OPEN = NPCSegmentSettings(min_locs=3, min_radius_nm=0.0, max_radius_nm=1e9,
                          max_inside=1.0, max_outside=1.0, min_spread_nm=0.0)


def adaptive(grouped, settings=CHECKS, cutoff=joint.CUTOFF):
    """The segmenter's candidates, judged on their precise localizations."""
    x = np.asarray(grouped["x_nm"], float)
    y = np.asarray(grouped["y_nm"], float)
    s = np.asarray(grouped["xy_err_nm"], float)
    precise = s < cutoff
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack((x[precise], y[precise])))
    xp, yp = x[precise], y[precise]
    R, dR = settings.radius_nm, settings.ring_width_nm
    out = []
    for site in segment_npcs(grouped, OPEN):
        near = np.asarray(tree.query_ball_point(site["center"], settings.window_nm), int)
        n = len(near)
        if n < 3:
            continue
        c = site["center"]
        r = np.hypot(xp[near] - c[0], yp[near] - c[1])
        inside, outside = int(np.sum(r < R - dR)), int(np.sum(r > R + dR))
        if binom.sf(inside - 1, n, settings.max_inside) < ALPHA:
            continue
        if binom.sf(outside - 1, n, settings.max_outside) < ALPHA:
            continue
        radius = fit_circle(xp[near], yp[near], c, scale=dR)[2]
        if not settings.min_radius_nm <= radius <= settings.max_radius_nm:
            continue
        out.append(c)
    return out


SEGMENTATIONS = {
    "default": lambda g: [q["center"] for q in segment_npcs(g, NPCSegmentSettings())
                          if q["use"]],
    "lenient": lambda g: [q["center"] for q in
                          segment_npcs(g, NPCSegmentSettings(min_locs=3)) if q["use"]],
    "adaptive": adaptive,
    "none": lambda g: [q["center"] for q in segment_npcs(g, OPEN)],
}


def one_condition(condition):
    efficiency, photons, blinks = condition
    rows = []
    for replicate in range(REPLICATES):
        pores = {name: [] for name in SEGMENTATIONS}
        stats = {name: {"found": 0, "junk": 0} for name in SEGMENTATIONS}
        sigma_all = []
        for seed in range(1 + replicate * POOL, 1 + (replicate + 1) * POOL):
            ctx = bench.prepare(bench.settings_for(efficiency, photons, blinks, seed,
                                                   junk=True))
            sigma_all.append(ctx["sigma"])
            for name, segment in SEGMENTATIONS.items():
                pores[name] += bench.pores_from_sites(ctx, segment(ctx["grouped"]),
                                                      truth_only=False)
                stats[name]["found"] += bench.pores_from_sites.found
                stats[name]["junk"] += bench.pores_from_sites.junk
        sigma_all = np.concatenate(sigma_all)
        row = {"efficiency": efficiency, "photons": photons, "blinks": blinks,
               "replicate": replicate, "pores_total": POOL * 162}
        for name in SEGMENTATIONS:
            k, n, radius = joint.pore_counts(pores[name])
            le, p, used, _ = joint.fit(k, n, sigma_all, radius)
            row.update({f"{name} found": stats[name]["found"],
                        f"{name} junk": stats[name]["junk"],
                        f"{name} fitted": used, f"{name} le": le, f"{name} p": p,
                        f"{name} counting": counting(pores[name])})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    return rows


def main():
    from multiprocessing import Pool
    conditions = list(itertools.product((0.35, 0.5, 0.7), (500, 5000), (1, 3, 10)))
    with Pool() as pool:
        rows = [r for part in pool.map(one_condition, conditions) for r in part]
    with open(HERE / "selection.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
