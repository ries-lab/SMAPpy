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
RADIAL_EXTRA_NM = 5.0            # tilt and the label, beyond the precision
FAR_FRACTION = 0.01              # of |z| > 3 a pore may have
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
    xp, yp, sp = x[precise], y[precise], s[precise]
    R, dR = settings.radius_nm, settings.ring_width_nm
    out = []
    for site in segment_npcs(grouped, OPEN):
        near = np.asarray(tree.query_ball_point(site["center"], settings.window_nm), int)
        n = len(near)
        if n < 3:
            continue
        c = np.asarray(site["center"], float)
        # centred on the precise localizations, the ring's radius fixed
        c = np.asarray(fit_circle(xp[near], yp[near], c, radius=R, scale=dR)[:2])
        r = np.hypot(xp[near] - c[0], yp[near] - c[1])
        radius = fit_circle(xp[near], yp[near], c, scale=dR)[2]
        if not settings.min_radius_nm <= radius <= settings.max_radius_nm:
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
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "selection.csv"
    with open(HERE / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
