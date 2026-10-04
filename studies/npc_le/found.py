"""Does each segmentation find the pores the fit uses?

The fit takes the pores showing at least 4 corners, so a segmentation may
lose any pore below that, but must find those above it -- and if it misses
some, not by how many corners they show.  For every true pore: its corners at
its true centre, and whether each segmentation found it.

    python studies/npc_le/found.py         # prints the table
"""
import itertools

import numpy as np
import pandas as pd

import bench
import joint
import selection


def main():
    rows = []
    for efficiency, photons, blinks in itertools.product((0.35, 0.5, 0.7), (500, 5000),
                                                         (1, 3, 10)):
        for seed in (1, 2):
            ctx = bench.prepare(bench.settings_for(efficiency, photons, blinks, seed,
                                                   junk=True))
            truth = bench.pores_from_sites(ctx, ctx["centres"][ctx["pores"]])
            k_true, _, _ = joint.pore_counts(truth)
            # pores_from_sites skips a pore with no precise ring localization
            ids = sorted(bench.pores_from_sites.taken)
            found = {}
            for name, segment in selection.SEGMENTATIONS.items():
                bench.pores_from_sites(ctx, segment(ctx["grouped"]), truth_only=True)
                found[name] = set(bench.pores_from_sites.taken)
            for c, k in zip(ids, k_true):
                rows.append({"photons": photons, "blinks": blinks, "efficiency": efficiency,
                             "k": int(k), **{n: c in f for n, f in found.items()}})
    d = pd.DataFrame(rows)
    d = d[d.k >= 4]
    pd.set_option("display.width", 200)
    names = list(selection.SEGMENTATIONS)
    print("found, of the pores showing 4 corners or more:")
    print(d.groupby(["photons", "blinks"])[names].mean().round(3).to_string())
    print("\nby corners shown (all conditions):")
    print(d.groupby("k")[names].mean().round(3).to_string())
    print("\nby corners, 500 photons, 1 blink:")
    print(d[(d.photons == 500) & (d.blinks == 1)].groupby("k")[names].mean().round(3).to_string())


if __name__ == "__main__":
    main()
