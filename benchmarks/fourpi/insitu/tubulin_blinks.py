"""(docs/4pi.md §34)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_blinks.py NAME=TABLE ...

A check with no reference: one blink fitted in consecutive frames is one molecule at one place, so its two fits
should differ by their errors alone.  Pairs within 0.5 px in frames f and f + 1 (the nearest, both sure of their
fringe), on the held-out (even) blocks: the robust sd of the differences in x, y and z (phase) in nm and in units
of the fits' combined errors -- a model whose maps or spot are off scatters more than its errors say."""
import sys

import numpy as np
from scipy.spatial import cKDTree

from smappy.io.formats import load


def rsd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    locs = load(path)[0]
    c = {k: np.asarray(locs[k]) for k in locs.keys()}
    ok = ((c["frame"] // 500) % 2 == 0) & (c["fringe_probability"] > 0.9) & np.isfinite(c["z_nm"])
    idx = np.flatnonzero(ok)
    f = c["frame"][idx]
    pairs = []
    for fr in np.unique(f):
        a, b = idx[f == fr], idx[f == fr + 1]
        if not len(a) or not len(b):
            continue
        d, j = cKDTree(np.c_[c["x_nm"][b], c["y_nm"][b]]).query(np.c_[c["x_nm"][a], c["y_nm"][a]],
                                                                   distance_upper_bound=64.0)
        hit = np.isfinite(d)
        pairs.append(np.c_[a[hit], b[j[hit]]])
    p = np.concatenate(pairs)
    out = {"pairs": len(p)}
    for q, e in (("x_nm", "x_err_nm"), ("y_nm", "y_err_nm"), ("z_nm", "z_err_nm")):
        d = c[q][p[:, 0]] - c[q][p[:, 1]]
        s = np.hypot(c[e][p[:, 0]], c[e][p[:, 1]])
        out[q] = [round(rsd(d), 2), round(rsd(d / s), 3)]
    print(name, out, flush=True)
