"""(i) on the example's real frames: the precision each fitter claims, against the one the data show.

    python benchmarks/decode/pixelation/consecutive_frames.py spline_B decode_B [...]

There is no truth in a real movie, but a molecule on in two consecutive frames is localized twice,
independently: the difference of the two positions has the variance ``s_1^2 + s_2^2`` of the two
precisions.  So the spread of ``x_t - x_t+1`` over many such pairs, against the root of the two
claimed variances (``x_err_nm`` and the rest: Spline 3D's CRLB, DECODE's predicted sigma), says
whether a fitter's errors are what it says -- within the ratio's bins of the claim, from the
localizations kept as the pixelation study keeps them (`analyse.keep`).

**Pairs:** in consecutive frames, nearest laterally within ``PAIR_NM`` and mutually so; molecules
close together make false pairs, which the robust spread (1.4826 MAD) mostly ignores.  Slow drift
over one frame is negligible.  **The ratio** is ``robust std(difference) / sqrt(mean(s_1^2 +
s_2^2))`` per bin of the claim: 1 where the claimed precision is the real one, above where the
fitter is overconfident.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyse import keep  # noqa: E402
from common import OUT  # noqa: E402

PAIR_NM = 100.0
BINS = {"x": (0, 5, 10, 15, 20, 30, 50), "y": (0, 5, 10, 15, 20, 30, 50), "z": (0, 10, 20, 30, 45, 70, 120)}


def pairs(frame, x, y):
    """Indices (i, j) of localizations in frames t and t+1 that are each other's nearest within PAIR_NM."""
    order = np.argsort(frame, kind="stable")
    frame, x, y = frame[order], x[order], y[order]
    starts = np.searchsorted(frame, np.arange(frame.max() + 2))
    out_i, out_j = [], []
    for t in range(frame.max()):
        a = slice(starts[t], starts[t + 1])
        b = slice(starts[t + 1], starts[t + 2])
        if a.start == a.stop or b.start == b.stop:
            continue
        pa = np.c_[x[a], y[a]]
        pb = np.c_[x[b], y[b]]
        d_ab, j = cKDTree(pb).query(pa, distance_upper_bound=PAIR_NM)
        d_ba, i = cKDTree(pa).query(pb, distance_upper_bound=PAIR_NM)
        ok = np.isfinite(d_ab)
        ia = np.nonzero(ok)[0]
        mutual = i[j[ok]] == ia
        out_i.append(a.start + ia[mutual])
        out_j.append(b.start + j[ok][mutual])
    i, j = np.concatenate(out_i), np.concatenate(out_j)
    return order[i], order[j]


def robust_std(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def measure(name):
    from smappy.io.hdf5 import load_localizations
    locs = load_localizations(OUT / f"{name}.hdf5")
    ok = keep(locs)
    col = {k: np.asarray(locs[k])[ok] for k in ("frame", "x_nm", "y_nm", "z_nm", "x_err_nm", "y_err_nm", "z_err_nm")}
    i, j = pairs(col["frame"].astype(np.int64), col["x_nm"], col["y_nm"])
    out = {"name": name, "pairs": int(len(i)), "kept": int(ok.sum())}
    for axis in ("x", "y", "z"):
        diff = col[f"{axis}_nm"][i] - col[f"{axis}_nm"][j]
        claim = np.sqrt(col[f"{axis}_err_nm"][i] ** 2 + col[f"{axis}_err_nm"][j] ** 2)
        rows = []
        edges = np.asarray(BINS[axis], float) * np.sqrt(2)          # the claim is of a difference
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (claim >= lo) & (claim < hi)
            if sel.sum() < 200:
                continue
            rows.append({"claim_lo": float(lo / np.sqrt(2)), "claim_hi": float(hi / np.sqrt(2)), "n": int(sel.sum()),
                         "ratio": float(robust_std(diff[sel]) / np.sqrt(np.mean(claim[sel] ** 2)))})
        good = claim < edges[-1]
        out[axis] = {"ratio": float(robust_std(diff[good]) / np.sqrt(np.mean(claim[good] ** 2))), "bins": rows}
    return out


def main(names):
    results = [measure(n) for n in names]
    with open(OUT / "consecutive_frames.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for r in results:
        print(f"{r['name']:10s} {r['pairs']:>9} pairs of {r['kept']} kept;  observed / claimed spread, all:"
              f"  x {r['x']['ratio']:.2f}  y {r['y']['ratio']:.2f}  z {r['z']['ratio']:.2f}")
        for axis in ("x", "z"):
            print("     " + axis + " by claim: " + "  ".join(f"{b['claim_lo']:.0f}-{b['claim_hi']:.0f} nm {b['ratio']:.2f}"
                                                         for b in r[axis]["bins"]))
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or ["spline_A", "decode_A", "spline_B", "decode_B"])
