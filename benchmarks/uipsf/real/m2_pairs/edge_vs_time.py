"""Did the focus drift over the movie?  The coverslip's edge per block of time, one model held.

    python edge_vs_time.py OUTDIR MODEL [MODEL ...] [--blocks 10] [--json FILE]

The nucleus movie is ~72 000 frames under the focus lock.  The in situ models learn one focus for all of it; if
the focus wandered, the molecules' heights wandered with it, the depth signal that tells the even aberrations
apart is blurred, and the learning may put the blur into the pupil (an open hypothesis of architecture.md, "In
situ against molecule stacks on M2").  With the pupil held, a drift of the focus shows directly: the molecules
on the coverslip are found at a height that moves.  Each block's edge (`coverslip_edge`) is that height.

Per model (``heights_<MODEL>.npz`` from `heights.py`): the units ordered by movie and frame, cut into ``blocks``
of equal count, each block's edge with its error, the spike fraction, and the median and 90th percentile of its
heights; then a weighted line through the edges, in nm per 10 000 frames.  The frame is counted across the movies
(each movie's frames after the previous movie's last frame seen), so the line's axis is the acquisition's order.

What to read: edges that stay within ~10 nm say the focus held; a trend or a jump of tens of nm says it did not,
and then a focus per block (or per movie) in the learning is the next thing to try.  The median height follows
where the molecules are as well as the focus, so it is shown, not fitted.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heights import coverslip_edge, load  # noqa: E402


def acquisition_frame(frame, source) -> np.ndarray:
    """The frame counted across the movies: each movie after the previous one's last frame seen."""
    frame, source = np.asarray(frame, int), np.asarray(source, int)
    out = frame.copy()
    offset = 0
    for s in np.unique(source):
        sel = source == s
        out[sel] = frame[sel] + offset
        offset += int(frame[sel].max()) + 1
    return out


def block_edges(t, h, sd, blocks: int):
    order = np.argsort(t, kind="stable")
    rows = []
    for part in np.array_split(order, blocks):
        e = coverslip_edge(h[part], sd[part])
        if e is None or not np.isfinite(e.sd_nm):
            continue
        rows.append(dict(frames=(int(t[part].min()), int(t[part].max())), t=float(np.median(t[part])),
                         edge_nm=e.shift_nm, sd_nm=e.sd_nm, spike=e.spike, n=len(part),
                         median_nm=float(np.nanmedian(h[part])), p90_nm=float(np.nanpercentile(h[part], 90))))
    return rows


def trend(rows):
    """Weighted line edge = a + b·t: b in nm per 10 000 frames, its error (χ²-scaled), χ² per degree of freedom."""
    t = np.array([r["t"] for r in rows]) / 1e4
    z = np.array([r["edge_nm"] for r in rows])
    w = 1 / np.maximum(np.array([r["sd_nm"] for r in rows]), 1.0) ** 2
    A = np.c_[np.ones_like(t), t]
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * z))
    dof = max(len(z) - 2, 1)
    chi2 = float(np.sum(w * (z - A @ coef) ** 2) / dof)
    return float(coef[1]), float(np.sqrt(cov[1, 1] * max(chi2, 1.0))), chi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    for name in a.names:
        d = load(a.out, name)
        t = acquisition_frame(d["frame"], d["source"])
        rows = block_edges(t, d["h"], d["sd_h"], a.blocks)
        if len(rows) < 3:
            print(f"{name}: only {len(rows)} blocks with an edge; fewer --blocks")
            continue
        slope, err, chi2 = trend(rows)
        edges = np.array([r["edge_nm"] for r in rows])
        print(f"\n{name}: {len(rows)} blocks; edge {slope:+.1f} ± {err:.1f} nm per 10 000 frames, range "
              f"{edges.min():+.0f} to {edges.max():+.0f} nm, χ²/dof about the line {chi2:.1f}")
        for r in rows:
            print(f"   frames {r['frames'][0]:6d}-{r['frames'][1]:6d}: edge {r['edge_nm']:+7.1f} ± {r['sd_nm']:5.1f} nm"
                  f"  spike {r['spike']:.2f}  median {r['median_nm']:6.0f}  p90 {r['p90_nm']:6.0f}  n {r['n']}")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(dict(kind="edge_vs_time", model=name, blocks=a.blocks,
                                        slope_nm_per_10k_frames=slope, slope_err=err, chi2_dof=chi2,
                                        block_rows=rows)) + "\n")


if __name__ == "__main__":
    main()
