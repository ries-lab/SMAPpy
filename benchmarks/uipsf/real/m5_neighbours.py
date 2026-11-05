"""Do M5's beads move with z because of something faint beside them?  Specks at each bead's focus against its slope.

    python benchmarks/uipsf/real/m5_neighbours.py NPZ FOLDER [--radius 10] [--core 4] [--k 4] [--beads BEAD.npy] \\
        [--out benchmarks/uipsf/results/m5_neighbours.jsonl]

Each bead's lateral slope with z is its own (`m5_beadplace.py`, 2 November 2026: it keeps it over ten places, and
beads differ by 18 nm/µm in x, 8 in y), which a bead's own dye cannot make -- an off-centre dye shifts the image,
it does not tilt it through focus.  The user's hypothesis: dirt or a few fluorophores close by.  A faint neighbour
inside the ROI pulls the fit towards it by an amount that changes as the bead's image grows and shrinks through
focus (and the 3D lens stretches it along x on one side of focus and y on the other).

At each bead's focus plane (`m5_sidewalk.py`'s window and focus), in a square of ±``--radius`` px about it:

- the bead's own image taken out by its azimuthal median about its centre (the round part of the PSF; the 3D
  lens's astigmatism is an m = 2 pattern, whose first moment is nil);
- *specks*: local maxima of the residual, smoothed by a Gaussian of 1 px, beyond ``--core`` px, above ``--k``
  times the smoothed background's robust noise; their number and their light against the bead's;
- *pull*: the residual's first moment beyond the core, Σ r̂ · residual over the bead's photons -- where the
  lopsided light lies and how strong it is, specks or not.

Then, between beads (a bead's neighbours move with it, so its places only average): its slope less the common one
against its pull, as a regression of each axis on both components of the pull, and the slope's spread for beads
with and without a speck.  With ``--beads`` (an id per observation, `m5_beadplace.py`'s matching) the places of a
bead are averaged first.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402
from m5_sidewalk import metadata  # noqa: E402


def planes(tif: str) -> np.ndarray:
    import tifffile
    with tifffile.TiffFile(tif) as t:
        return np.stack([p.asarray() for p in t.pages]).astype(float)


def look(img: np.ndarray, cx: float, cy: float, radius: int, core: float, k: float) -> dict:
    """Specks and pull about one bead at (cx, cy) px in one plane."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    x0, y0 = int(round(cx)), int(round(cy))
    H, W = img.shape
    if not (radius <= x0 < W - radius and radius <= y0 < H - radius):
        return {}
    a = img[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    # the bead's centre to a fraction of a pixel: the intensity-weighted centroid of its core
    c = a[np.hypot(xx, yy) <= 2]
    w = np.clip(c - np.median(a), 0, None)
    dx = np.sum(w * xx[np.hypot(xx, yy) <= 2]) / w.sum()
    dy = np.sum(w * yy[np.hypot(xx, yy) <= 2]) / w.sum()
    r = np.hypot(xx - dx, yy - dy)
    ring = np.round(r).astype(int)
    profile = np.array([np.median(a[ring == i]) for i in range(ring.max() + 1)])
    resid = a - profile[ring]
    edge = r >= radius - 1
    bg = np.median(a[edge])
    bead = float(np.clip(a - bg, 0, None)[r <= core].sum())
    smooth = gaussian_filter(resid, 1.0)
    noise = 1.4826 * np.median(np.abs(smooth[r > core] - np.median(smooth[r > core])))
    peaks = (smooth == maximum_filter(smooth, 3)) & (r > core) & (smooth > k * noise)
    light = [float(resid[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2].sum()) for py, px in zip(*np.nonzero(peaks),
                                                                                                   strict=True)]
    out = r > core
    unit = np.stack([(xx - dx)[out], (yy - dy)[out]]) / r[out]
    pull = (unit * resid[out]).sum(1) / max(bead, 1.0)
    return dict(specks=int(peaks.sum()), speck_light=float(sum(light)) / max(bead, 1.0),
                brightest=float(max(light, default=0.0)) / max(bead, 1.0), pull=pull, bead=bead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("folder")
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--core", type=float, default=4.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--beads", default=None, help="a bead id per observation (.npy), to average a bead's places")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_neighbours.jsonl"))
    args = ap.parse_args()
    d = np.load(args.npz)
    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.ome.tif"), recursive=True))
    roi = [metadata(f)["roi"] for f in files]
    stack, chip, slope = d["stack"], d["chip"], d["slope"]
    E = len(stack)
    focus = d["first"] + np.abs(d["zk"] - d["centre"][:, None]).argmin(1)   # the focus plane in its stack
    seen = []
    for s in np.unique(stack):
        stack_planes = planes(files[s])
        for e in np.flatnonzero(stack == s):
            seen.append((e, look(stack_planes[focus[e]], chip[e, 0] - roi[s][0], chip[e, 1] - roi[s][1],
                                 args.radius, args.core, args.k)))
    seen = dict(seen)
    good = np.array([bool(seen[e]) for e in range(E)])
    pull = np.array([seen[e]["pull"] if seen[e] else [np.nan, np.nan] for e in range(E)])
    specks = np.array([seen[e]["specks"] if seen[e] else -1 for e in range(E)])
    light = np.array([seen[e]["speck_light"] if seen[e] else np.nan for e in range(E)])
    bead = np.load(args.beads) if args.beads else np.arange(E)
    ids = [b for b in np.unique(bead) if good[bead == b].any()]
    mean = lambda v: np.array([np.nanmean(v[(bead == b) & good], 0) for b in ids])  # noqa: E731
    S, P, N, L = mean(slope), mean(pull), mean(specks.astype(float)), mean(light)
    dS = S - np.nanmedian(S, 0)
    out = dict(npz=os.path.basename(args.npz), beads=len(ids), radius_px=args.radius, core_px=args.core, k=args.k,
               with_speck=int((N >= 0.5).sum()), specks_median=float(np.median(N)),
               speck_light_median=round(float(np.nanmedian(L)), 4), pull_median=np.round(np.nanmedian(np.abs(P), 0),
                                                                                         4).tolist())
    for a, axis in enumerate("xy"):
        ok = np.isfinite(dS[:, a]) & np.isfinite(P).all(1)
        A = np.column_stack([np.ones(ok.sum()), P[ok]])
        coef, *_ = np.linalg.lstsq(A, dS[ok, a], rcond=None)
        r = dS[ok, a] - A @ coef
        sd = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 2 else float("nan")  # noqa: E731
        out[f"slope_{axis}"] = dict(
            spread=round(sd(dS[ok, a]), 2), spread_after_pull=round(sd(r), 2),
            r_pull_x=round(float(np.corrcoef(dS[ok, a], P[ok, 0])[0, 1]), 2),
            r_pull_y=round(float(np.corrcoef(dS[ok, a], P[ok, 1])[0, 1]), 2),
            r2=round(float(1 - np.var(r) / np.var(dS[ok, a])), 2),
            coef_per_pull=np.round(coef[1:], 1).tolist(),
            spread_with_speck=round(sd(dS[ok & (N >= 0.5), a]), 2), spread_without=round(sd(dS[ok & (N < 0.5), a]), 2),
            r_abs_speck_light=round(float(np.corrcoef(np.abs(dS[ok, a]), np.nan_to_num(L[ok]))[0, 1]), 2))
    emit(out, args.out)


if __name__ == "__main__":
    main()
