"""How strong is the field dependence on M5?  Read from results already kept, no data needed.

    python benchmarks/uipsf/real/m5_field_map.py

`docs/uipsf/field-dependence.md`, §7.  Two views:

1. *Aberrations* (`results/m5_field.jsonl`, `m5_field.py`: each bead's own phase to order 6 beside the common
   model): per Zernike mode, the beads' spread and a plane fitted over the field, its gradient in nm of path per
   half-field (u, v normalised to ±1 over the beads' span) and the scatter it leaves.
2. *Localisation* (`results/m5_modes/*__zernike_free.npz`, the shared model's slice fits, `bead_error` = each
   slice fitted alone less the stack fit): per axis, the part of each bead's error that is a plane over the
   field, per slice within ±600 nm, as an rms; beside it the same for the beads' positions shuffled (400 times,
   the null: a plane fitted to noise), its p-value, and the plane's slope at the edge of the half-field.

The beads' own focus is removed in both (each bead's z is fitted), so neither says anything about a focal
surface: tilt and field curvature are absorbed by the beads' z.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def normalised(x, y):
    half = max(np.ptp(x), np.ptp(y)) / 2
    return (x - x.mean()) / half, (y - y.mean()) / half, half


def aberrations():
    rows = [json.loads(line) for line in open(os.path.join(RESULTS, "m5_field.jsonl"))]
    for acq in sorted({r["acquisition"] for r in rows}):
        beads = [r for r in rows if r["acquisition"] == acq and r["stage"] == "bead"]
        x, y = (np.array([r[k] for r in beads], float) for k in ("x_px", "y_px"))
        u, v, half = normalised(x, y)
        A = np.c_[np.ones_like(u), u, v]
        print(f"{acq}: {len(beads)} beads over {2 * half:.0f} px")
        for n in ("5", "6", "7", "8", "11"):
            p = np.array([r["phase_nm_path"][n] for r in beads])
            c, *_ = np.linalg.lstsq(A, p, rcond=None)
            print(f"  Noll {n:>2}: mean {p.mean():6.1f}  sd {p.std():5.1f}  gradient u {c[1]:6.1f}  v {c[2]:6.1f} "
                  f"nm/half-field; left {np.std(p - A @ c):5.1f} nm")


def plane_rms(e, A):
    c, *_ = np.linalg.lstsq(A, e, rcond=None)
    f = A @ c
    return float(np.sqrt(np.mean((f - f.mean(0)) ** 2))), c


def localisation(shuffles: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    print(f"\n{'acquisition':40s} {'beads':>5s}  per axis: plane rms nm (null median, p), slope at the edge nm")
    for path in sorted(glob.glob(os.path.join(RESULTS, "m5_modes", "*__zernike_free.npz"))):
        d = np.load(path)
        u, v, _ = normalised(d["bead_x_px"].astype(float), d["bead_y_px"].astype(float))
        A = np.c_[np.ones_like(u), u, v]
        near = np.abs(d["bead_z"][0]) <= 600
        out = []
        for axis in range(3):
            e = d["bead_error"][:, near, axis]
            e = e - np.nanmedian(e, 0)
            ok = np.isfinite(e).all(1)
            obs, c = plane_rms(e[ok], A[ok])
            null = [plane_rms(e[ok], A[ok][rng.permutation(ok.sum())])[0] for _ in range(shuffles)]
            p = (np.sum(np.array(null) >= obs) + 1) / (shuffles + 1)
            slope = np.median(np.hypot(c[1], c[2]))
            out.append(f"{'xyz'[axis]} {obs:4.1f} ({np.median(null):4.1f}, p={p:.2f}) {slope:4.1f}")
        print(f"{os.path.basename(path).split('__')[0]:40s} {len(u):5d}  " + " ; ".join(out))


if __name__ == "__main__":
    aberrations()
    localisation()
