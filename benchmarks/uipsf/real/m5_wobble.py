"""Is M5's lateral shift with z a wobble of the stage, a function of its absolute z, or the optics, of defocus?

    python benchmarks/uipsf/real/m5_wobble.py NPZ FOLDER [--bin 100] [--z-range 1000] \\
        [--out benchmarks/uipsf/results/m5_wobble.jsonl]

The user's (2 November 2026): if the stage wobbles as it moves, the shift is a curve x(Z) of the stage's absolute
position, the same for every stack, and a linear drift per stack is only its local slope -- different for
stacks centred at different heights.  If it is the optics, it follows each bead's defocus z − z_focus and its
place on the chip.  The 25 stacks of `100nm_25beads_stack` start between 34.6 and 35.6 µm, and their beads focus at
different heights, so the two can be told apart.

From `m5_sidewalk.py`'s slice fits (NPZ: each slice of each used bead localised alone on the minimal model) and
the stacks' Micro-Manager metadata (each plane's ZPositionUm), each slice's x (and y) is fitted, weighted by its
CRLB, with an offset per bead and one of:

- *line per stack*: a slope per stack in z − z_focus (the bead model's linear drift);
- *wobble*: one curve W(Z) shared by every stack, in bins of the absolute Z (``--bin``);
- *wobble + line per stack*: both;
- *optics*: a slope per bead in z − z_focus that is a plane over the chip, c0 + c·(x, y) -- a non-telecentric
  image or coma that changes over the field;
- *wobble + optics*: the wobble with the plane's slopes, so the field's part is out before the wobble is judged;
- *defocus curve*: the optics' counterpart of the wobble, one curve of z − z_focus shared by every bead (alone,
  with the plane, and with the wobble: the beads' foci span about a micrometre, so the two are told apart only
  where the foci differ);
- *line per bead*: every bead its own slope, the reference that holds both.

And pairs of beads (`m5_sidewalk.py`'s slope per bead): at the same place on the chip (``--near-px``) in other
stacks, whose slopes differ by what is not the field, against far apart in one stack, by what is.

Per model: the weighted χ² per degree of freedom and the rms of what is left (nm), so the models compare on the
same slices.  The wobble's curve is kept.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402


def plane_z(tif: str) -> np.ndarray:
    """Every plane's ZPositionUm of a Micro-Manager stack, nm, sorted ascending as the reader sorts the planes."""
    text = open(re.sub(r"\.ome\.tif$", "_metadata.txt", tif)).read()
    z = np.array([float(v) for v in re.findall(r'"ZPositionUm":\s*(-?[\d.]+)', text)]) * 1000
    # the summary repeats the first plane's position: keep one per plane, in order of acquisition
    return np.sort(z[-int(re.findall(r'"Slices":\s*(\d+)', text)[0]):]) if re.search(r'"Slices":', text) else np.sort(z)


def fit(columns, y, w):
    """Weighted least squares on a design of [n, p] columns: χ²/dof, rms of the residual, coefficients."""
    A = np.column_stack(columns)
    keep = np.abs(A).sum(0) > 0
    A = A[:, keep]
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    r = y - A @ coef
    dof = max(len(y) - A.shape[1], 1)
    full = np.zeros(len(keep))
    full[keep] = coef
    return dict(chi2_dof=round(float(np.sum(w * r ** 2) / dof), 3), rms_nm=round(float(np.sqrt(np.mean(r ** 2))), 2),
                params=int(A.shape[1])), full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("folder", help="the acquisition the NPZ was made from: its stacks' metadata")
    ap.add_argument("--bin", type=float, default=100.0)
    ap.add_argument("--z-range", type=float, default=1000.0, help="slices within this of each bead's focus, nm")
    ap.add_argument("--near-px", type=float, default=20.0, help="beads this close on the chip share its field")
    ap.add_argument("--far-px", type=float, default=100.0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_wobble.jsonl"))
    args = ap.parse_args()
    d = np.load(args.npz)
    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.ome.tif"), recursive=True))
    stacks_z = [plane_z(f) for f in files]
    found, crlb, zk, centre, stack, chip = d["found"], d["crlb"], d["zk"], d["centre"], d["stack"], d["chip"]
    E, K = found.shape[:2]
    # each slice's absolute stage z: the window is consecutive planes of its stack from `first` (`m5_media.subset`),
    # thinned to the learning's step
    step = abs(float(np.median(np.diff(zk, axis=1))))           # zk falls along the window: the emitter's height
    zabs = np.full((E, K), np.nan)
    if "first" not in d.files:
        raise SystemExit("the NPZ has no window starts: run m5_sidewalk.py again (it saves `first`)")
    for e in range(E):
        z = stacks_z[stack[e]]
        zabs[e] = z[d["first"][e] + np.arange(K) * int(round(step / np.median(np.diff(z))))]
    rel = zk - centre[:, None]
    out = dict(npz=os.path.basename(args.npz), beads=int(E), stacks=int(len(np.unique(stack))),
               stack_start_um=[round(float(z[0]) / 1000, 3) for z in stacks_z],
               focus_abs_um=dict(min=round(float(np.nanmin(zabs[np.arange(E), np.abs(rel).argmin(1)])) / 1000, 3),
                                 max=round(float(np.nanmax(zabs[np.arange(E), np.abs(rel).argmin(1)])) / 1000, 3)))
    ok = (np.abs(rel) <= args.z_range) & np.isfinite(zabs)
    bead = np.broadcast_to(np.arange(E)[:, None], (E, K))[ok]
    st = np.broadcast_to(stack[:, None], (E, K))[ok]
    t = rel[ok] / 1000.0                                             # µm from focus
    Z = zabs[ok]
    bins = np.round((Z - np.nanmin(Z)) / args.bin).astype(int)
    xy = (chip[bead] - chip.mean(0)) / 100.0                          # per 100 px from the beads' centre
    B = (bead[:, None] == np.arange(E)[None]).astype(float)           # an offset per bead
    S = (st[:, None] == np.unique(stack)[None]).astype(float)
    W = (bins[:, None] == np.arange(bins.max() + 1)[None]).astype(float)
    W = W[:, 1:]                                                      # one bin is the beads' offsets' zero
    # the optics' alternative to the wobble: one curve of the defocus, shared by every bead, in the same bins
    dbins = np.round((t * 1000 - t.min() * 1000) / args.bin).astype(int)
    R = (dbins[:, None] == np.arange(dbins.max() + 1)[None]).astype(float)[:, 1:]
    for a, axis in enumerate("xy"):
        y = found[..., a][ok]
        w = 1 / np.maximum(crlb[..., a][ok], 1e-3) ** 2
        fin = np.isfinite(y) & np.isfinite(w)
        f = lambda *cols, fin=fin, y=y, w=w: fit([c[fin] for c in cols], y[fin], w[fin])  # noqa: E731
        res = {}
        res["offsets only"], _ = f(B)
        res["line per stack"], _ = f(B, S * t[:, None])
        res["wobble"], cw = f(B, W)
        res["wobble + line per stack"], _ = f(B, W, S * t[:, None])
        res["optics"], co = f(B, t, t * xy[:, 0], t * xy[:, 1])
        res["optics + line per stack"], _ = f(B, S * t[:, None], t * xy[:, 0], t * xy[:, 1])
        res["wobble + optics"], _ = f(B, W, t, t * xy[:, 0], t * xy[:, 1])
        res["defocus curve"], cr = f(B, R)
        res["defocus curve + optics"], _ = f(B, R, t * xy[:, 0], t * xy[:, 1])
        res["wobble + defocus curve"], _ = f(B, W, R)
        res["line per bead"], _ = f(B, B * t[:, None])
        dc = np.concatenate([[0.0], cr[E:]])
        res["defocus curve"]["curve_nm"] = np.round(dc - np.median(dc), 1).tolist()
        res["optics"]["plane_nm_per_um_per_100px"] = np.round(co[-2:], 2).tolist()
        res["optics"]["slope_nm_per_um"] = round(float(co[-3]), 2)
        curve = np.concatenate([[0.0], cw[E:]])
        res["wobble"]["curve_z_um"] = np.round((np.nanmin(Z) + np.arange(len(curve)) * args.bin) / 1000, 2).tolist()
        res["wobble"]["curve_nm"] = np.round(curve - np.median(curve), 1).tolist()
        out[axis] = res
    out["pairs"] = pairs(d["slope"], d["se"], stack, chip, zabs[np.arange(E), np.abs(rel).argmin(1)],
                         d["stage"], args.near_px, args.far_px)
    emit(out, args.out)


def pairs(slope, se, stack, chip, focus_z, stage, near_px, far_px):
    """The user's (2 November 2026): beads at about the same place on the chip share its field-dependent coma, so
    their slopes differ by what is not the field.  Per class of pairs, per axis: the robust spread of the slope
    difference (1.4826·MAD, nm/µm) over what the slopes' errors alone give (√(se₁² + se₂²), median), and for
    pairs across stacks its correlation with the difference of their focus's absolute z and of the stage's x, y."""
    E = len(stack)
    i, j = np.triu_indices(E, 1)
    dist = np.hypot(*(chip[i] - chip[j]).T)
    classes = {"same place, other stack": (dist <= near_px) & (stack[i] != stack[j]),
               "same place, same stack": (dist <= near_px) & (stack[i] == stack[j]),
               "far apart, same stack": (dist >= far_px) & (stack[i] == stack[j])}
    out = {}
    for name, sel in classes.items():
        row = dict(pairs=int(sel.sum()))
        for a, axis in enumerate("xy"):
            diff = slope[i[sel], a] - slope[j[sel], a]
            noise = np.sqrt(se[i[sel], a] ** 2 + se[j[sel], a] ** 2)
            okp = np.isfinite(diff) & np.isfinite(noise)
            if okp.sum() < 3:
                continue
            spread = 1.4826 * float(np.median(np.abs(diff[okp] - np.median(diff[okp]))))
            entry = dict(spread=round(spread, 2), noise=round(float(np.median(noise[okp])), 2),
                         over_noise=round(spread / float(np.median(noise[okp])), 2))
            if "other stack" in name:
                for label, other in (("focus z", focus_z[i[sel]] - focus_z[j[sel]]),
                                     ("stage x", stage[i[sel], 0] - stage[j[sel], 0]),
                                     ("stage y", stage[i[sel], 1] - stage[j[sel], 1])):
                    # pairs as ordered by index: the sign of both differences flips together, so r is unchanged
                    good = okp & np.isfinite(other)
                    entry[f"r_{label.replace(' ', '_')}"] = round(float(np.corrcoef(diff[good], other[good])[0, 1]), 2)
            row[axis] = entry
        out[name] = row
    return out


if __name__ == "__main__":
    main()
