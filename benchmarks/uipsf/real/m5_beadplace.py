"""The same M5 beads at nine places: does a bead keep its lateral slope and its z bend wherever it is imaged?

    python benchmarks/uipsf/real/m5_beadplace.py NPZ FOLDER [--match-px 3] [--min-places 4] [--z-range 600] \\
        [--out benchmarks/uipsf/results/m5_beadplace.jsonl]

M5, 2 November 2026 (the user's test 3, `100nm_xy_shift`): four fields of 100 nm beads, each imaged on a 3 × 3 grid
of stage positions 1 µm apart (about 9.5 px; the camera is turned 90° against the stage) and once more at the
grid's centre.  `m5_wobble.py` found each bead's slope of x against z its own (pairs within 20 px in other
stacks of the 25-bead set differ by 23 nm/µm, against 3 nm/µm of noise); and a bead scanned up and down keeps it
(`m5_updown.py`, 1.8× the noise).  Whether that is the bead or its place: aberrations change smoothly over the
chip (`field-dependence.md` §7: 6-9 nm of astigmatism per half-field), so over 20 px a slope that belongs to the
place stays put whichever bead is there, and one that belongs to the bead goes where the bead goes.  The grid's
steps also move each bead by fractions of a pixel, so a slope or bend the pixels make shows as a dependence on
the bead's place within its pixel.

From `m5_sidewalk.py`'s fits of these stacks (NPZ): each stack's shift against its field's first stack, by the
cross-correlation of their maximum projections; each bead found again in the other stacks of its field within
``--match-px`` after that shift; then, over the beads seen at ``--min-places`` places or more:

- *slopes*: dx/dz and dy/dz, nm/µm -- the spread of a bead's slope over its places (pooled, robust) against its
  slopes' errors, and the spread between beads of their means;
- *pixel phase*: each bead's slope less its own mean against the cos and sin of its place within the pixel;
- *z bend*: each slice's z about its bead's own line in z, in 100 nm bins of the defocus -- the bend each bead
  repeats over its places (its mean curve) against what changes between its places, rms over the bins.
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

PIXEL_NM = (104.7, 106.4)


def projection(tif: str) -> np.ndarray:
    import tifffile
    with tifffile.TiffFile(tif) as t:
        return np.max(np.stack([p.asarray() for p in t.pages]).astype(float), 0)


def shift(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """How far ``img`` is moved against ``ref``, (x, y) px, by the peak of their cross-correlation (whole px)."""
    c = np.fft.ifft2(np.fft.fft2(img - img.mean()) * np.conj(np.fft.fft2(ref - ref.mean()))).real
    i = np.array(np.unravel_index(c.argmax(), c.shape))
    i = np.where(i < np.array(c.shape) // 2, i, i - np.array(c.shape))
    return i[::-1].astype(float)


def robust_sd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("folder")
    ap.add_argument("--match-px", type=float, default=3.0)
    ap.add_argument("--min-places", type=int, default=4)
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--field-um", type=float, default=10.0, help="stacks whose stage lies this close: one field")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_beadplace.jsonl"))
    args = ap.parse_args()
    d = np.load(args.npz)
    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.ome.tif"), recursive=True))
    stage = np.array([metadata(f)["stage"] for f in files])
    # the fields: stacks whose stage positions lie within --field-um of the first stack of a field
    field = -np.ones(len(files), int)
    for s in range(len(files)):
        if field[s] < 0:
            field[np.hypot(*(stage - stage[s]).T) <= args.field_um] = field.max() + 1
    ref = {}
    moved = np.zeros((len(files), 2))
    for s in range(len(files)):
        first = int(np.flatnonzero(field == field[s])[0])
        ref.setdefault(first, projection(files[first]))
        moved[s] = shift(ref[first], projection(files[s])) if s != first else 0.0

    slope, se, stack, chip = d["slope"], d["se"], d["stack"], d["chip"]
    found, crlb, zk, centre = d["found"], d["crlb"], d["zk"], d["centre"]
    E = len(stack)
    home = chip - moved[stack]                                     # where each bead sits in its field's first stack
    bead = -np.ones(E, int)
    for e in np.argsort(stack):
        if bead[e] >= 0:
            continue
        same = (field[stack] == field[stack[e]]) & (np.hypot(*(home - home[e]).T) <= args.match_px) & (bead < 0)
        bead[same] = bead.max() + 1
    ids, counts = np.unique(bead, return_counts=True)
    many = ids[counts >= args.min_places]
    use = np.isin(bead, many)
    out = dict(npz=os.path.basename(args.npz), stacks=len(files), fields=int(field.max() + 1),
               shifts_px=dict(min=moved.min(0).tolist(), max=moved.max(0).tolist()),
               bead_observations=int(E), beads=int(len(ids)), beads_seen_often=int(len(many)),
               places_median=float(np.median(counts[counts >= args.min_places])) if len(many) else 0)

    # slopes: within a bead over its places, between beads
    for a, axis in enumerate("xy"):
        dev = np.concatenate([slope[bead == b, a] - np.nanmean(slope[bead == b, a]) for b in many])
        n = np.concatenate([np.full((bead == b).sum(), (bead == b).sum()) for b in many])
        within = robust_sd(dev * np.sqrt(n / np.maximum(n - 1, 1)))
        means = np.array([np.nanmean(slope[bead == b, a]) for b in many])
        out[f"slope_{axis}"] = dict(within_bead=round(within, 2), noise=round(float(np.nanmedian(se[use, a])), 2),
                                    within_over_noise=round(within / float(np.nanmedian(se[use, a])), 2),
                                    between_beads=round(robust_sd(means), 2), mean=round(float(np.nanmean(means)), 2))
        # the place within the pixel, from the stack fit's position at focus (found, nm, about the ROI's centre)
        k = np.abs(zk - centre[:, None]).argmin(1)
        sub = (chip[:, a] + found[np.arange(E), k, a] / PIXEL_NM[a]) % 1.0
        dv = slope[:, a] - np.array([np.nanmean(slope[bead == b, a]) if b in many else np.nan for b in bead])
        ok = use & np.isfinite(dv) & np.isfinite(sub)
        A = np.column_stack([np.cos(2 * np.pi * sub[ok]), np.sin(2 * np.pi * sub[ok])])
        coef, *_ = np.linalg.lstsq(A, dv[ok], rcond=None)
        out[f"slope_{axis}"]["pixel_phase_amplitude"] = round(float(np.hypot(*coef)), 2)

    # z bends: each slice's z about its bead's own line in z, in bins of the defocus
    rel = zk - centre[:, None]
    bins = np.arange(-args.z_range, args.z_range + 1, 100.0)
    curves = np.full((E, len(bins)), np.nan)
    for e in np.flatnonzero(use):
        ok = (np.abs(rel[e]) <= args.z_range) & np.isfinite(found[e, :, 2]) & (crlb[e, :, 2] > 0)
        if ok.sum() < 5:
            continue
        w = 1 / crlb[e, ok, 2] ** 2
        A = np.column_stack([np.ones(ok.sum()), rel[e, ok]])
        coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], found[e, ok, 2] * np.sqrt(w), rcond=None)[0]
        r = found[e, ok, 2] - A @ coef
        at = np.round((rel[e, ok] + args.z_range) / 100.0).astype(int)
        curves[e, at] = r
    shared = np.nanmedian(curves[use], 0)                            # the bend every bead shares
    own = np.array([np.nanmean(curves[bead == b], 0) - shared for b in many])          # each bead's repeated bend
    within = np.concatenate([curves[bead == b] - np.nanmean(curves[bead == b], 0) for b in many])
    nb = np.concatenate([np.full((bead == b).sum(), (bead == b).sum()) for b in many])
    within = within * np.sqrt(nb / np.maximum(nb - 1, 1))[:, None]
    out["z_bend"] = dict(z_nm=bins.tolist(), shared_nm=np.round(shared, 1).tolist(),
                         shared_rms=round(float(np.sqrt(np.nanmean(shared ** 2))), 2),
                         bead_own_rms=round(float(np.sqrt(np.nanmean(own ** 2))), 2),
                         within_bead_rms=round(float(np.sqrt(np.nanmean(within ** 2))), 2),
                         crlb_median=round(float(np.nanmedian(np.where(np.abs(rel) <= args.z_range, crlb[..., 2],
                                                                       np.nan)[use])), 2),
                         # a bead's own bend, its mean over n places, still holds within/√n of noise
                         own_noise_rms=round(float(np.sqrt(np.nanmean(within ** 2)) / np.sqrt(np.median(nb))), 2))
    emit(out, args.out)


if __name__ == "__main__":
    main()
