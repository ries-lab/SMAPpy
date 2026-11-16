"""Which way does M2's stage z go?  The metadata's direction against what the images show, without any model.

    python z_direction.py [--movie-early 2200] [--movie-late 30000]

Every SMAPpy path assumes that a larger ``ZPositionUm`` is the objective raised: the focus moves from below
the glass into the buffer.  Bead stacks (`calibrate.input.read_bead_stacks`) order their planes by increasing
``ZPositionUm``; molecule sweeps (`molecule_stack.stage_z_um`) take z = −(stage z − focus); in situ's focus is
how far the objective was raised above the coverslip's focus (docs/uipsf/conventions.md, "Positions and the
sign of z").  If on M2 ``ZPositionUm`` were the other way round -- another focus device, or a piezo counting
down -- beads and sweeps would learn the twin of the true pupil (astigmatism and spherical flipped, z mirrored)
and could not tell: molecules and beads on the glass see no depth, so for them the twin is nearly degenerate.
A cell can: its molecules sit above the glass, behind the index mismatch.  The twin test preferring the
mirrored pupil by 800-1500 nats in every M2 in situ run (architecture.md, "In situ against molecule stacks on
M2") is what an inverted stage z would give.

Three checks:

1. **The metadata.**  For every bead stack and molecule sweep file: which frame tags name a z (the key
   ``ZPositionUm`` comes from, and any PI or focus device beside it), the first and last value, and whether z
   rises along the frames.
2. **The stack's elongation against stage z.**  Each spot's elliptical Gaussian (`molecule_stack.first_spots`)
   per frame, the median log₂(σx/σy) of each stage z.  An astigmatic PSF goes from one elongation to the other
   through focus; which one is at the *smallest* ``ZPositionUm`` is the one of molecules *above* the focus if
   the convention holds.
3. **The movie's molecules on the glass.**  In the nucleus movie the glass molecules appear only in the first
   ~2200 frames, and the focus is inside the nucleus, above them: they are *below* the focus, and the most
   defocused spots of the early frames.  Their elongation is the excess of the early frames' log₂(σx/σy) over
   the late frames' at the extremes.

What to read: if the convention holds, the movie's glass molecules (below focus) are elongated the *opposite*
way to the sweeps' spots at the smallest stage z (above focus).  The same way means ``ZPositionUm`` runs
opposite to what the code assumes for these files, and every M2 bead and sweep model is the twin.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heights import MOVIE, ROOT  # noqa: E402

SETS = {"beads": "261112_MR_Intercalib2_Beads_1_700/*/*.ome.tif",
        "ab": "261112_MR_Intercalib2_AB_50ms_coverslip/*/*.ome.tif"}
FAR = 0.4          # |log₂(σx/σy)| beyond which a spot counts as clearly elongated


def read_with_tags(path, frames=None):
    """The file's frames (all, or ``frames`` = (start, stop)) and its frame tags, as `molecule_stack.read_file`
    reads them."""
    from smappy.frametags import FrameTags
    from smappy.io.tiff import open_stack
    source = open_stack(path)
    source.tags = FrameTags()
    start, stop = (0, None) if frames is None else frames
    out = [block for _, block in source.frames(chunk=200, start=start, stop=stop)]
    return np.concatenate(out), source


def z_keys(source) -> dict:
    """Every frame tag whose name says z, focus or PI: {name: (first, last)} where the values are numbers."""
    table = source.tags.table() if source.tags is not None else {}
    out = {}
    for key, values in table.items():
        k = key.lower()
        if "z" in k.split("-")[-1] or "focus" in k or "pi" in k.split("-")[0] or "zposition" in k:
            try:
                v = np.asarray(values, float)
            except (TypeError, ValueError):
                continue
            if v.size:
                out[key] = (float(v[0]), float(v[-1]), float(np.ptp(v)))
    return out


def elongation(spots) -> np.ndarray:
    return np.log2(np.asarray(spots["sigma_x_pix"]) / np.asarray(spots["sigma_y_pix"]))


def stack_check(name, pattern):
    from smappy.calibrate.molecule_stack import first_spots, stage_z_um
    files = sorted(glob.glob(os.path.join(ROOT, pattern)))
    print(f"\n== {name}: {len(files)} files")
    lows, highs, curve_z, curve_e = [], [], [], []
    for path in files:
        frames, source = read_with_tags(path)
        z = stage_z_um(source, path)
        print(f"  {os.path.basename(os.path.dirname(path)) or os.path.basename(path)}: {len(frames)} frames")
        for key, (a, b, span) in z_keys(source).items():
            print(f"     tag {key}: first {a:.3f}, last {b:.3f}, range {span:.3f}")
        if z is None or len(z) != len(frames) or np.ptp(z) == 0:
            print("     no usable stage z")
            continue
        rising = np.mean(np.diff(z) > 0)
        print(f"     stage z (ZPositionUm): {z[0]:.3f} -> {z[-1]:.3f} µm, rising in {rising:.0%} of steps")
        spots = first_spots(frames)
        e = elongation(spots)
        zs = z[np.asarray(spots["frame"], int)]
        lo, hi = np.quantile(z, [0.1, 0.9])
        bright = np.asarray(spots["photons"]) > np.nanmedian(spots["photons"])
        e_lo, e_hi = e[(zs <= lo) & bright], e[(zs >= hi) & bright]
        lows.append(e_lo)
        highs.append(e_hi)
        print(f"     log2(sx/sy), bright spots: at the lowest 10% of stage z {np.nanmedian(e_lo):+.2f} (n {len(e_lo)}),"
              f" at the highest 10% {np.nanmedian(e_hi):+.2f} (n {len(e_hi)})")
        curve_z.append(zs - z.min())
        curve_e.append(e)
    if not curve_z:
        return None
    lo, hi = np.concatenate(lows), np.concatenate(highs)
    print(f"  all {name}: smallest stage z {np.nanmedian(lo):+.2f}, largest {np.nanmedian(hi):+.2f}")
    return curve(name, np.concatenate(curve_z), np.concatenate(curve_e))


def curve(name, dz, e, step=0.25, min_count=20):
    """The elongation through the sweep, every spot, in bins of stage z above each file's first plane.

    The 10 % extremes above miss it: there the spots are so defocused that the elliptical Gaussian falls back
    towards round (beads), or nothing is detected at all (the AB sweep starts out of focus), and a median of
    nothing was read as "−".  The verdict takes the populated bins' lower half against their upper half."""
    bins = np.round(dz / step).astype(int)
    rows = [(b * step, int((bins == b).sum()), float(np.nanmedian(e[bins == b]))) for b in np.unique(bins)]
    rows = [r for r in rows if r[1] >= min_count]
    for z0, n, m in rows:
        print(f"     stage z − first plane {z0:4.2f} µm: n {n:5d}, median log2(sx/sy) {m:+.2f}")
    if not rows:
        return None
    mid = 0.5 * (rows[0][0] + rows[-1][0])
    low = float(np.nanmedian(e[(dz < mid) & (dz >= rows[0][0] - step / 2)]))
    high = float(np.nanmedian(e[(dz >= mid) & (dz <= rows[-1][0] + step / 2)]))
    print(f"  {name}, populated range split at {mid:.2f} µm: lower half {low:+.2f}, upper half {high:+.2f}")
    return low, high


def movie_check(early_frames: int, late_start: int):
    from smappy.calibrate.molecule_stack import first_spots
    path = sorted(glob.glob(os.path.join(ROOT, MOVIE)))[0]
    print(f"\n== movie {os.path.basename(path)}: frames 0-{early_frames} against {late_start}-"
          f"{late_start + early_frames}")
    out = {}
    for label, span in (("early", (0, early_frames)), ("late", (late_start, late_start + early_frames))):
        frames, _ = read_with_tags(path, span)
        spots = first_spots(frames)
        bright = np.asarray(spots["photons"]) > np.nanmedian(spots["photons"])
        out[label] = elongation(spots)[bright]
        e = out[label]
        print(f"  {label:5s}: {len(e)} bright spots; elongated log2(sx/sy) > +{FAR}: {np.mean(e > FAR):.1%}, "
              f"< -{FAR}: {np.mean(e < -FAR):.1%}; median {np.nanmedian(e):+.2f}")
    plus = np.mean(out["early"] > FAR) - np.mean(out["late"] > FAR)
    minus = np.mean(out["early"] < -FAR) - np.mean(out["late"] < -FAR)
    side = "+" if plus > minus else "-"
    print(f"  the early frames' excess (the glass molecules, below focus): {plus:+.1%} at sx > sy, "
          f"{minus:+.1%} at sy > sx -> elongated {side}")
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie-early", type=int, default=2200)
    ap.add_argument("--movie-late", type=int, default=30000)
    a = ap.parse_args()
    stacks = {name: stack_check(name, pattern) for name, pattern in SETS.items()}
    glass = movie_check(a.movie_early, a.movie_late)
    print("\n== verdict")
    for name, r in stacks.items():
        if r is None or not np.isfinite(r[0]):
            print(f"  {name}: no elongation at the low side of the sweep")
            continue
        low_side = "+" if r[0] > 0 else "-"
        agree = low_side != glass
        print(f"  {name}: spots at the lower stage z elongated {low_side}, the movie's glass molecules {glass}: "
              + ("opposite, as the convention says (smallest z = focus below the glass)" if agree else
                 "THE SAME: the stage z runs opposite to the code's convention for these files"))


if __name__ == "__main__":
    main()
