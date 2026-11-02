"""Do neighbouring M5 beads share their tilt?  The beads' slopes with z correlated against their distance.

    python benchmarks/uipsf/real/m5_tiltmap.py NPZ [NPZ ...] [--bins 0 25 50 100 150 250 400] [--beads ID.npy ...]

Each bead's lateral slope with z is a tilt of its pupil intensity (`m5_beadtilt.py`, r 0.84-0.96, in oil as much as
in buffer), and it moves with the bead over 2 µm (`m5_beadplace.py`).  The user's question (2 November 2026): is it
a fingerprint that is not the bead's own but not local either -- dirt or a scratch on the coverslip, residue, a
smudge on the objective, dirt in the oil -- spread over more than those 2 µm?  Two kinds, told apart by what moves:

- *fixed to the sample*: neighbours in one field share it -- pairs in the same stack, by their distance;
- *fixed to the objective*: beads at the same place on the chip share it, whichever field they are in -- pairs
  in different stacks, by their distance on the chip (camera ROI offsets added).

From `m5_sidewalk.py`'s slopes (NPZ), each bead's slope less a plane over the chip (the field's large-scale
part is not the question), the places of one bead averaged first where ``--beads`` names them.  Per kind and
distance bin: the semivariance ½⟨(s_i − s_j)²⟩ less the slopes' own noise ½⟨se_i² + se_j²⟩, as a correlation
ρ = 1 − γ / σ² against the beads' signal variance σ² (their variance less their noise); 0 is no sharing, 1 is the
same tilt.  Its uncertainty is a bootstrap over the beads (pairs share beads, so pairs are not independent).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402


def detrended(slope, chip):
    """The slopes less their least-squares plane over the chip, per axis."""
    A = np.column_stack([np.ones(len(chip)), chip - chip.mean(0)])
    out = slope.copy()
    for a in range(slope.shape[1]):
        ok = np.isfinite(slope[:, a])
        coef, *_ = np.linalg.lstsq(A[ok], slope[ok, a], rcond=None)
        out[:, a] = slope[:, a] - A @ coef
    return out


def correlation(s, se, stack, chip, bins, same_stack: bool):
    """ρ per distance bin for one axis: [(n pairs, ρ)], from the beads given."""
    i, j = np.triu_indices(len(s), 1)
    keep = (stack[i] == stack[j]) if same_stack else (stack[i] != stack[j])
    i, j = i[keep], j[keep]
    d = np.hypot(*(chip[i] - chip[j]).T)
    sig2 = np.nanvar(s) - np.nanmean(se ** 2)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (d >= lo) & (d < hi) & np.isfinite(s[i]) & np.isfinite(s[j])
        if m.sum() < 3:
            out.append((int(m.sum()), np.nan))
            continue
        gamma = 0.5 * np.mean((s[i[m]] - s[j[m]]) ** 2) - 0.5 * np.mean(se[i[m]] ** 2 + se[j[m]] ** 2)
        out.append((int(m.sum()), 1 - gamma / sig2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--beads", nargs="*", default=[],
                    help="per NPZ: a bead id per observation (.npy: results/m5_sidewalk/xyshift_bead.npy) or -")
    ap.add_argument("--bins", type=float, nargs="+", default=[0, 25, 50, 100, 150, 250, 400])
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_tiltmap.jsonl"))
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    for n, path in enumerate(args.npz):
        d = np.load(path)
        slope, se, stack, chip = d["slope"], d["se"], d["stack"], d["chip"]
        ids = args.beads[n] if n < len(args.beads) else "-"
        if ids != "-":
            # a bead's places averaged: one row per bead, at its first place (its places are within 2 µm)
            bead = np.load(ids)
            u = np.unique(bead)
            first = np.array([np.flatnonzero(bead == b)[0] for b in u])
            cnt = np.array([(bead == b).sum() for b in u])
            slope = np.array([np.nanmean(slope[bead == b], 0) for b in u])
            se = np.array([np.sqrt(np.nanmean(se[bead == b] ** 2, 0) / c) for b, c in zip(u, cnt, strict=True)])
            stack, chip = stack[first], chip[first]
        s = detrended(slope, chip)
        row = dict(npz=os.path.basename(path), beads=int(len(s)), stacks=int(len(np.unique(stack))),
                   bins_px=args.bins)
        for a, axis in enumerate("xy"):
            for kind, same in (("same stack (sample)", True), ("other stacks (objective)", False)):
                est = correlation(s[:, a], se[:, a], stack, chip, args.bins, same)
                boots = []
                for _ in range(args.boot):
                    k = rng.integers(0, len(s), len(s))
                    # a bead drawn twice is a pair at distance 0 with itself: drop those pairs by jittering ids
                    boots.append([r for _, r in correlation(s[k, a], se[k, a], stack[k], chip[k] + rng.normal(
                        0, 1e-3, (len(k), 2)), args.bins, same)])
                boots = np.array(boots, float)
                spread = np.nanstd(boots, 0)
                row[f"{axis} {kind}"] = [dict(pairs=p, rho=None if not np.isfinite(r) else round(float(r), 2),
                                              se=None if not np.isfinite(r) else round(float(spread[b]), 2))
                                         for b, (p, r) in enumerate(est)]
        emit(row, args.out)


if __name__ == "__main__":
    main()
