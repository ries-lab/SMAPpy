#!/usr/bin/env python3
"""Simulate a sparse blinking SMLM dataset, without and with a little drift.

    python scripts/simulate_blinks.py [--out DIR] [--frames 20000] [--seed 0]

Emitters sit on a 3D structure (a tilted ring, two crossing lines, a few
scattered points) spread over ~10 um, far apart compared with a PSF.  Each
one blinks a few times; a blink lasts one to a few frames and every frame
gives one localization with noise from its photon count (precision ~
150 / sqrt(N) nm laterally, three times that in z).  Two emitters that are
active in the same frame closer than ``--min-separation`` (a PSF width)
could not have been fitted apart, so both are dropped -- the labelling is
dense, the activation sparse, as in a real experiment.  The drifted copy adds a
smooth random walk plus a slow linear creep of ~100 nm; the true drift per
frame is stored in the file's metadata as ``drift_truth`` (x, y, z in nm).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smappy.io.hdf5 import save_localizations  # noqa: E402
from smappy.locs import Localizations           # noqa: E402


def structure(rng) -> np.ndarray:
    """Emitter positions (n, 3) in nm."""
    n_ring = 1200
    t = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + rng.normal(0, 0.01, n_ring)
    ring = np.column_stack([5000 + 2500 * np.cos(t), 5000 + 2500 * np.sin(t),
                            300 * np.sin(2 * t)])                 # a tilted, wavy ring
    s = np.linspace(0, 1, 500)
    jitter = rng.normal(0, 15, (len(s), 3))          # the lines are not razor thin
    line1 = np.column_stack([1000 + 8000 * s, 1500 + 6000 * s, -200 + 400 * s]) + jitter
    line2 = np.column_stack([1500 + 7000 * s, 8500 - 7000 * s, 150 * np.ones_like(s)]) \
        + rng.normal(0, 15, (len(s), 3))
    scatter = np.column_stack([rng.uniform(500, 9500, 120), rng.uniform(500, 9500, 120),
                               rng.uniform(-300, 300, 120)])
    return np.vstack([ring, line1, line2, scatter])


def unresolvable(x, y, frame, min_separation: float) -> np.ndarray:
    """True where another emitter is active in the same frame within a PSF."""
    from scipy.spatial import cKDTree
    drop = np.zeros(len(frame), bool)
    order = np.argsort(frame, kind="stable")
    edges = np.flatnonzero(np.diff(frame[order]) != 0) + 1
    for block in np.split(order, edges):
        if block.size < 2:
            continue
        pairs = cKDTree(np.column_stack([x[block], y[block]])).query_pairs(
            min_separation, output_type="ndarray")
        if pairs.size:
            drop[block[np.unique(pairs)]] = True
    return drop


def simulate(n_frames: int, seed: int, drift: bool, density: float = 1.0,
             min_separation: float = 250.0):
    rng = np.random.default_rng(seed)
    emitters = structure(rng)
    rows = []
    for i, (x, y, z) in enumerate(emitters):
        for _ in range(rng.poisson(12 * density) + 1):      # blinks per emitter
            start = rng.integers(0, n_frames)
            length = rng.geometric(0.35)                     # 1, 2, 3 ... frames on
            for f in range(start, min(start + length, n_frames)):
                rows.append((i, f))
    rows = np.array(rows)
    emitter, frame = rows[:, 0], rows[:, 1]
    order = np.argsort(frame, kind="stable")
    emitter, frame = emitter[order], frame[order]
    n = len(frame)
    photons = rng.gamma(4.0, 500.0, n).astype(np.float32)          # mean 2000
    prec = (150.0 / np.sqrt(photons)).astype(np.float32)             # ~3-10 nm
    prec_z = (3 * prec).astype(np.float32)
    # two emitters within a PSF in one frame are one blob: neither is fitted
    lost = unresolvable(emitters[emitter, 0], emitters[emitter, 1], frame, min_separation)
    emitter, frame, n = emitter[~lost], frame[~lost], int((~lost).sum())
    photons, prec, prec_z = photons[~lost], prec[~lost], prec_z[~lost]
    true = emitters[emitter]
    xyz = true + np.column_stack([rng.normal(0, prec), rng.normal(0, prec), rng.normal(0, prec_z)])
    drift_nm = np.zeros((n_frames, 3))
    if drift:
        walk = np.cumsum(rng.normal(0, 0.6, (n_frames, 3)), axis=0)   # a random walk
        walk -= walk[0]
        creep = np.linspace(0, 1, n_frames)[:, None] * np.array([80.0, -50.0, 30.0])
        drift_nm = walk + creep
        xyz = xyz + drift_nm[frame]
    columns = {
        "frame": frame.astype(np.int64),
        "x_nm": xyz[:, 0].astype(np.float32), "y_nm": xyz[:, 1].astype(np.float32),
        "z_nm": xyz[:, 2].astype(np.float32),
        "photons": photons, "background": rng.gamma(20, 1.0, n).astype(np.float32),
        "loc_precision_nm": prec, "loc_precision_z_nm": prec_z,
        "sigma_nm": rng.normal(120, 8, n).astype(np.float32),
        "logl_rel": rng.normal(-0.5, 0.3, n).astype(np.float32),
        "emitter": emitter.astype(np.int32),                   # ground truth identity
    }
    metadata = {"units": "nm", "simulation": "smappy sparse blinks", "seed": seed,
                "n_emitters": len(emitters), "n_frames": n_frames,
                "min_separation_nm": min_separation, "n_unresolvable_dropped": int(lost.sum()),
                "drift_truth": drift_nm.round(3).tolist() if drift else None}
    return Localizations(columns, metadata)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="directory (default: here)")
    p.add_argument("--frames", type=int, default=20000)
    p.add_argument("--density", type=float, default=1.0,
                   help="scale the number of blinks per emitter")
    p.add_argument("--min-separation", type=float, default=250.0,
                   help="two emitters closer than this in one frame are both dropped (nm)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    for drift, name in ((False, "sim_blinks.hdf5"), (True, "sim_blinks_drift.hdf5")):
        locs = simulate(args.frames, args.seed, drift, args.density, args.min_separation)
        save_localizations(out / name, locs)
        per_frame = len(locs) / args.frames
        print(f"{name}: {len(locs)} localizations, {locs.metadata['n_emitters']} emitters, "
              f"{args.frames} frames, {per_frame:.1f} per frame"
              + f", {locs.metadata['n_unresolvable_dropped']} overlapping dropped"
              + (", drift ~150 nm" if drift else ""))


if __name__ == "__main__":
    main()
