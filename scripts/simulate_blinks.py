#!/usr/bin/env python3
"""Simulate a sparse blinking SMLM dataset, without and with a little drift.

    python scripts/simulate_blinks.py [--out DIR] [--frames 20000] [--seed 0]

Emitters sit on a 3D structure (a tilted ring, two crossing lines, a few
scattered points) spread over ~10 um, far apart compared with a PSF.  Each
one blinks a few times; a blink lasts one to a few frames and every frame
gives one localization with noise from its photon count (precision ~
150 / sqrt(N) nm laterally, three times that in z).  The drifted copy adds a
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
    n_ring = 160
    t = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + rng.normal(0, 0.01, n_ring)
    ring = np.column_stack([5000 + 2500 * np.cos(t), 5000 + 2500 * np.sin(t),
                            300 * np.sin(2 * t)])                 # a tilted, wavy ring
    s = np.linspace(0, 1, 60)
    line1 = np.column_stack([1000 + 8000 * s, 1500 + 6000 * s, -200 + 400 * s])
    line2 = np.column_stack([1500 + 7000 * s, 8500 - 7000 * s, 150 * np.ones_like(s)])
    scatter = np.column_stack([rng.uniform(500, 9500, 40), rng.uniform(500, 9500, 40),
                               rng.uniform(-300, 300, 40)])
    return np.vstack([ring, line1, line2, scatter])


def simulate(n_frames: int, seed: int, drift: bool):
    rng = np.random.default_rng(seed)
    emitters = structure(rng)
    rows = []
    for i, (x, y, z) in enumerate(emitters):
        for _ in range(rng.poisson(6) + 1):                 # blinks per emitter
            start = rng.integers(0, n_frames)
            length = rng.geometric(0.4)                      # 1, 2, 3 ... frames on
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
                "drift_truth": drift_nm.round(3).tolist() if drift else None}
    return Localizations(columns, metadata)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="directory (default: here)")
    p.add_argument("--frames", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    for drift, name in ((False, "sim_blinks.hdf5"), (True, "sim_blinks_drift.hdf5")):
        locs = simulate(args.frames, args.seed, drift)
        save_localizations(out / name, locs)
        print(f"{name}: {len(locs)} localizations, {locs.metadata['n_emitters']} emitters, "
              f"{args.frames} frames" + (", drift ~100 nm" if drift else ""))


if __name__ == "__main__":
    main()
