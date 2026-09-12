"""Simulate a sparse blinking SMLM dataset, without and with a little drift.

Emitters sit on a 3D structure (a tilted ring, two crossing lines, a few
scattered points) spread over ~10 um, far apart compared with a PSF.  Each one
blinks a few times; a blink lasts one to a few frames and every frame gives one
localization with noise from its photon count (precision ~ 150 / sqrt(N) nm
laterally, three times that in z).  Two emitters active in the same frame closer
than ``min_separation`` (a PSF width) could not have been fitted apart, so both
are dropped -- the labelling is dense, the activation sparse, as in a real
experiment.  The drifted variant adds a smooth random walk plus a slow linear
creep of ~100 nm; the true drift per frame is in the metadata as ``drift_truth``.

In the package rather than in ``scripts/`` because it is also a plugin
(``File/Simulate/Blinking Structure``): something to try the whole pipeline on
without a microscope.
"""
from __future__ import annotations

import numpy as np

from .locs import Localizations


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
