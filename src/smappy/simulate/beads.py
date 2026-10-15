"""Simulated fiducial beads: bright, still, seen in nearly every frame.

What the fiducial drift correction (`smappy.fiducials`) is tested on and
drawn with.  A bead sits where it was put and moves only with the drift;
it is localized in each frame at a precision set by its photons, missed in
a frame now and then (``p_missed``), and over longer stretches when it is
``gap``-ped; one may land on the glass partway through (``late``).  The
truth is ``bead_true`` (0, 1, ...).  Added to a blinking structure simulated
with the same drift (`with_beads`), it is a table with beads and structure,
as an acquisition with fiducials is.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations, concat


def simulate_beads(n_frames: int, drift_nm: Optional[np.ndarray] = None, *,
                   n_beads: int = 5, field_nm: Tuple[float, float, float, float]
                   = (0.0, 0.0, 10000.0, 10000.0), photons: float = 30000.0,
                   precision_nm: float = 3.0, z_precision_nm: float = 8.0,
                   p_missed: float = 0.02,
                   gaps: Sequence[Tuple[int, int, int]] = (),
                   late: Sequence[Tuple[int, int]] = (),
                   seed: int = 0) -> Localizations:
    """Beads at random places in ``field_nm`` (x0, y0, x1, y1), drifting by
    ``drift_nm`` ((n_frames, 3), default none).  ``gaps``: (bead, first,
    last) frames it is not seen in; ``late``: (bead, frame) it first
    appears in."""
    rng = np.random.default_rng([seed, 7])
    drift = np.zeros((n_frames, 3)) if drift_nm is None else np.asarray(drift_nm, float)
    x0, y0, x1, y1 = field_nm
    where = np.column_stack([rng.uniform(x0, x1, n_beads), rng.uniform(y0, y1, n_beads),
                             rng.uniform(-200.0, 200.0, n_beads)])
    seen = rng.random((n_beads, n_frames)) >= p_missed
    for b, first, last in gaps:
        seen[b, first:last + 1] = False
    for b, start in late:
        seen[b, :start] = False
    bead, frame = np.nonzero(seen)
    order = np.argsort(frame, kind="stable")
    bead, frame = bead[order], frame[order]
    n = len(frame)
    xyz = where[bead] + drift[frame]
    noise = rng.normal(0.0, 1.0, (n, 3)) * np.array([precision_nm, precision_nm,
                                                     z_precision_nm])
    columns = {
        "frame": frame.astype(np.int64),
        "x_nm": (xyz[:, 0] + noise[:, 0]).astype(np.float32),
        "y_nm": (xyz[:, 1] + noise[:, 1]).astype(np.float32),
        "z_nm": (xyz[:, 2] + noise[:, 2]).astype(np.float32),
        "photons": rng.poisson(photons, n).astype(np.float32),
        "xy_err_nm": np.full(n, precision_nm, np.float32),
        "z_err_nm": np.full(n, z_precision_nm, np.float32),
        "bead_true": bead.astype(np.int32),
    }
    return Localizations(columns, {"units": "nm", "simulation": "smappy beads",
                                   "n_frames": n_frames, "seed": seed,
                                   "drift_truth": drift.round(3).tolist()})


def with_beads(locs: Localizations, **beads) -> Localizations:
    """A simulated table with beads added, drifting as its structure does
    (its ``drift_truth``), sorted by frame; ``bead_true`` is -1 for the
    structure's own localizations."""
    n_frames = int(locs.metadata.get("n_frames") or int(np.max(locs["frame"])) + 1)
    drift = locs.metadata.get("drift_truth")
    added = simulate_beads(n_frames, None if drift is None else np.asarray(drift),
                           **beads)
    own = Localizations({**locs.columns,
                         "bead_true": np.full(len(locs), -1, np.int32)},
                        dict(locs.metadata))
    out = concat([own, added])
    order = np.argsort(np.asarray(out["frame"]), kind="stable")
    return Localizations({k: np.asarray(v)[order] for k, v in out.columns.items()},
                         dict(locs.metadata))
