"""Simulated 4Pi fits of a long movie: what drift correction on the phase z is tested on.

A table as *Spline 4Pi* writes it, before *4Pi z*: lateral positions, the
envelope's z (``z_env_nm``, coarse, from the spot's shape) and the
interference phase (``phase_rad``, fine but known only modulo 2 pi).  The
sample drifts in x, y and z, which moves the envelope and the phase together;
the cavity -- the two arms' path difference -- wanders on its own, which
moves the phase alone.  *4Pi z* then takes the cavity's phase out window by
window and unwraps each phase on the fringe its envelope makes most likely;
a localization whose envelope is far off lands a fringe away, a ghost.

The numbers default to the 2017-11-28 tubulin movie's as fitted in P1
(docs/4pi.md §26): a 294 nm fringe period with the phase's slope negative,
envelope errors from 8 to 100 nm (median 22) scattering twice what they say,
the phase's z precision 3.4 nm and x's 5.9 nm at the median 4400 photons.
The cavity's wander, a random walk of 0.004 rad per frame on a slow swing,
is of the size *4Pi z*'s cavity moved over 250-frame windows there, a few
nm as z within a thousand frames.  The structure is filaments, straight lines
through the field at a gentle slope in z, as tubulin is.

The truth comes back beside the table: the drift per frame, the cavity per
frame, each row's true position and molecule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..locs import Localizations

TUBULIN_PHASE_PER_NM = -0.021357736947536773     # P1's pupil model, 294.2 nm period


@dataclass
class FourPiDriftTruth:
    drift_nm: np.ndarray        # (n_frames, 3)
    cavity_rad: np.ndarray      # (n_frames,)
    xyz_nm: np.ndarray          # (n, 3) each row's true position, drift included
    molecule: np.ndarray        # (n,) which molecule
    phase_per_nm: float


def fourpi_drift(n_frames: int) -> np.ndarray:
    """A sample drift of tens of nm in each axis over the movie, (n, 3):
    a swing and a creep laterally, a creep with a faster swing in z."""
    t = np.arange(n_frames) / max(n_frames, 1)
    return np.column_stack([60 * np.sin(2 * np.pi * t) + 40 * t,
                            -50 * t + 20 * np.sin(5 * t),
                            80 * t + 25 * np.sin(3 * np.pi * t)])


def cavity_wander(n_frames: int, rng, swing_rad: float = 3.0,
                  step_rad: float = 0.004) -> np.ndarray:
    """The cavity's phase per frame: a slow swing and creep plus a random
    walk of ``step_rad`` per frame."""
    t = np.arange(n_frames) / max(n_frames, 1)
    return (swing_rad * np.sin(2 * np.pi * 0.7 * t) + 2 * swing_rad * t
            + np.cumsum(rng.normal(0.0, step_rad, n_frames)))


def simulate_fourpi_drift(n_frames: int = 20000, per_frame: float = 15.0, *,
                          drift_nm: Optional[np.ndarray] = None,
                          cavity_rad: Optional[np.ndarray] = None,
                          field_nm: float = 10000.0, n_filaments: int = 60,
                          n_molecules: int = 20000, z_range_nm: float = 450.0,
                          phase_per_nm: float = TUBULIN_PHASE_PER_NM,
                          photons: float = 4400.0, xy_precision_nm: float = 5.9,
                          z_precision_nm: float = 3.4, envelope_error_nm: float = 22.0,
                          envelope_scale: float = 2.0, on_frames: float = 2.0,
                          seed: int = 0):
    """A 4Pi fit's table of a drifting filament sample, and its truth.

    ``per_frame`` localizations a frame on average, in blinks of
    ``on_frames`` frames on average.  ``drift_nm`` (n_frames, 3) and
    ``cavity_rad`` (n_frames,) default to `fourpi_drift` and
    `cavity_wander`.  Precisions scale with one over the square root of each
    localization's photons, lognormal about ``photons``; the envelope's
    error spreads lognormally on top, and the envelope scatters
    ``envelope_scale`` times its stated error -- the model's imperfection,
    which *4Pi z* measures and allows for.  Returns ``(locs, truth)``.
    """
    rng = np.random.default_rng([seed, 4])
    drift = fourpi_drift(n_frames) if drift_nm is None else np.asarray(drift_nm, float)
    cavity = (cavity_wander(n_frames, rng) if cavity_rad is None
              else np.asarray(cavity_rad, float))

    start = np.column_stack([rng.uniform(0, field_nm, (n_filaments, 2)),
                             rng.uniform(-0.8 * z_range_nm, 0.8 * z_range_nm, n_filaments)])
    angle = rng.uniform(0, np.pi, n_filaments)
    slope = rng.normal(0, 0.03, n_filaments)
    on = rng.integers(0, n_filaments, n_molecules)
    s = rng.uniform(-0.4 * field_nm, 0.4 * field_nm, n_molecules)
    molecules = start[on] + np.column_stack([s * np.cos(angle[on]), s * np.sin(angle[on]),
                                             s * slope[on]])
    molecules[:, 2] = np.clip(molecules[:, 2], -z_range_nm, z_range_nm)

    n_blinks = int(n_frames * per_frame / on_frames)
    which = rng.integers(0, n_molecules, n_blinks)
    first = rng.integers(0, n_frames, n_blinks)
    length = rng.geometric(1.0 / on_frames, n_blinks)
    owner = np.repeat(which, length)
    frame = np.repeat(first, length) + (np.arange(length.sum())
                                        - np.repeat(np.cumsum(length) - length, length))
    keep = frame < n_frames
    owner, frame = owner[keep], frame[keep]
    order = np.argsort(frame, kind="stable")
    owner, frame = owner[order], frame[order]
    n = len(frame)

    counts = rng.lognormal(np.log(photons), 0.8, n)
    scale = np.sqrt(photons / counts)
    xy_err = xy_precision_nm * scale
    z_err = z_precision_nm * scale
    env_err = envelope_error_nm * scale * rng.lognormal(0.0, 0.5, n)
    truth = molecules[owner] + drift[frame]
    x = truth[:, 0] + rng.normal(0, 1, n) * xy_err
    y = truth[:, 1] + rng.normal(0, 1, n) * xy_err
    z_env = truth[:, 2] + rng.normal(0, 1, n) * env_err * envelope_scale
    phase = np.angle(np.exp(1j * (phase_per_nm * truth[:, 2] + cavity[frame]
                                  + rng.normal(0, 1, n) * abs(phase_per_nm) * z_err)))
    f32 = np.float32
    locs = Localizations(
        {"frame": frame.astype(np.int64), "x_nm": x.astype(f32), "y_nm": y.astype(f32),
         "x_err_nm": xy_err.astype(f32), "y_err_nm": xy_err.astype(f32),
         "xy_err_nm": xy_err.astype(f32), "photons": counts.astype(f32),
         "z_env_nm": z_env.astype(f32), "z_env_err_nm": env_err.astype(f32),
         "phase_rad": phase.astype(f32),
         "phase_err_rad": (abs(phase_per_nm) * z_err).astype(f32)},
        {"units": "nm", "fourpi": {"phase_per_nm": float(phase_per_nm)}})
    return locs, FourPiDriftTruth(drift, cavity, truth, owner, float(phase_per_nm))


def apparent_drift(truth: FourPiDriftTruth, locs: Localizations) -> np.ndarray:
    """The drift a table after *4Pi z* shows, per frame, (n_frames, 3): the
    sample's, and in z also what the cavity phase *4Pi z* took out
    (``cavity_phase_rad``) missed of the true one -- the same for every
    molecule of a frame, so a drift correction on the phase z sees it as
    drift.  Frames without a localization take the sample's alone."""
    n_frames = len(truth.drift_nm)
    frame = np.asarray(locs["frame"], np.int64)
    missed = np.angle(np.exp(1j * (np.asarray(locs["cavity_phase_rad"], float)
                                   - truth.cavity_rad[frame]))) / truth.phase_per_nm
    counts = np.bincount(frame, minlength=n_frames)
    per_frame = np.bincount(frame, missed, n_frames) / np.maximum(counts, 1)
    out = truth.drift_nm.copy()
    out[:, 2] -= per_frame
    return out
