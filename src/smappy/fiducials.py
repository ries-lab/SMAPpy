"""Drift from fiducial markers: the tracks of beads, one spline through all.

A bead (a fluorescent microsphere, a gold particle) sits on the coverslip
and is bright in every frame, so where it appears to move, the sample moved.
Its track is the drift, measured directly, with no structure to correlate:
what COMET and RCC infer from the overlap of blinking molecules, a bead
records frame by frame at a precision of a few nanometres.

**Beads** are found as tracks (`find_beads`): the localizations brighter
than *min photons* are linked with *Link tracks*' simple linker
(`tracking.link.link_tracks`), and a track that lasts at least *min length*
(a fraction of the frames) is a bead.  Brightness first, because it is what
tells a bead from a dye, and because it leaves few localizations to link: a
generous memory (50 frames) then costs nothing, where over a dense table of
blinks it would give a candidate pair per neighbour per frame.  The memory
bridges frames where a bead was missed or fitted badly; a longer gap ends
the track and the rest is another one, which costs nothing either, below.

**The fit** (`fit_drift`) is linear.  Localization ``i`` of bead ``b`` at
frame ``t`` is at

    x_i = o_b + d(t_i) + noise,

``o_b`` the bead's own position (its offset) and ``d`` the drift, a cubic
B-spline in frames with a coefficient every *knot spacing* frames -- the
representation of COMET's spline fit (`drift._spline_basis`), so that the
two curves are the same kind of thing.  Every bead's offset and the spline's
coefficients are fitted together in one least-squares problem.  So:

* a bead is placed by its offset, *fitted*, not by its first point (which
  carries its whole precision into the curve) nor by its mean (SMAP's
  ``driftcorrection_beads``): the mean of a bead that starts late is the
  mean of the drift over *its* frames, not over all of them, and subtracting
  it shifts its piece of the curve by the difference;
* a gap is no hole: the beads seen at that time carry the curve, and a bead
  that lands on the glass halfway through the measurement adds to the second
  half without being lined up with anything by hand;
* the problem has one degeneracy -- a constant can move from every offset
  into the curve -- which the *reference* fixes: the drift is zero at the
  first frame, or the last with *reference* (`Drift.referenced`).

A small penalty on the coefficients' second differences (1e-3 of the mean
data weight per coefficient, `_PENALTY`) keeps the system solvable where no
bead was seen at all -- the curve then runs straight across -- and where
beads were seen it is a slight smoothing, far below their precision.

**Kept simple for now, on purpose** (the design, wave 5 step 5c): every
bead that passes is used, all with equal weight; rejecting a bead that
disagrees with the others, or a robust fit, comes later.  The beads stay in
the table, corrected like everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .drift import REFERENCES, Drift
from .locs import Localizations
from .params import param

# the second-difference penalty, relative to the mean data weight per
# coefficient: enough to bridge a stretch no bead covers, little enough not
# to bend a curve the beads do cover.  Measured on the page's simulation
# (five 3 nm beads, 8000 frames, knots every 100): it moves the curve by
# 0.3 nm rms (0.7 at most) against no penalty, and brings it slightly
# closer to the truth (2.3 against 2.4 nm rms in x).
_PENALTY = 1e-3


@dataclass
class FiducialSettings:
    min_photons: float = param(5000.0, label="min photons", min=0,
                               help="localizations dimmer than this are not "
                                    "beads")
    min_fraction: float = param(0.25, label="min length", min=0.0, max=1.0,
                                help="a bead is seen in at least this "
                                     "fraction of the frames")
    max_jump_nm: float = param(150.0, label="maximum jump", unit="nm", min=1.0,
                               help="the farthest a bead moves between the "
                                    "frames it is seen in",
                               advanced=True)
    memory: int = param(50, label="memory", unit="frames", min=0,
                        help="frames a bead may be missing from its track",
                        advanced=True)
    spline_knot_frames: int = param(100, label="knot spacing", unit="frames",
                                    min=5,
                                    help="frames between the spline's "
                                         "coefficients")
    use_z: Optional[bool] = param(None, label="correct z",
                                  help="auto: if the table has z")
    # `Drift.referenced`; "as estimated" is the fit's own gauge, d(0) = 0
    reference: str = param("as estimated", label="reference",
                           choices=REFERENCES,
                           help="the frame the drift is zero at; first or "
                                "last frame for consecutive files",
                           advanced=True)


@dataclass
class Beads:
    """The beads found: a bead number per localization (-1: none) and, per
    bead, its first and last frame and its localization count."""
    bead: np.ndarray
    first: np.ndarray
    last: np.ndarray
    count: np.ndarray

    def __len__(self) -> int:
        return len(self.count)


def find_beads(x, y, frame, photons, z=None, *, n_frames: Optional[int] = None,
               min_photons: float = 5000.0, min_fraction: float = 0.25,
               max_jump_nm: float = 150.0, memory: int = 50,
               select=None) -> Beads:
    """The bright, long-lived tracks.  ``select``, a mask, is what may be a
    bead (the layer's filter); ``n_frames`` is what *min length* is a
    fraction of (default: the last frame + 1)."""
    from .tracking.link import link_tracks
    frame = np.asarray(frame, np.int64)
    n = len(frame)
    if n_frames is None:
        n_frames = int(frame.max()) + 1 if n else 0
    bright = np.asarray(photons, float) >= min_photons
    if select is not None:
        bright &= np.asarray(select, bool)
    groups = np.where(bright, 0, -1)
    min_length = max(2, int(np.ceil(min_fraction * n_frames)))
    track, _ = link_tracks(x, y, frame, z, max_jump_nm=max_jump_nm,
                           memory=memory, min_length=min_length,
                           groups=groups)
    # the minimum is of localizations: a track has at most one a frame, so
    # it is the frames the bead was *seen* in, its gaps not counted
    bead = np.where(track > 0, track - 1, -1)
    n_beads = int(track.max(initial=0))
    used = bead >= 0
    count = np.bincount(bead[used], minlength=n_beads)
    first = np.full(n_beads, np.iinfo(np.int64).max)
    last = np.full(n_beads, -1)
    np.minimum.at(first, bead[used], frame[used])
    np.maximum.at(last, bead[used], frame[used])
    return Beads(bead, first, last, count)


@dataclass
class FiducialFit:
    """The drift and what it was fitted from: per bead its offset (nm, one
    column per axis) and the residuals of its localizations."""
    drift_nm: np.ndarray          # (n_frames, 3)
    offsets_nm: np.ndarray        # (n_beads, n_axes)
    residuals_nm: np.ndarray      # (n_used, n_axes)
    rows: np.ndarray              # the localizations used, indices into the input
    bead: np.ndarray              # their bead number

    @property
    def rms_nm(self) -> np.ndarray:
        """The scatter of the beads about the curve, per axis."""
        if not len(self.residuals_nm):
            return np.full(self.residuals_nm.shape[1], np.nan)
        return np.sqrt(np.mean(self.residuals_nm ** 2, axis=0))


def fit_drift(frame, positions, bead, n_frames: int,
              knot_frames: int = 100) -> FiducialFit:
    """One spline through every bead, each bead at its own fitted offset.

    ``positions`` is (n, n_axes) nm, ``bead`` a bead number per row (-1: not
    a bead).  The drift is zero at frame 0.  The fit itself is
    `beads.fit_offsets_and_spline`, on frames here and in seconds for MBM.
    """
    from .beads import fit_offsets_and_spline

    frame = np.asarray(frame, np.int64)
    positions = np.asarray(positions, float)
    if positions.ndim == 1:
        positions = positions[:, None]
    bead = np.asarray(bead, np.int64)
    rows = np.flatnonzero(bead >= 0)
    if not len(rows):
        raise ValueError("no beads to fit the drift to")
    _, b = np.unique(bead[rows], return_inverse=True)
    n_beads = int(b.max()) + 1
    t = frame[rows]
    p = positions[rows]
    n_axes = p.shape[1]
    n_coefficients = max(4, int(round(n_frames / knot_frames)) + 3)
    hi = max(n_frames - 1.0, 1.0)
    offsets_nm, spline = fit_offsets_and_spline(t, p, b, n_beads, 0.0, hi,
                                                n_coefficients, reference=0.0,
                                                penalty=_PENALTY)
    curve = spline(np.arange(n_frames))
    drift = np.zeros((n_frames, 3))
    drift[:, :n_axes] = curve
    residuals = p - offsets_nm[b] - curve[t]
    return FiducialFit(drift, offsets_nm, residuals, rows, b)


def estimate_drift_fiducials(locs: Localizations,
                             settings: Optional[FiducialSettings] = None,
                             select=None) -> "tuple[Drift, Beads, FiducialFit]":
    """The drift of a table from its beads: the `Drift`, the beads found and
    the fit.  ``select`` (a mask) is what may be a bead."""
    settings = settings or FiducialSettings()
    missing = [c for c in ("x_nm", "y_nm", "frame", "photons") if c not in locs]
    if missing:
        raise ValueError(f"finding beads needs {', '.join(missing)}: the table "
                         f"has {', '.join(sorted(locs.keys()))}")
    use_z = settings.use_z
    if use_z is None:
        use_z = "z_nm" in locs
    elif use_z and "z_nm" not in locs:
        raise ValueError("correct z is on but the table has no z_nm column")
    frame = np.asarray(locs["frame"], np.int64)
    n_frames = int(frame.max()) + 1 if len(frame) else 0
    x = np.asarray(locs["x_nm"], float)
    y = np.asarray(locs["y_nm"], float)
    z = np.asarray(locs["z_nm"], float) if use_z else None
    beads = find_beads(x, y, frame, locs["photons"], z, n_frames=n_frames,
                       min_photons=settings.min_photons,
                       min_fraction=settings.min_fraction,
                       max_jump_nm=settings.max_jump_nm, memory=settings.memory,
                       select=select)
    if not len(beads):
        raise ValueError(
            f"no beads: no track of localizations brighter than "
            f"{settings.min_photons:g} photons lasts {settings.min_fraction:g} "
            f"of the {n_frames} frames; lower min photons or min length")
    positions = np.column_stack([x, y] + ([z] if use_z else []))
    fit = fit_drift(frame, positions, beads.bead, n_frames,
                    settings.spline_knot_frames)
    drift = Drift(fit.drift_nm, settings, n_used=len(fit.rows),
                  method="fiducials").referenced(settings.reference)
    return drift, beads, fit


def describe(drift: Drift, beads: Beads, fit: FiducialFit) -> str:
    """The text of a run: the curve, the beads, their scatter about it."""
    rms = ", ".join(f"{a} {v:.1f}" for a, v in zip("xyz", fit.rms_nm))
    return (f"{drift}; {len(beads)} bead(s), {len(fit.rows)} localizations, "
            f"scatter about the curve {rms} nm")


def draw_beads(figure, frame, residuals, bead, drift_nm) -> None:
    """Each bead's positions less its offset, over the curve, per axis."""
    n_axes = residuals.shape[1]
    axes = np.atleast_1d(figure.subplots(n_axes, 1, sharex=True))
    frames = np.arange(len(drift_nm))
    for axis, ax in enumerate(axes):
        values = residuals[:, axis] + drift_nm[frame, axis]
        for b in np.unique(bead):
            mine = bead == b
            ax.plot(frame[mine], values[mine], ".", ms=1.5, alpha=0.5)
        ax.plot(frames, drift_nm[:, axis], color="k", lw=1)
        ax.set_ylabel(f"{'xyz'[axis]} (nm)")
    axes[-1].set_xlabel("frame")
    axes[0].set_title(f"{len(np.unique(bead))} beads, each less its offset; "
                      f"the curve in black", fontsize=9)
