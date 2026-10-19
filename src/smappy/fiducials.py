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

**Which beads** (version 3): the beads found are judged against each
other as MBM's are (`beads.select_beads`): each bead's noise, wander and
jump about the curve the *other* beads give, a bead out where one is well
above the others', the fit repeated without it.  With every bead used the
curve is the fit above, unchanged.  A bead is named by where it is, its
median position in um (``x12.3y4.5``), not by a number: the linker numbers
beads in the order it meets them, which a filter or *min photons* changes,
and a bead chosen by hand (*by hand*, which the bead selector writes) must
stay that bead.  A choice is matched to the nearest bead within 0.3 um
(`_match`).  The beads stay in the table, corrected like everything else.

**Beads from the ROI manager** (*beads from*, `beads_in_regions`): for the
rare table where the beads were picked by hand as ROIs, each used ROI is a
bead -- its localizations brighter than *min photons*, the brightest where
a frame has several -- named after it (``ROI 5``, a number the ROI manager
never gives again).  Nothing is linked, so *maximum jump*, *memory* and
*min length* do not apply: the ROI says where the bead is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .beads import BeadSet, MadRule, RejectionSettings, parse_overrides, select_beads
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


SOURCES = ("find them", "ROI manager")


@dataclass
class FiducialSettings:
    source: str = param(SOURCES[0], label="beads from", choices=SOURCES,
                        help="find the beads as bright, long tracks, or take each "
                             "used ROI of the ROI manager as a bead")
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
    rejection: RejectionSettings = param(default_factory=RejectionSettings,
                                         label="rejection")
    beads: str = param("", label="by hand",
                       help="beads in or out whatever the rule says, by where they "
                            "are: 'x12.3y4.5 out'; Select beads... writes it")
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
    # every bead found, which were used and why (`beads.select_beads`)
    chosen: Optional[BeadSet] = field(default=None, repr=False)
    empty: List[str] = field(default_factory=list)   # ROIs that held no bead

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


def beads_in_regions(frame, photons, regions: List[np.ndarray],
                     min_photons: float = 5000.0) -> Beads:
    """A bead per region (rows of the table): its localizations brighter
    than ``min_photons``, one a frame -- the brightest, since a dye molecule
    blinking in the ROI is the dimmer one.  A region left with fewer than
    two is no bead and keeps its number with no rows."""
    frame = np.asarray(frame, np.int64)
    photons = np.asarray(photons, float)
    bead = np.full(len(frame), -1, np.int64)
    for k, rows in enumerate(regions):
        rows = np.asarray(rows, np.int64)
        rows = rows[photons[rows] >= min_photons]
        if len(rows) < 2:
            continue
        # brightest first, then the first row of each frame is kept
        rows = rows[np.lexsort((-photons[rows], frame[rows]))]
        first = np.concatenate([[True], np.diff(frame[rows]) != 0])
        bead[rows[first]] = k
    n = len(regions)
    used = bead >= 0
    count = np.bincount(bead[used], minlength=n)
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, -1)
    np.minimum.at(first, bead[used], frame[used])
    np.maximum.at(last, bead[used], frame[used])
    return Beads(bead, first, last, count)


# a bead's name, from its median position in um: x12.3y-4.5
_NAME = re.compile(r"^x(-?\d+(?:\.\d+)?)y(-?\d+(?:\.\d+)?)$")
MATCH_UM = 0.3


def bead_names(where_nm: np.ndarray) -> List[str]:
    """Each bead by where it is, (n, 2) nm -> ``x12.3y4.5`` (um); a second
    bead in the same 0.1 um gets a suffix."""
    names: List[str] = []
    for x, y in where_nm:
        # + 0.0: a bead at y = -0.04 um is y0.0, not y-0.0
        name = f"x{round(x / 1000, 1) + 0.0:.1f}y{round(y / 1000, 1) + 0.0:.1f}"
        k, base = 2, name
        while name in names:
            name, k = f"{base}#{k}", k + 1
        names.append(name)
    return names


def _match(overrides: Dict[str, bool], names: List[str], where_nm: np.ndarray
           ) -> Dict[str, bool]:
    """Choices by hand, onto the beads found now: by name, or else to the
    nearest bead within `MATCH_UM` of the place the name says."""
    out: Dict[str, bool] = {}
    for key, use in overrides.items():
        if key in names:
            out[key] = use
            continue
        m = _NAME.match(key.split("#")[0])
        if m is None or not len(where_nm):
            continue
        d = np.hypot(where_nm[:, 0] / 1000 - float(m.group(1)),
                     where_nm[:, 1] / 1000 - float(m.group(2)))
        if d.min() <= MATCH_UM:
            out[names[int(np.argmin(d))]] = use
    return out


@dataclass
class FoundBeads:
    """What `find_and_choose` hands back: the beads found and judged, and
    the arrays the fit was made from."""
    beads: Beads
    chosen: BeadSet
    where_nm: np.ndarray
    frame: np.ndarray
    positions: np.ndarray
    n_frames: int
    axes: tuple
    empty: List[str] = field(default_factory=list)   # the ROIs that held no bead


def find_and_choose(locs: Localizations, settings: Optional[FiducialSettings] = None,
                    select=None, regions=None) -> FoundBeads:
    """Find the beads (`find_beads`) and choose among them
    (`beads.select_beads`, on frames, over the whole file).  ``regions``,
    ``[(name, rows)]``: a bead per region instead (`beads_in_regions`)."""
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
    if regions is not None:
        beads = beads_in_regions(frame, locs["photons"], [r for _, r in regions],
                                 settings.min_photons)
        empty = [name for (name, _), n in zip(regions, beads.count) if n < 2]
        keep = [k for k in range(len(regions)) if beads.count[k] >= 2]
        if not keep:
            raise ValueError(
                f"no beads: none of the {len(regions)} ROIs holds two "
                f"localizations brighter than {settings.min_photons:g} photons")
        # renumbered over the ROIs that hold a bead
        renumber = np.full(len(regions), -1, np.int64)
        renumber[keep] = np.arange(len(keep))
        b = np.where(beads.bead >= 0, renumber[np.maximum(beads.bead, 0)], -1)
        beads = Beads(b, beads.first[keep], beads.last[keep], beads.count[keep])
        names = [regions[k][0] for k in keep]
    else:
        empty = []
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
        names = None
    positions = np.column_stack([x, y] + ([z] if use_z else []))
    rows = beads.bead >= 0
    where = np.array([[np.median(x[beads.bead == b]), np.median(y[beads.bead == b])]
                      for b in range(len(beads))])
    names = names or bead_names(where)
    r = settings.rejection
    # the fiducial fit's own span and coefficients, so that with every bead
    # used the curve is `fit_drift`'s to the last bit
    chosen = select_beads(frame[rows], positions[rows], beads.bead[rows] + 1, names,
                          knot=settings.spline_knot_frames,
                          rule=MadRule(k=r.k, margin_nm=r.margin_nm),
                          overrides=_match(parse_overrides(settings.beads), names, where),
                          min_beads=min(r.min_beads, len(beads)),
                          span=(0.0, max(n_frames - 1.0, 1.0)),
                          n_coeffs=max(4, int(round(n_frames / settings.spline_knot_frames))
                                       + 3))
    return FoundBeads(beads, chosen, where, frame, positions, n_frames,
                      ("x", "y", "z") if use_z else ("x", "y"), empty)


def estimate_drift_fiducials(locs: Localizations,
                             settings: Optional[FiducialSettings] = None,
                             select=None, regions=None
                             ) -> "tuple[Drift, Beads, FiducialFit]":
    """The drift of a table from its beads: the `Drift`, the beads found and
    the fit of the beads used (``fit.chosen``: all of them, judged).
    ``regions``: the beads are these (`find_and_choose`)."""
    settings = settings or FiducialSettings()
    found = find_and_choose(locs, settings, select, regions)
    chosen = found.chosen
    n_axes = found.positions.shape[1]
    curve = chosen.curve(np.arange(found.n_frames))
    drift_nm = np.zeros((found.n_frames, 3))
    drift_nm[:, :n_axes] = curve
    bead = found.beads.bead
    rows = np.flatnonzero((bead >= 0) & chosen.used[np.maximum(bead, 0)])
    # `select_beads` kept the bead rows in input order: its residuals line up
    every = np.flatnonzero(bead >= 0)
    keep = np.isin(every, rows)
    _, b = np.unique(bead[rows], return_inverse=True)
    used_beads = np.flatnonzero(chosen.used)
    fit = FiducialFit(drift_nm, chosen.offsets_nm[used_beads],
                      chosen.residuals[keep], rows, b, chosen, found.empty)
    drift = Drift(drift_nm, settings, n_used=len(rows),
                  method="fiducials").referenced(settings.reference)
    return drift, found.beads, fit


def describe(drift: Drift, beads: Beads, fit: FiducialFit) -> str:
    """The text of a run: the curve, the beads, their scatter about it."""
    rms = ", ".join(f"{a} {v:.1f}" for a, v in zip("xyz", fit.rms_nm))
    left = ""
    if fit.chosen is not None and not fit.chosen.used.all():
        out = int((~fit.chosen.used).sum())
        left = f" ({out} left out)"
    return (f"{drift}; {len(beads)} bead(s){left}, {len(fit.rows)} localizations, "
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
