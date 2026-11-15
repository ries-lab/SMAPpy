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
*min length* do not apply: the ROI says where the bead is.  From an *id
column* (`column_markers`) each id (-1 is none) is a marker the same way --
`bead_id` from a run before, `track_id` -- among the localizations the
selection keeps, named ``"cluster_id 17"``.

**Structures** (*markers* = structures; Picasso's *Undrift from picked*):
a DNA origami, or one of its binding sites, is a fiducial too -- it sits
still, and where it appears to move the sample moved -- but it shows itself
only when an imager strand binds, a few frames every few hundred, dimly,
and at a different site of the structure each time.  The brightest point a
frame above *min photons* throws nearly all of that away, so a structure is
every localization of its region (`structures_in_regions`): no photon cut,
several a frame, each weighted by its precision and the structures' extent
(`marker_weights`).  They are named, never found -- *find them* links
bright, long tracks, which no DNA-PAINT site makes -- so they come from the
ROI manager (whole origami, from *Segment regions*' *like examples*: the
first round Picasso recommends) or an id column (the binding sites of *Find
clusters*' *local maxima*, after the first round has sharpened them: the
second).  The fit is the beads' unchanged, which is the better model for
markers this sparse: a frame needs no localization, and a structure whose
sites bind late lands where it belongs.

*min length* means nothing here and is not used: a binding site is seen
in a few percent of the frames by design, and what a short-lived marker
costs is its own offset, nothing more -- the weighted fit gives it what
its points are worth.  The judging (`beads.select_beads`) is the beads',
each structure against the others' noise, wander and jump, with hundreds
judged against the curve of all (`beads.LEAVE_ONE_OUT`).  It matters
there: of the 429 binding sites *Find clusters*' local maxima named in the
page's figure, 115 were left out, for a jump or a wander -- nine in ten
of them clusters that took localizations of the site next to it (under 95 %
from one site, against a quarter of those kept), so that they move between
the two as each binds.  With every site used the second round was good to
0.75 nm rms, with them left out to 0.2.

**Where it departs from Picasso** (`postprocess.undrift_from_fiducials`,
`_undrift_from_picked_coordinate`; Picasso is MIT licensed, Jungmann lab):
Picasso takes, per frame, each pick's localization less the pick's mean
position (one a frame: where it has several, the last), averages the
picks with weights 1 / (each pick's mean squared deviation from that
average) and interpolates linearly over the frames with none.  Here every
marker's offset is fitted with one spline through all of them, by least
squares on every localization: a frame where no marker blinked is carried
by its neighbours rather than interpolated afterwards, a pick's mean
position is not taken as its place (it holds the mean drift over the
pick's own frames, which differs between a pick seen early and one seen
late), and the weight is per localization, not per pick (`marker_weights`
says why).  The curve is smoother than Picasso's frame by frame -- one value
per *knot spacing* -- so a knot spacing of a few hundred frames suits
sparse markers (`STRUCTURE_KNOT`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .beads import (LEAVE_ONE_OUT, BeadSet, MadRule, RejectionSettings, parse_overrides,
                    select_beads)
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


SOURCES = ("find them", "ROI manager", "id column")
# the knot spacing for beads, seen in every frame, and what *markers* =
# structures sets in its place (`FiducialDrift.react`): 85 origami of the
# Simulate preset (10 um, seed 0), about 3 localizations a frame between
# them, gave 3.8 nm rms at 100 frames, 1.5 at 500, 1.0 at 1000 -- on a
# smooth drift; a faster one wants the smallest spacing the markers allow
BEAD_KNOT = 100
STRUCTURE_KNOT = 500

def _bead_ids(bead: np.ndarray) -> np.ndarray:
    """Beads numbered from 0 (-1 for none) as the bead ids written to the
    table, from 1 (-1 still none)."""
    bead = np.asarray(bead, np.int64)
    return np.where(bead >= 0, bead + 1, -1)

#: what a marker is: a bead, one bright point a frame, or a structure (a DNA
#: origami, a binding site), every localization of its region
MARKERS = ("beads", "structures")
#: the id attributes offered before a table is open; `describe` offers the
#: table's own integer attributes
ID_COLUMNS = ("cluster_id", "bead_id", "track_id")


@dataclass
class FiducialSettings:
    markers: str = param(MARKERS[0], label="markers", choices=MARKERS,
                         help="beads: the brightest localization a frame, above min "
                              "photons; structures (DNA origami, binding sites): "
                              "every localization, weighted by its precision")
    source: str = param(SOURCES[0], label="markers from", choices=SOURCES,
                        help="find the beads as bright, long tracks; take each "
                             "used ROI of the ROI manager as a marker; or each id "
                             "of an attribute")
    id_column: str = param(ID_COLUMNS[0], label="id attribute",
                           choices=lambda: [(c, c) for c in ID_COLUMNS],
                           help="each id (-1 is none) is one marker: cluster_id "
                                "from Find clusters, bead_id from a run before")
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
    spline_knot_frames: int = param(BEAD_KNOT, label="knot spacing", unit="frames",
                                    min=5,
                                    help="frames between the spline's "
                                         "coefficients")
    rejection: RejectionSettings = param(default_factory=RejectionSettings,
                                         label="rejection")
    beads: str = param("", label="by hand",
                       help="markers in or out whatever the rule says, by name: "
                            "'x12.3y4.5 out', 'ROI 5 in'; Select beads... writes it")
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


def structures_in_regions(frame, regions: List[np.ndarray]) -> Beads:
    """A marker per region: every one of its rows, several a frame, no
    photon cut -- a DNA origami's binding sites blink one at a time and
    dimly, and each localization is a point of the structure.  A row in two
    regions is the later one's; a region with fewer than two rows is no
    marker and keeps its number with none."""
    frame = np.asarray(frame, np.int64)
    marker = np.full(len(frame), -1, np.int64)
    for k, rows in enumerate(regions):
        rows = np.asarray(rows, np.int64)
        if len(rows) >= 2:
            marker[rows] = k
    n = len(regions)
    used = marker >= 0
    count = np.bincount(marker[used], minlength=n)
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, -1)
    np.minimum.at(first, marker[used], frame[used])
    np.maximum.at(last, marker[used], frame[used])
    return Beads(marker, first, last, count)


def column_markers(locs: Localizations, column: str, select=None
                   ) -> "List[tuple]":
    """Each id of ``column`` (-1, `ids.NONE`, is none) as a marker, ``[(name, rows)]``, its
    rows those ``select`` (a mask) keeps.  Named ``"cluster_id 17"``: the id
    is what the user sees of it -- in its region table, as an ROI made from
    it -- and binding sites 20 nm apart would share a name by position
    (`bead_names`, 0.1 um)."""
    from .region_runs import split_ids
    from .tracking.columns import integer_columns
    if column not in locs:
        have = integer_columns(locs)
        raise ValueError(f"markers from an id column, and the table has no "
                         f"{column!r}: choose one of "
                         + (", ".join(have) if have else "(no integer attribute)")
                         + "; Find clusters (local maxima) names binding sites")
    ids = np.asarray(locs[column])
    if select is not None:
        ids = np.where(np.asarray(select, bool), ids, -1)    # -1: in none
    values, rows = split_ids(ids)
    if not len(values):
        raise ValueError(f"no markers: {column} is -1 (none) on every localization "
                         f"of the selection")
    return [(f"{column} {int(v)}", r) for v, r in zip(values, rows)]


def marker_errors(locs: Localizations, use_z: bool) -> Optional[np.ndarray]:
    """Each localization's precision per axis, (n, 2 or 3) nm: `xy_err_nm`
    for x and y, `z_err_nm` for z (NaN where the table has none); None when
    there is no `xy_err_nm`, and then every point weighs the same."""
    if "xy_err_nm" not in locs:
        return None
    xy = np.asarray(locs["xy_err_nm"], float)
    columns = [xy, xy]
    if use_z:
        columns.append(np.asarray(locs["z_err_nm"], float) if "z_err_nm" in locs
                       else np.full(len(xy), np.nan))
    return np.column_stack(columns)


def marker_weights(frame, positions, errors, marker, n_frames: int,
                   knot_frames: int) -> np.ndarray:
    """The weight of each point of a structure, per axis, (n, n_axes):

        w_i = 1 / (sigma_i^2 + s^2),

    ``sigma_i`` its precision (``errors``, NaN where unknown) and ``s`` the
    markers' own extent.  A point of a structure scatters about its marker's
    offset by its precision *and* by where on the structure it sits: a
    binding site of an origami 60 nm wide is up to 30 nm from the centre
    whatever its photons.  So the points of whole origami weigh nearly
    alike, and those of single binding sites (``s`` near 0) by their
    precision -- each of Picasso's two rounds gets the weights it deserves.
    ``s^2`` is, over the markers, the median of the mean squared residual
    of a first fit (weighted by precision) less the mean ``sigma^2``; one
    value for x and y together, one for z.

    One extent for every marker, not each its own (Picasso's 1 / MSD per
    pick): on the Simulate preset's origami (six seeds, 85 origami in 10 um,
    knots every 500 frames), each marker's own extent gave 1.9 nm rms against the
    truth, the median 1.7 -- as good as no weights at all; precision alone,
    2.0.  An origami's extent is mostly how many of its sites were labelled
    and where a few stray localizations fell, and down-weighting a whole
    origami for it threw its information away.  On the binding sites of the
    second round, precision weights cut the error of none by a third (0.21 nm
    against 0.33), and the common extent keeps that.

    A precision below a third of the median is taken as that third, so a
    broken fit's 0.01 nm cannot carry the curve; an unknown one (NaN, or no
    such column, ``errors`` None) is the median, or 1 nm where nothing is
    known, so an axis without precisions is weighed nearly alike.
    """
    from .beads import fit_offsets_and_spline
    frame = np.asarray(frame, np.int64)
    p = np.asarray(positions, float)
    marker = np.asarray(marker, np.int64)
    n, n_axes = p.shape
    if errors is None:
        errors = np.full((n, n_axes), np.nan)
    sigma2 = np.empty((n, n_axes))
    for a in range(n_axes):
        e = np.asarray(errors[:, a], float)
        good = np.isfinite(e) & (e > 0)
        median = float(np.median(e[good])) if good.any() else 1.0
        sigma2[:, a] = np.maximum(np.where(good, e, median), median / 3.0) ** 2
    _, index = np.unique(marker, return_inverse=True)
    n_markers = int(index.max()) + 1
    hi = max(n_frames - 1.0, 1.0)
    n_coeffs = max(4, int(round(n_frames / knot_frames)) + 3)
    offsets, curve = fit_offsets_and_spline(frame, p, index, n_markers, 0.0, hi,
                                            n_coeffs, reference=0.0,
                                            weights=1.0 / sigma2)
    excess = (p - offsets[index] - curve(frame)) ** 2 - sigma2
    count = np.maximum(np.bincount(index, minlength=n_markers), 1)
    per_marker = np.column_stack([np.bincount(index, excess[:, a], n_markers) / count
                                  for a in range(n_axes)])
    # x and y one extent: two estimates of one width average to a steadier one
    per_marker[:, :2] = per_marker[:, :2].mean(axis=1, keepdims=True)
    extent2 = np.maximum(np.median(per_marker, axis=0), 0.0)
    return 1.0 / (sigma2 + extent2)


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
    """Find the markers and choose among them (`beads.select_beads`, on
    frames, over the whole file).  Beads are found as tracks (`find_beads`);
    ``regions``, ``[(name, rows)]``, are a marker each instead
    (`beads_in_regions`, `structures_in_regions`), and with *markers from*
    an id column they are its ids (`column_markers`, among ``select``)."""
    settings = settings or FiducialSettings()
    structures = settings.markers == MARKERS[1]
    need = ("x_nm", "y_nm", "frame") + (() if structures else ("photons",))
    missing = [c for c in need if c not in locs]
    if missing:
        raise ValueError(f"finding markers needs {', '.join(missing)}: the table "
                         f"has {', '.join(sorted(locs.keys()))}")
    if regions is None and settings.source == SOURCES[2]:
        regions = column_markers(locs, settings.id_column, select)
    if structures and regions is None:
        raise ValueError(
            "structures are not found, they are named: take them from the ROI "
            "manager (Segment regions' like examples picks origami) or from an id "
            "column (Find clusters' local maxima names binding sites) in markers from")
    use_z = settings.use_z
    if use_z is None:
        use_z = "z_nm" in locs
    elif use_z and "z_nm" not in locs:
        raise ValueError("correct z is on but the table has no z_nm attribute")
    frame = np.asarray(locs["frame"], np.int64)
    n_frames = int(frame.max()) + 1 if len(frame) else 0
    x = np.asarray(locs["x_nm"], float)
    y = np.asarray(locs["y_nm"], float)
    z = np.asarray(locs["z_nm"], float) if use_z else None
    if regions is not None:
        rows_of = [r for _, r in regions]
        if structures:
            beads = structures_in_regions(frame, rows_of)
            why = "two localizations"
        else:
            beads = beads_in_regions(frame, locs["photons"], rows_of,
                                     settings.min_photons)
            why = f"two localizations brighter than {settings.min_photons:g} photons"
        empty = [name for (name, _), n in zip(regions, beads.count) if n < 2]
        keep = [k for k in range(len(regions)) if beads.count[k] >= 2]
        if not keep:
            raise ValueError(f"no markers: none of the {len(regions)} regions "
                             f"holds {why}")
        # renumbered over the regions that hold a marker
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
    where = _medians(x, y, beads.bead, len(beads))
    names = names or bead_names(where)
    weights = None
    if structures:
        weights = marker_weights(frame[rows], positions[rows],
                                 _rows_of(marker_errors(locs, use_z), rows),
                                 beads.bead[rows], n_frames, settings.spline_knot_frames)
    r = settings.rejection
    # the fiducial fit's own span and coefficients, so that with every bead
    # used the curve is `fit_drift`'s to the last bit
    chosen = select_beads(frame[rows], positions[rows], _bead_ids(beads.bead[rows]), names,
                          knot=settings.spline_knot_frames,
                          rule=MadRule(k=r.k, margin_nm=r.margin_nm),
                          overrides=_match(parse_overrides(settings.beads), names, where),
                          min_beads=min(r.min_beads, len(beads)),
                          span=(0.0, max(n_frames - 1.0, 1.0)),
                          n_coeffs=max(4, int(round(n_frames / settings.spline_knot_frames))
                                       + 3),
                          weights=weights,
                          leave_one_out=LEAVE_ONE_OUT if structures else None)
    return FoundBeads(beads, chosen, where, frame, positions, n_frames,
                      ("x", "y", "z") if use_z else ("x", "y"), empty)


def _rows_of(a: Optional[np.ndarray], rows) -> Optional[np.ndarray]:
    return None if a is None else a[rows]


def _medians(x, y, marker, n: int) -> np.ndarray:
    """Each marker's median position, (n, 2) nm: one sort, not a pass over
    the table per marker -- a thousand binding sites made that a second."""
    rows = np.flatnonzero(marker >= 0)
    order = rows[np.argsort(marker[rows], kind="stable")]
    cuts = np.searchsorted(marker[order], np.arange(n + 1))
    return np.array([[np.median(x[order[a:b]]), np.median(y[order[a:b]])]
                     if b > a else [np.nan, np.nan]
                     for a, b in zip(cuts[:-1], cuts[1:])]).reshape(n, 2)


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


def describe(drift: Drift, beads: Beads, fit: FiducialFit,
             markers: str = MARKERS[0]) -> str:
    """The text of a run: the curve, the markers, their scatter about it --
    for structures their extent and precision together, not a precision."""
    rms = ", ".join(f"{a} {v:.1f}" for a, v in zip("xyz", fit.rms_nm))
    left = ""
    if fit.chosen is not None and not fit.chosen.used.all():
        out = int((~fit.chosen.used).sum())
        left = f" ({out} left out)"
    noun = "structure(s)" if markers == MARKERS[1] else "bead(s)"
    return (f"{drift}; {len(beads)} {noun}{left}, {len(fit.rows)} localizations, "
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
