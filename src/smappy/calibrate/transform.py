"""A channel transformation measured from localizations, not from beads.

The 2D two-colour workflow has no PSF model to calibrate, only a registration:
where on the chip the second channel puts the same molecule.  In a ratiometric
experiment that registration can be read off the data itself, because every
molecule is *already* imaged twice in the same frame.  Fit the whole frame
with a plain Gaussian, ignoring the split, and each emitter appears as a pair
of localizations in the same frame a constant offset apart; this module finds
that offset, pairs the localizations frame by frame, and fits the projective
map `smappy.dualfit` needs.

The shape of it follows SMAP's ``RegisterLocs2``: a global alignment that owes
nothing to an initial guess, then repeated pair-and-refit at shrinking
tolerance.  The global step differs.  ``RegisterLocs2`` renders both channels
into 500 nm histograms and cross-correlates them; here the same question is
asked of the point pairs directly:

    for every pair of localizations in a frame, the vector between them; the
    offset between the channels is the one that shows up over and over.

That is a cross-correlation, computed on pairs rather than on rendered images,
and it costs nothing in accuracy.  What it buys is that **no part of it
assumes where the split is**.  The offset is found first and the split
position follows from it, so a splitter whose boundary sits nowhere near the
middle of the chip needs no initial shift, no magnification guess, and no
correctly-set image centre -- which is the setting that most often makes a
registration silently fail.

A mirrored layout is the same argument with one sign changed.  Reflected about
a line at ``m``, partners satisfy ``a + b = 2m - 1`` rather than ``b - a = d``,
so the vote is taken on the *sum* along the mirrored axis.  The peak is then
the mirror line itself, which is the split position, measured rather than
assumed.  Voting in all three feature spaces and keeping the sharpest peak is
what lets the layout be detected instead of declared.

Coordinates are chip pixels throughout, as in `smappy.calibrate.dual`, and so
is the ``split_position`` written into the geometry -- `dualfit.which_channel`
compares it against chip coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple

import json
import os
import tempfile
import warnings

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .dual import (LAYOUTS, ChannelTransform, Polynomial, fit_dual_transform,
                   map_points, n_polynomial_terms, point_coverage_area,
                   polynomial_terms)

FORMAT = 'smappy-channel-transform'
VERSION = 2
READABLE = (1, 2)      # version 1 had no model but the projective one

# The three feature spaces the vote is taken in.  'none' is the plain
# difference and serves both unmirrored layouts -- it cannot tell them apart,
# and does not need to: the split axis is the one the offset is large along.
MODES = ('none', 'x', 'y')
MODE_LAYOUTS = {'none': ('right-left', 'up-down'),
                'x': ('right-left mirrored',), 'y': ('up-down mirrored',)}
# Bins around the origin excluded from the unmirrored vote.  Two molecules
# close together in the same half also produce a short vector; the channel
# offset is half a chip away and never anywhere near here.
ORIGIN_GUARD_BINS = 3
# How much room the tight round leaves around the misfit the coarse round
# already shows.  A projective map cannot represent field distortion, so where
# it misfits, the true partners sit further apart than any fixed tolerance
# would allow -- and it misfits most at the edges of the field, which is
# exactly where pairs are scarcest and most worth keeping.
PAIR_MARGIN = 3.0
# The residual percentile the margin is taken of: high enough to speak for the
# periphery rather than the well-fitted middle, not the outright maximum,
# which one mismatched pair would set.
PAIR_PERCENTILE = 98.0

# Order three, and no option for two: radial distortion is cubic in the
# coordinates (see `Polynomial`), so a quadratic cannot describe it at all.
POLYNOMIAL_ORDER = 3
# Twenty coefficients fitted on two hundred pairs.  A polynomial follows its
# data and then leaves, so the guard is not really about the count -- it is
# about not handing that many degrees of freedom to a handful of points.
MIN_PAIRS_PER_COEFFICIENT = 10
# ...and this is the guard that matters.  Outside the region the pairs cover,
# a cubic is an extrapolation and a violent one: measured on a clean field,
# pairs confined to the middle give a cubic that is eighty times worse than
# the projective map outside them.  Below this fraction of the field, the
# projective fit is kept and the reason said out loud.
MIN_POLYNOMIAL_COVERAGE = 0.5
# The coarse round only gets to widen the tight one if its own pair set is
# trustworthy.  A contaminated set -- chance partners, which at SMLM densities
# and a twenty-pixel tolerance are plentiful -- has a large residual *because*
# of the contamination, and widening on the strength of that lets more of it
# in.  The fraction of matched pairs the coarse fit accepts separates the two
# cleanly: a distorted field keeps nearly all of them, a contaminated one a
# third.
MIN_CLEAN_FRACTION = 0.6


@dataclass
class RegisterSettings:
    """How hard to look, and what is already known.

    Everything has a default that works on a ratiometric splitter; ``layout``
    and ``split_position`` are the two worth setting by hand, and only when
    the automatic answer is wrong.
    """
    layout: str = 'auto'
    main_channel: str = 'auto'
    # 'projective' is eight coefficients and extrapolates gracefully;
    # 'polynomial' is twenty and describes a distorted field far better where
    # the pairs reach, and not at all where they do not.  See `Polynomial`.
    model: str = 'projective'
    split_position: Optional[int] = None
    # the vote
    vote_bin_px: float = 2.0
    vote_smooth_px: float = 4.0
    # Pair and refit twice, as in SMAP's ``RegisterLocs2``: coarse, then over
    # clean matches only.  The coarse tolerance has to cover the rotation the
    # vote cannot see -- 3 degrees over 512 px is 27 px -- and the fine one is
    # the registration accuracy being claimed.  A third, intermediate stage
    # was measured and changes nothing: once the coarse round has the
    # transformation to a fifth of a pixel, the tight round keeps the same
    # pairs whether or not it was approached gradually.  Dropping the tight
    # round costs a factor of three to seven, so 0 -- which skips it -- is for
    # comparison and not for use.
    coarse_tolerance_px: float = 20.0
    # How close a pair must be, in the reference channel, once the coarse
    # round has placed the two channels on top of each other.  One pixel is
    # roughly seven times the residual a healthy registration achieves
    # (~0.13 px measured), so it is loose enough not to shave the edges of
    # the field and tight enough to keep chance partners out at SMLM
    # densities.  It is a floor rather than a fixed value: the tight round
    # widens to whatever the coarse round's own misfit says it must be, and
    # tightening past that is what silently strips the periphery.
    fine_tolerance_px: float = 1.0
    # On, but guarded: see MIN_CLEAN_FRACTION.  It protects the edges of a
    # distorted field, where the projective fit misses by more than any fixed
    # tolerance allows; the guard is what stops it misfiring on a dense
    # dataset, where the coarse round's large residual means contamination
    # rather than distortion.
    adapt_fine_tolerance: bool = True

    @property
    def tolerances(self) -> Tuple[float, ...]:
        if self.fine_tolerance_px <= 0:
            return (self.coarse_tolerance_px,)
        return (self.coarse_tolerance_px, self.fine_tolerance_px)
    min_pairs: int = 20
    max_pairs: int = 20000
    reprojection_threshold_px: float = 1.0
    transform_axis_limit_px: float = 0.5

    def validate(self):
        if self.layout != 'auto' and self.layout not in LAYOUTS:
            raise ValueError(f'layout must be auto or one of {LAYOUTS}')
        if self.model not in ('projective', 'polynomial'):
            raise ValueError("model must be 'projective' or 'polynomial'")
        if self.main_channel != 'auto':
            # with the layout still to be detected, any of the four names is
            # allowed here and the one that turns out not to fit the detected
            # split axis is refused later, by name
            halves = (('left', 'right', 'upper', 'lower') if self.layout == 'auto'
                      else ('left', 'right') if 'right-left' in self.layout
                      else ('upper', 'lower'))
            if self.main_channel not in halves:
                raise ValueError('main channel must match the split layout')
        if self.split_position is not None and self.split_position < 1:
            raise ValueError('split_position must be a positive integer or None')
        if self.vote_bin_px <= 0 or self.vote_smooth_px <= 0:
            raise ValueError('the vote bin and smoothing must be positive')
        if self.coarse_tolerance_px <= 0:
            raise ValueError('the coarse matching tolerance must be positive')
        if self.fine_tolerance_px > self.coarse_tolerance_px:
            raise ValueError('the fine matching tolerance must not exceed the '
                             'coarse one; the second round tightens the first')
        if self.min_pairs < 4:
            raise ValueError('min_pairs must be at least four')
        if self.max_pairs < self.min_pairs:
            raise ValueError('max_pairs cannot be below min_pairs')
        if self.reprojection_threshold_px <= 0 or self.transform_axis_limit_px <= 0:
            raise ValueError('the reprojection and dx/dy limits must be positive')


@dataclass
class Vote:
    """The global alignment, before a single pair has been matched."""
    mode: str                    # which axis, if any, was summed
    peak: np.ndarray             # (2,) the winning feature vector
    score: float                 # peak height over the map's noise
    histogram: np.ndarray        # for the diagnostic plot
    edges: Tuple[np.ndarray, np.ndarray]
    n_vectors: int

    @property
    def mirrored(self) -> bool:
        return self.mode != 'none'

    @property
    def mirror_axis(self) -> Optional[int]:
        """0 for x, 1 for y, in (x, y) order -- ``mirror_axis_xy``."""
        return None if self.mode == 'none' else 'xy'.index(self.mode)


@dataclass
class Round:
    """One pair-and-refit pass, kept so the progression can be shown."""
    tolerance_px: float
    n_pairs: int
    residual_px: np.ndarray      # per-pair distance after the fit
    accepted: np.ndarray
    coverage_px2: float = 0.0    # hull of the pairs it kept


@dataclass
class RegistrationResult:
    transform: ChannelTransform
    vote: Vote
    rounds: list
    reference_points: np.ndarray     # the pairs the last round kept
    secondary_points: np.ndarray
    residuals: np.ndarray            # (n, 2) dx, dy after the final fit
    accepted: np.ndarray
    counts: dict

    @property
    def transformation(self):
        return self.transform.transformation

    @property
    def geometry(self):
        return self.transform.geometry

    def save(self, path, overwrite=False):
        save_channel_transform(path, self.transform, self, overwrite)


# ----------------------------------------------------------------- the vote
def _frame_bounds(frame: np.ndarray):
    """Start and stop of every frame in a table sorted by frame."""
    edges = np.flatnonzero(np.diff(frame)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(frame)]))
    return starts, stops


def _feature_vectors(x, y, frame, mode, max_vectors=4_000_000):
    """Every within-frame pair, as the feature the vote is taken in.

    ``mode`` names the axis whose *sum* is taken -- the mirrored one -- and
    the other axis contributes a difference.  Ordered pairs, both directions,
    so an unmirrored vote has a peak at ``+d`` and one at ``-d`` and the
    choice between them is the choice of which half is the reference.
    """
    starts, stops = _frame_bounds(frame)
    out, total = [], 0
    for start, stop in zip(starts, stops):
        k = stop - start
        if k < 2:
            continue
        fx, fy = x[start:stop], y[start:stop]
        dx = (fx[None, :] + fx[:, None]) if mode == 'x' else (fx[None, :] - fx[:, None])
        dy = (fy[None, :] + fy[:, None]) if mode == 'y' else (fy[None, :] - fy[:, None])
        off = ~np.eye(k, dtype=bool)
        out.append(np.c_[dx[off], dy[off]])
        total += out[-1].shape[0]
        if total >= max_vectors:
            break
    if not out:
        return np.empty((0, 2))
    return np.concatenate(out)


def _vote(x, y, frame, mode, settings: RegisterSettings) -> Vote:
    """Accumulate the feature vectors and find the constant among them.

    The map is scored as a difference of Gaussians: the answer is a *sharp*
    peak, while everything the pairing is not interested in -- the short
    vectors between two molecules in the same half, and, in a mirrored vote,
    the broad ridge that same-half pairs draw along the sum axis -- is smooth
    on a scale of many bins and subtracts away.  The same reasoning, and the
    same filter, as the peak finder uses on an image.
    """
    vectors = _feature_vectors(x, y, frame, mode)
    if len(vectors) < 4:
        return Vote(mode, np.zeros(2), 0.0, np.zeros((1, 1)),
                    (np.zeros(2), np.zeros(2)), len(vectors))

    bin_px = float(settings.vote_bin_px)
    ranges = []
    for axis, summed in enumerate((mode == 'x', mode == 'y')):
        values = (x, y)[axis]
        if summed:
            ranges.append((2 * values.min() - bin_px, 2 * values.max() + bin_px))
        else:
            span = values.max() - values.min() + bin_px
            ranges.append((-span, span))
    bins = [max(int(round((hi - lo) / bin_px)), 4) for lo, hi in ranges]

    hist, ex, ey = np.histogram2d(vectors[:, 0], vectors[:, 1], bins=bins,
                                  range=ranges)
    narrow = settings.vote_smooth_px / bin_px
    score = (ndimage.gaussian_filter(hist, narrow)
             - ndimage.gaussian_filter(hist, narrow * 4))
    if mode == 'none':
        # the only place a short vector can come from is one half
        centre = [np.searchsorted(e, 0.0) - 1 for e in (ex, ey)]
        lo = [max(c - ORIGIN_GUARD_BINS, 0) for c in centre]
        hi = [c + ORIGIN_GUARD_BINS + 1 for c in centre]
        score[lo[0]:hi[0], lo[1]:hi[1]] = -np.inf

    flat = int(np.argmax(score))
    i, j = np.unravel_index(flat, score.shape)
    finite = score[np.isfinite(score)]
    spread = float(np.std(finite)) or 1.0
    peak = _centroid(hist, ex, ey, i, j, max(int(round(narrow)), 1))
    return Vote(mode, peak, float(score[i, j] / spread), hist, (ex, ey),
                len(vectors))


def _centroid(hist, ex, ey, i, j, radius):
    """Sub-bin position of the peak, from the counts around it."""
    lo_i, hi_i = max(i - radius, 0), min(i + radius + 1, hist.shape[0])
    lo_j, hi_j = max(j - radius, 0), min(j + radius + 1, hist.shape[1])
    window = hist[lo_i:hi_i, lo_j:hi_j]
    cx = (ex[lo_i:hi_i] + ex[lo_i + 1:hi_i + 1]) / 2
    cy = (ey[lo_j:hi_j] + ey[lo_j + 1:hi_j + 1]) / 2
    weight = window.sum()
    if weight <= 0:
        return np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
    return np.array([float(window.sum(axis=1) @ cx / weight),
                     float(window.sum(axis=0) @ cy / weight)])


def find_alignment(x, y, frame, settings: RegisterSettings) -> Vote:
    """The global alignment: which layout, and the constant that defines it.

    Every allowed feature space is tried and the sharpest peak wins.  A layout
    the user has declared narrows the field to the one space that can answer
    for it, so declaring it is a restriction and never a hint that is ignored.
    """
    modes = MODES if settings.layout == 'auto' else [
        m for m, layouts in MODE_LAYOUTS.items() if settings.layout in layouts]
    votes = [_vote(x, y, frame, mode, settings) for mode in modes]
    best = max(votes, key=lambda v: v.score)
    if best.score <= 0 or not np.isfinite(best.score):
        raise ValueError('no channel offset stands out among the localization '
                         'pairs: too few localizations per frame, or the two '
                         'channels do not show the same molecules')
    return best


# ------------------------------------------------------- geometry and guesses
def _relation(vote: Vote) -> np.ndarray:
    """The vote as a map: where a low-half point's partner is.

    "Low" and "high" are the two sides of the split, not the reference and
    secondary channels -- which is the reference is a separate question, asked
    of ``main_channel`` once the split axis is known.

    Unmirrored, the partner is one translation away.  Mirrored on an axis, the
    partners satisfy ``low + high = s`` there, so the map reflects about
    ``s/2`` -- and is its own inverse, which is why that axis of a mirrored
    vote has no sign to resolve.
    """
    h = np.eye(3)
    for axis in range(2):
        if vote.mirrored and axis == vote.mirror_axis:
            h[axis, axis] = -1.0
            h[axis, 2] = vote.peak[axis]
        else:
            h[axis, 2] = vote.peak[axis]
    return h


def _channel_map(vote: Vote, reference_is_first: bool) -> np.ndarray:
    """Secondary -> reference, from the vote alone: the starting transform."""
    forward = _relation(vote)
    return np.linalg.inv(forward) if reference_is_first else forward


def _split_axis_from(vote: Vote, extent) -> int:
    """Which coordinate the chip is split along: 0 for x, 1 for y.

    A mirrored vote says so outright.  Otherwise it is the axis the channels
    are far apart along -- half a chip -- rather than the one they are merely
    a few pixels apart along.
    """
    if vote.mirrored:
        return vote.mirror_axis
    relative = np.abs(vote.peak) / np.maximum(extent, 1.0)
    return int(np.argmax(relative))


def _partners(x, y, frame, forward, tolerance: float):
    """Mutually-nearest partners under `forward`, frame by frame.

    ``forward`` maps a low-side position to where its partner sits, so the two
    index arrays returned are the high side and the low side of each pair.
    For a reflection those names are empty: it is its own inverse and matches
    both ways round, and only the *count* means anything -- which is all
    `_orient` asks of it.
    """
    from ..dualfit import _match          # the same mutually-nearest pairing
    points = np.c_[x, y]
    back = map_points(np.linalg.inv(forward), points)
    starts, stops = _frame_bounds(frame)
    high, low = [], []
    for start, stop in zip(starts, stops):
        a, b = _match(back[start:stop], points[start:stop], tolerance)
        high.append(start + a)
        low.append(start + b)
    if not high:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(high), np.concatenate(low)


def _sides(x, y, frame, forward, tolerance: float):
    """Which side of the split each localization is on, by where its partner is.

    Not by where it is: that would need the split position, which is what this
    is on the way to measuring.  A localization that looks like both sides --
    two molecules lining up by chance -- is claimed by neither.
    """
    high_idx, low_idx = _partners(x, y, frame, forward, tolerance)
    high = np.zeros(len(x), bool)
    low = np.zeros(len(x), bool)
    high[high_idx] = True
    low[low_idx] = True
    return low & ~high, high & ~low


def _orient(x, y, frame, vote: Vote, tolerance: float) -> Vote:
    """Resolve the sign the vote cannot resolve on its own.

    Every ordered pair is counted in both directions, so an unmirrored vote
    has a peak at ``+d`` and one at ``-d``.  Those two are one answer read
    from opposite ends, and choosing between them is pure convention: take the
    one pointing from the low side of the split to the high side, which is
    what the rest of this module assumes.

    A mirrored vote has a single peak on its sum axis but still two on the
    other, and there the two are *not* equivalent -- only one of them pairs
    any localizations at all.  That one is found by trying both and counting.
    """
    if not vote.mirrored:
        dominant = int(np.argmax(np.abs(vote.peak)))
        if vote.peak[dominant] < 0:
            vote = replace(vote, peak=-vote.peak)
        return vote
    other = 1 - vote.mirror_axis
    flipped = vote.peak.copy()
    flipped[other] *= -1
    options = [vote, replace(vote, peak=flipped)]
    counts = [len(_partners(x, y, frame, _relation(v), tolerance)[0])
              for v in options]
    return options[int(np.argmax(counts))]


def _split_estimate(x, y, frame, vote: Vote, forward, coord: int, tolerance: float):
    """Where the two halves of the frame meet, in chip coordinates.

    Returns ``(split, low_edge, high_edge, how)``; the two edges bracket the
    values consistent with the data, and how wide that bracket is, is the
    thing worth looking at.

    **Unmirrored**, the pairs place it: a localization whose partner lies
    above it is below the seam and vice versa, so any value between the two
    populations classifies all of them correctly and the midpoint is the
    natural pick.  The bracket is narrow when the two halves see the same
    field, which is what a splitter is for, and wide when a misalignment
    leaves a strip of one half with nothing to pair against.  This is asked of
    the *fitted* transformation at the tightest tolerance, not of the vote at
    the loosest: at 20 px and a few thousand localizations, chance matches
    reach right across the frame and it is their tails, not the halves, that
    the percentiles would be measuring.

    **Mirrored**, the pairs cannot place it at all.  A seam at ``c`` with a
    residual shift ``t`` and one at ``c + t/2`` with no shift predict exactly
    the same pairs, so no amount of data separates them.  What the vote does
    measure is the mirror line, which is the seam for a perfectly aligned
    splitter and within half the misalignment of it otherwise -- a good
    estimate, and the honest name for it is an estimate.  Set
    ``split_position`` when that is not good enough.
    """
    if vote.mirrored:
        line = float((vote.peak[coord] + 1) / 2)
        return line, line, line, 'the mirror line, which a reflection cannot separate from the seam'
    low, high = _sides(x, y, frame, forward, tolerance)
    if not low.any() or not high.any():
        raise ValueError('could not tell the two halves apart from the pairs found')
    values = (x, y)[coord]
    below = float(np.percentile(values[low], 99.9))
    above = float(np.percentile(values[high], 0.1))
    return (below + above) / 2, below, above, 'measured from the pairs'


def _geometry(vote, coord, split, main_channel, image_shape, roi) -> dict:
    layout = ('right-left' if coord == 0 else 'up-down') + \
             (' mirrored' if vote.mirrored else '')
    halves = ('left', 'right') if coord == 0 else ('upper', 'lower')
    if main_channel == 'auto':
        main_channel = halves[0]
    height, width = int(image_shape[0]), int(image_shape[1])
    return {'layout': layout, 'main_channel': main_channel,
            'split_position': int(round(split)),
            'image_shape': [height, width],
            'coordinate_system': 'camera-chip' if roi is not None else 'roi-local',
            'sources': [{'path': None, 'axes': None,
                         'roi': list(roi) if roi is not None else None,
                         'shape': [height, width], 'z_nm': None}],
            'mirror_axis_xy': vote.mirror_axis,
            'convention': 'zero-based pixel centers; split is first index of '
                          'second half, in chip coordinates'}

# ----------------------------------------------------------- pair and refit
def _as_mapping(h):
    """A 3x3 as the callable the pairing takes, so both models look alike."""
    return lambda points: map_points(h, points)


def _pair(x, y, frame, mapping, is_reference, is_secondary, tolerance):
    """Match every secondary localization to a reference one, frame by frame.

    The secondary half is mapped into reference coordinates first, so the
    tolerance is a distance in the reference channel and means the same thing
    in every round.  ``mapping`` is a callable, not a matrix, so a polynomial
    round pairs through exactly the same code as a projective one.
    """
    from ..dualfit import _match
    points = np.c_[x, y]
    mapped = np.empty_like(points)
    mapped[is_secondary] = mapping(points[is_secondary])
    starts, stops = _frame_bounds(frame)
    reference, secondary = [], []
    for start, stop in zip(starts, stops):
        here = np.arange(start, stop)
        ref = here[is_reference[start:stop]]
        sec = here[is_secondary[start:stop]]
        if not len(ref) or not len(sec):
            continue
        a, b = _match(points[ref], mapped[sec], tolerance)
        reference.append(ref[a])
        secondary.append(sec[b])
    if not reference:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(reference), np.concatenate(secondary)


def _subsample(n, limit, rng):
    """At most `limit` of `n` indices, for the robust fit.

    RANSAC maps every pair five hundred times; a million of them buys nothing
    a well-spread twenty thousand does not.  SMAP caps the same way
    (``maxlocsused``).
    """
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, limit, replace=False))


def _residual_by_radius(target, dxdy, inliers):
    """Median residual in the inner and outer half of the paired region.

    The comparison a single spread cannot make.  Two runs that kept different
    pairs have incomparable overall residuals -- the one that reached further
    out reports the larger number while being the better registration -- so
    what is wanted is not how large the residual is but how it varies over the
    field.  Flat means noise; rising means the wrong model.
    """
    kept = target[inliers]
    error = np.linalg.norm(dxdy[inliers], axis=1)
    if len(kept) < 8:
        median = float(np.median(error)) if len(error) else 0.0
        return median, median
    radius = np.linalg.norm(kept - kept.mean(axis=0), axis=1)
    edge = radius >= np.median(radius)
    return float(np.median(error[~edge])), float(np.median(error[edge]))


def _fit_one_polynomial(source, target, order, loss_scale):
    """Least squares from `source` to `target`, reweighted against outliers.

    The model is linear in its coefficients, so this is a solve rather than an
    optimisation; the reweighting is the soft-L1 of `refine_projective` done
    by hand, which is all a linear model needs to stop a handful of bad pairs
    from bending the whole field.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    centre = source.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((source - centre) ** 2, axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('degenerate coordinates for a polynomial fit')
    design = polynomial_terms(source, order, centre, scale)
    if np.linalg.matrix_rank(design, tol=1e-8) < design.shape[1]:
        raise ValueError('the pairs do not span the field well enough to fit a '
                         'polynomial of this order')
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    for _ in range(4):
        residual = np.linalg.norm(design @ coef - target, axis=1)
        weight = 1.0 / np.sqrt(1.0 + (residual / max(loss_scale, 1e-6)) ** 2)
        w = weight[:, None]
        coef, *_ = np.linalg.lstsq(design * w, target * w, rcond=None)
    return centre, scale, coef


def fit_polynomial(secondary, reference, order=POLYNOMIAL_ORDER, loss_scale=0.5):
    """Both directions of a polynomial map between the channels.

    Fitted independently rather than one inverted, because a polynomial has no
    closed-form inverse.
    """
    fc, fs, fcoef = _fit_one_polynomial(secondary, reference, order, loss_scale)
    ic, isc, icoef = _fit_one_polynomial(reference, secondary, order, loss_scale)
    return Polynomial(order, fc, fs, fcoef, ic, isc, icoef)


class _PolynomialFit:
    """What `_refit` returns, for a polynomial: the same two fields used after."""

    def __init__(self, polynomial, source, target, limit):
        self.polynomial = polynomial
        self.dxdy = polynomial.forward(source) - target
        self.accepted = np.all(np.abs(self.dxdy) <= limit, axis=1)


def pair_weights(precision, ref_ids, sec_ids):
    """Inverse variance of a pair's separation, from both of its partners.

    A pair is only as good as its worse half, and a registration built from
    localizations has partners spanning orders of magnitude in brightness --
    especially in the dim channel, which is where the threshold has just been
    lowered to find any partners at all.  Unweighted, those faint pairs count
    as much as the bright ones and the fit follows their noise; weighted, they
    still contribute, but in proportion to what they know.

    The variance of the *difference* is the sum of the two, so the weight is
    ``1/(s_ref^2 + s_sec^2)``.  Normalised to a mean of one, so the soft-L1
    scale downstream goes on meaning pixels.
    """
    if precision is None:
        return None
    sigma = np.asarray(precision, float)
    variance = sigma[ref_ids] ** 2 + sigma[sec_ids] ** 2
    good = np.isfinite(variance) & (variance > 0)
    if not good.any():
        return None
    # a pair with no precision recorded gets the median one rather than being
    # dropped: it is a real pair, only an unmeasured one
    variance = np.where(good, variance, np.median(variance[good]))
    weight = 1.0 / variance
    return weight / weight.mean()


def _refit(x, y, frame, ref_ids, sec_ids, tolerance, s, rng, precision=None):
    """One robust projective fit over the pairs matched at this tolerance."""
    if len(ref_ids) < s.min_pairs:
        raise ValueError(f'only {len(ref_ids)} pairs within {tolerance:g} px; '
                         f'require {s.min_pairs}.  Check the layout, or raise '
                         'the coarse matching tolerance.')
    take = _subsample(len(ref_ids), s.max_pairs, rng)
    target = np.c_[x[ref_ids[take]], y[ref_ids[take]]]
    source = np.c_[x[sec_ids[take]], y[sec_ids[take]]]
    weights = pair_weights(precision, ref_ids[take], sec_ids[take])
    # the screens tighten with the tolerance, but never past what was asked
    # for: the last round is the accuracy the result claims
    round_settings = _replace(
        s, reprojection_threshold_px=max(tolerance / 2, s.reprojection_threshold_px),
        transform_axis_limit_px=max(tolerance / 4, s.transform_axis_limit_px))
    fitted = fit_dual_transform(source, target, round_settings, weights)
    return source, target, fitted


def _polynomial_round(x, y, frame, ref_ids, sec_ids, is_reference, is_secondary,
                      covered, tolerance, s, rng, report):
    """Fit a cubic on the projective round's pairs, then re-pair through it.

    Returns ``(fit, why_not)``: one of them is None.  The refusals are the
    point of the function as much as the fit is -- a cubic that is not
    supported by the pairs is worse than the projective map it would replace,
    and quietly so, because its residual on those pairs looks excellent while
    it does whatever it likes everywhere else.
    """
    need = MIN_PAIRS_PER_COEFFICIENT*n_polynomial_terms(POLYNOMIAL_ORDER)
    if len(ref_ids) < need:
        return None, (f'{len(ref_ids)} pairs is too few for a polynomial of '
                      f'order {POLYNOMIAL_ORDER} ({need} wanted)')
    field = point_coverage_area(np.c_[x[is_reference], y[is_reference]])
    if field > 0 and covered < MIN_POLYNOMIAL_COVERAGE*field:
        return None, (f'the pairs cover {covered/field:.0%} of the reference '
                      f'channel, and a polynomial outside its pairs is an '
                      f'extrapolation, not a transformation')
    take = _subsample(len(ref_ids), s.max_pairs, rng)
    target = np.c_[x[ref_ids[take]], y[ref_ids[take]]]
    source = np.c_[x[sec_ids[take]], y[sec_ids[take]]]
    try:
        polynomial = fit_polynomial(source, target, POLYNOMIAL_ORDER,
                                    loss_scale=max(tolerance/4, 0.05))
    except (ValueError, np.linalg.LinAlgError) as error:
        return None, str(error)
    # re-pair through it: the whole reason for a better model is that it
    # reaches pairs the projective one put out of range
    ref_ids, sec_ids = _pair(x, y, frame, polynomial.forward, is_reference,
                             is_secondary, tolerance)
    if len(ref_ids) < need:
        return None, f'only {len(ref_ids)} pairs after the polynomial re-pairing'
    take = _subsample(len(ref_ids), s.max_pairs, rng)
    target = np.c_[x[ref_ids[take]], y[ref_ids[take]]]
    source = np.c_[x[sec_ids[take]], y[sec_ids[take]]]
    polynomial = fit_polynomial(source, target, POLYNOMIAL_ORDER,
                                loss_scale=max(tolerance/4, 0.05))
    fitted = _PolynomialFit(polynomial, source, target,
                            max(tolerance/4, s.transform_axis_limit_px))
    report(f'Registration: polynomial order {POLYNOMIAL_ORDER} -> '
           f'{len(ref_ids):,} pairs, {fitted.accepted.sum():,} inliers, '
           f'dx {np.std(fitted.dxdy[fitted.accepted, 0]):.3f} px, '
           f'dy {np.std(fitted.dxdy[fitted.accepted, 1]):.3f} px')
    return (polynomial, source, target, fitted, ref_ids, sec_ids), None


def register_channels(x, y, frame, image_shape, roi=None, settings=None,
                      progress=None, precision=None) -> RegistrationResult:
    """Find the transformation between the two halves, from localizations.

    ``x`` and ``y`` are chip pixels, ``frame`` the frame each was found in,
    ``image_shape`` the frame's ``(height, width)`` and ``roi`` the camera ROI
    ``(x0, y0, width, height)`` those chip coordinates are relative to -- None
    when it is unknown, which makes the result ROI-local, exactly as for a
    bead calibration.

    The order of the steps is the point of this function.  The first round
    pairs by the vote alone, because the vote is the one thing that needs no
    split position; only once that round has produced a transformation good to
    a fraction of a pixel is the seam between the halves measured, at the
    tightest tolerance, where the answer means something.  Every later round
    then works with geometric halves, so a localization whose partner did not
    blink in this frame still takes part in the next.
    """
    s = settings or RegisterSettings()
    s.validate()
    report = progress or (lambda message: None)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    frame = np.asarray(frame)
    if not (len(x) == len(y) == len(frame)):
        raise ValueError('x, y and frame must be the same length')
    if len(x) < 2 * s.min_pairs:
        raise ValueError(f'{len(x)} localizations is too few to register on')
    order = np.argsort(frame, kind='stable')
    x, y, frame = x[order], y[order], frame[order]
    if precision is not None:
        precision = np.asarray(precision, float)[order]

    report('Registration: voting for the channel offset')
    coarse, fine = s.coarse_tolerance_px, s.fine_tolerance_px
    vote = _orient(x, y, frame, find_alignment(x, y, frame, s), coarse)
    extent = np.array([x.max() - x.min(), y.max() - y.min()])
    coord = _split_axis_from(vote, extent)
    what = 'mirror line' if vote.mirrored else 'offset'
    report(f'Registration: {vote.mode if vote.mirrored else "unmirrored"} '
           f'{what} {vote.peak[0]:.1f}, {vote.peak[1]:.1f} px '
           f'(contrast {vote.score:.1f} from {vote.n_vectors:,} vectors)')

    halves = ('left', 'right') if coord == 0 else ('upper', 'lower')
    if s.main_channel != 'auto' and s.main_channel not in halves:
        raise ValueError(f"the chip is split {halves[0]}/{halves[1]}, so the main "
                         f"channel cannot be {s.main_channel!r}; use "
                         f"{halves[0]!r} or {halves[1]!r}, or set the layout")
    main_channel = halves[0] if s.main_channel == 'auto' else s.main_channel
    first = main_channel in ('left', 'upper')

    rng = np.random.default_rng(1729)
    rounds = []

    # round one: paired by the vote, which needs no split position
    high_idx, low_idx = _partners(x, y, frame, _relation(vote), coarse)
    ref_ids, sec_ids = (low_idx, high_idx) if first else (high_idx, low_idx)
    source, target, fitted = _refit(x, y, frame, ref_ids, sec_ids, coarse, s,
                                    rng, precision)
    transformation = fitted.transformation
    rounds.append(Round(float(coarse), len(ref_ids),
                        np.linalg.norm(fitted.dxdy, axis=1), fitted.accepted,
                        point_coverage_area(target[fitted.accepted])))
    report(f'Registration: {coarse:g} px -> {len(ref_ids):,} pairs, '
           f'{fitted.accepted.sum():,} inliers, '
           f'dx {np.std(fitted.dxdy[fitted.accepted, 0]):.3f} px, '
           f'dy {np.std(fitted.dxdy[fitted.accepted, 1]):.3f} px')

    # now the seam, from that transformation rather than from the vote
    forward = np.linalg.inv(transformation) if first else transformation
    measured, below, above, how = _split_estimate(x, y, frame, vote, forward,
                                                  coord, fine if fine > 0 else coarse)
    split = float(s.split_position) if s.split_position is not None else measured
    geometry = _geometry(vote, coord, split, main_channel, image_shape, roi)
    report(f"Registration: {geometry['layout']}, split at "
           f"{geometry['split_position']} px "
           f"({'given' if s.split_position is not None else how})")
    if s.split_position is None and above - below > 8:
        warnings.warn(f'the two halves leave a {above - below:.0f} px band with no '
                      'pairs in it, so the split position is only placed to within '
                      'that band; set split_position if it matters', stacklevel=2)

    from ..dualfit import which_channel
    is_reference = which_channel(x, y, geometry)
    is_secondary = ~is_reference

    # How well the coarse round actually fits, so that the tight round is not
    # tightened past it -- which is what strips the edges of the field, where
    # a projective map misfits most.
    #
    # Measured over the pairs the coarse fit *accepted*, and this is the
    # subtle part.  The obvious choice is every pair it matched, on the
    # grounds that the ones it rejected are the periphery and they are what
    # must stay reachable.  In a dense dataset that is wrong: matching at
    # twenty pixels over a hundred thousand localizations turns up chance
    # partners all over the frame, and it is their residuals, not the
    # periphery's, that a high percentile then measures.  On the NPC dataset
    # it read eighteen pixels of "misfit" that did not exist and widened the
    # tight round until it let the chance matches back in, doubling the final
    # residual.  The accepted pairs are real pairs, and how badly the fit
    # misses on its own inliers is the honest measure of how far it is from
    # describing the field.
    clean = float(np.mean(fitted.accepted)) if len(fitted.accepted) else 0.0
    if fine > 0 and s.adapt_fine_tolerance and clean < MIN_CLEAN_FRACTION:
        report(f'Registration: the coarse round kept {clean:.0%} of its pairs, '
               f'so its residual is chance partners rather than field '
               f'distortion; leaving the tight round at {fine:g} px')
    elif fine > 0 and s.adapt_fine_tolerance:
        misfit = np.linalg.norm(fitted.dxdy[fitted.accepted], axis=1)
        needed = PAIR_MARGIN * float(np.percentile(misfit, PAIR_PERCENTILE))
        if needed > fine:
            report(f'Registration: widening the tight round to {needed:.2f} px '
                   f'-- the coarse fit misses its own pairs by '
                   f'{needed / PAIR_MARGIN:.2f} px')
            # up to the coarse tolerance, but no further: beyond that the
            # coarse round did not look, so there is nothing there to find.
            # What keeps this from letting chance partners back in is the
            # clean-fraction guard above, not a cap -- a cap tight enough to
            # do that job would also be too tight to reach the periphery,
            # which is the whole point.
            fine = min(needed, coarse)

    # round two: re-paired through that transformation, over clean matches only
    if fine > 0:
        ref_ids, sec_ids = _pair(x, y, frame, _as_mapping(transformation),
                                 is_reference, is_secondary, fine)
        source, target, fitted = _refit(x, y, frame, ref_ids, sec_ids, fine, s,
                                        rng, precision)
        transformation = fitted.transformation
        rounds.append(Round(float(fine), len(ref_ids),
                            np.linalg.norm(fitted.dxdy, axis=1), fitted.accepted,
                            point_coverage_area(target[fitted.accepted])))
        report(f'Registration: {fine:g} px -> {len(ref_ids):,} pairs, '
               f'{fitted.accepted.sum():,} inliers, '
               f'dx {np.std(fitted.dxdy[fitted.accepted, 0]):.3f} px, '
               f'dy {np.std(fitted.dxdy[fitted.accepted, 1]):.3f} px')

    polynomial = None
    if s.model == 'polynomial' and fine > 0:
        polynomial, why = _polynomial_round(x, y, frame, ref_ids, sec_ids,
                                            is_reference, is_secondary,
                                            rounds[-1].coverage_px2, fine, s, rng,
                                            report)
        if polynomial is None:
            warnings.warn(f'keeping the projective fit: {why}', stacklevel=2)
        else:
            source, target, fitted, ref_ids, sec_ids = polynomial[1:]
            polynomial = polynomial[0]
            rounds.append(Round(float(fine), len(ref_ids),
                                np.linalg.norm(fitted.dxdy, axis=1), fitted.accepted,
                                point_coverage_area(target[fitted.accepted])))

    # A tighter round is meant to clean the pairs up, not to shrink the part of
    # the field they speak for.  When it does shrink it, the transformation has
    # quietly become an extrapolation over the edges -- and nothing in the
    # residual will say so, because the residual is only ever measured on what
    # survived.  This is the one check that catches it.
    inliers = fitted.accepted
    centre_px, edge_px = _residual_by_radius(target, fitted.dxdy, inliers)
    # A residual that grows toward the edge of the field is not noise and is
    # not something a tolerance can fix: it is the projective model failing to
    # represent the distortion.  Worth separating from an honestly noisy
    # registration, because the remedies are opposite -- more pairs will not
    # help, and a wider tolerance only spreads the same error more evenly.
    if edge_px > 2*centre_px and edge_px > 0.1:
        warnings.warn(
            f'the residual grows from {centre_px:.2f} px in the middle of the '
            f'paired region to {edge_px:.2f} px at its edge, so a projective '
            f'map is not describing this field; the transformation is at its '
            f'best near the centre', stacklevel=2)
    # How much of the field the transformation is actually measured on.  Not
    # the round-over-round change: the coarse round pairs at twenty pixels, and
    # in a dense dataset that picks up chance matches scattered over the whole
    # frame, so its hull is inflated and shrinking away from it is the tight
    # round working rather than failing.  What does compare is the pairs
    # against the localizations -- the part of the reference channel that has
    # molecules in it at all is the part a transformation has to describe, and
    # beyond the pairs it is an extrapolation whatever the residual says.
    field = point_coverage_area(np.c_[x[is_reference], y[is_reference]])
    covered = point_coverage_area(target[inliers])
    if field > 0 and covered < 0.5*field:
        warnings.warn(
            f'the pairs cover {covered/field:.0%} of the reference channel, so '
            f'the transformation is an extrapolation over the rest of it; '
            f'register on frames from across the whole movie, or look at the '
            f'coverage panel to see which part of the field it speaks for',
            stacklevel=2)
    parameters = {
        'method': 'localization pairs',
        'n_localizations': int(len(x)),
        'n_frames': int(len(np.unique(frame))),
        'n_pairs': int(len(source)),
        'n_inliers': int(inliers.sum()),
        'vote_contrast': float(vote.score),
        'residual_dx_px': float(np.std(fitted.dxdy[inliers, 0])),
        'residual_dy_px': float(np.std(fitted.dxdy[inliers, 1])),
        # split by where in the field the pair sits.  A single spread is not
        # comparable between two runs that kept different pairs: widen the
        # tolerance and the number grows simply because the edges came back,
        # which is an improvement reported as a regression.  These two are
        # comparable, because they say *where* the error is.
        'residual_centre_px': float(centre_px),
        'residual_edge_px': float(edge_px),
        'model': 'polynomial' if polynomial is not None else 'projective',
        'polynomial_order': POLYNOMIAL_ORDER if polynomial is not None else None,
        'coverage_px2': float(covered),
        'field_px2': float(field),
        'covered_fraction': float(covered/field) if field > 0 else None,
        'tolerances_px': [float(r.tolerance_px) for r in rounds],
        'coverage_by_round_px2': [float(r.coverage_px2) for r in rounds],
        'split_interval_px': [float(below), float(above)],
        'split_measured_px': float(measured),
        'split_how': how,
        'mirror_line_px': (float((vote.peak[coord] + 1) / 2)
                           if vote.mirrored else None),
    }
    transform = ChannelTransform(transformation, geometry, parameters, polynomial)
    counts = {'localizations': int(len(x)),
              'reference': int(is_reference.sum()),
              'secondary': int(is_secondary.sum()),
              'pairs': int(len(source)), 'inliers': int(inliers.sum())}
    return RegistrationResult(transform, vote, rounds, target, source,
                              fitted.dxdy, inliers, counts)


def _replace(settings: RegisterSettings, **changes) -> RegisterSettings:
    return RegisterSettings(**{**asdict(settings), **changes})


# ------------------------------------------------------------------- storage
def save_channel_transform(path, transform: ChannelTransform, result=None,
                           overwrite=False):
    """Write the transformation, its geometry and what it was measured from."""
    import h5py
    path = Path(path)
    h = np.asarray(transform.transformation, float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError('invalid projective transformation')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp',
                                     dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as f:
            f.attrs.update(format=FORMAT, version=VERSION)
            f['geometry_json'] = json.dumps(transform.geometry)
            f['parameters_json'] = json.dumps(transform.parameters)
            f['transformation'] = h
            # the projective fit is always written, whatever the model: it is
            # what a polynomial was started from, it is what a reader that
            # does not know about polynomials should fall back to, and it is
            # the one of the two that can be trusted outside the pairs
            f.attrs['model'] = transform.model
            if transform.polynomial is not None:
                q = transform.polynomial
                g = f.create_group('polynomial')
                g.attrs['order'] = int(q.order)
                g['forward_centre'] = np.asarray(q.forward_centre, float)
                g['forward_scale'] = float(q.forward_scale)
                g['forward_coef'] = np.asarray(q.forward_coef, float)
                g['inverse_centre'] = np.asarray(q.inverse_centre, float)
                g['inverse_scale'] = float(q.inverse_scale)
                g['inverse_coef'] = np.asarray(q.inverse_coef, float)
            if result is not None:
                g = f.create_group('diagnostics')
                g['vote_mode'] = result.vote.mode
                for name, value in {
                        'vote_peak': result.vote.peak,
                        'vote_histogram': result.vote.histogram,
                        'vote_edges_x': result.vote.edges[0],
                        'vote_edges_y': result.vote.edges[1],
                        'reference_points': result.reference_points,
                        'secondary_points': result.secondary_points,
                        'residuals_px': result.residuals,
                        'accepted': result.accepted}.items():
                    g.create_dataset(name, data=value, compression='gzip')
                g['rounds_json'] = json.dumps(
                    [{'tolerance_px': r.tolerance_px, 'n_pairs': r.n_pairs,
                      'n_inliers': int(r.accepted.sum())} for r in result.rounds])
                g['counts_json'] = json.dumps(result.counts)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_channel_transform(path) -> ChannelTransform:
    import h5py
    with h5py.File(path, 'r') as f:
        if f.attrs.get('format') != FORMAT or f.attrs.get('version') not in READABLE:
            raise ValueError('unsupported channel transformation format/version')
        h = f['transformation'][...]
        if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
            raise ValueError('invalid projective transformation')
        polynomial = None
        if 'polynomial' in f:
            g = f['polynomial']
            polynomial = Polynomial(
                int(g.attrs['order']), g['forward_centre'][...],
                float(g['forward_scale'][()]), g['forward_coef'][...],
                g['inverse_centre'][...], float(g['inverse_scale'][()]),
                g['inverse_coef'][...])
        return ChannelTransform(h, json.loads(f['geometry_json'][()]),
                                json.loads(f['parameters_json'][()]), polynomial)


def load_transform(path) -> ChannelTransform:
    """The registration in a file, whichever kind of file it is.

    A transformation measured from localizations, or the one inside a
    dual-colour bead calibration -- a 2D two-colour fit needs only the
    registration, so a bead calibration serves it just as well.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        kind = f.attrs.get('format')
    if kind == FORMAT:
        return load_channel_transform(path)
    from .dual import load_dual_color_calibration
    return load_dual_color_calibration(path).as_transform()
