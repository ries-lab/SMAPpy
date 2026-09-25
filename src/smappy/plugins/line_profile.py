"""Profiles along a line ROI, fitted without binning them.

Draw a line ROI over a filament, a membrane or a pair of membranes, and this
takes the localizations inside it, projects them onto the line's own
coordinates -- ``along`` the line and ``across`` it -- and fits the
distribution of one of those coordinates with a model:

* **Gaussian** -- one structure: a filament, a single membrane seen edge-on.
* **two Gaussians** -- two of them, and the number that is wanted is the
  distance between them (a nuclear envelope, the two leaflets, a doublet).
* **step (error function)** -- an edge: a region that is uniformly labelled on
  one side and empty on the other, blurred by the localization error.
* **disk** -- a homogeneously filled circle seen edge-on, as SMAP's Disk.
* **ring** -- a circle seen edge-on, as SMAP's Ring: the profile with the two
  horns, which is what a vesicle or an NPC gives across a line ROI.

Every model carries a uniform **offset**, the flat fraction of unspecific
localizations in the ROI, so a peak is not widened to explain them.

SMAP's ``Analyze/measure/lineprofile`` is the original, and it fits the
*histogram*.  That is the one thing not ported.

How to fit a profile of few localizations
-----------------------------------------

A histogram of thirty localizations is mostly a statement about the bin width:
move the edges by half a bin and the fitted width changes by more than its
error bar.  Two things are wrong with fitting one.  The binning throws
information away, and least squares on Poisson counts is the wrong estimator
where the counts are small -- an empty bin carries as much information as a
full one, and a weight of ``1/sqrt(N)`` cannot express that.

So the default here is **unbinned maximum likelihood** (`method="mle"`): the
model is a probability density over the ROI window, and what is maximized is
``sum(log p(t_i))`` over the localizations themselves.  No bins exist; the
histogram in the figure is drawing only, and changing its bin width changes no
fitted number.  The window is part of the likelihood -- each density is
normalized over the ROI's own width -- so the truncation the ROI imposes costs
no bias.

Two refinements matter more than the estimator when there are few
localizations:

* **Each localization brings its own precision** (`use_precision`).  A
  localization is not a point but a Gaussian of width ``xy_err_nm``, so
  the density is the structure convolved with *that* localization's error:
  width ``sqrt(s^2 + sigma_i^2)``.  The fitted ``s`` is then the width of the
  structure rather than of the picture of it, and for a step or a pair of
  points it is the difference between a number and a number that still has the
  microscope in it.  With few localizations this is most of the accuracy
  available: the precisions are known, and a fit that ignores them spends
  parameters re-measuring them.
* **A uniform background fraction** (`background`).  Unspecific localizations
  inside the ROI are flat, and a Gaussian asked to explain them alone comes
  back too wide.  One extra parameter, bounded to [0, 1].

**The starting values are read off the data, not guessed.**  A likelihood
with a distance, a radius or an edge in it is not convex, and a fit finds the
maximum nearest where it started -- a two-Gaussian fit started on one peak
reports a distance of zero, which looks like an answer.  So the peak, the
half-maximum span and the edge come off a coarsely smoothed histogram (the
profile as the eye reads it, and the one place a bin width enters a number
here), the width has the localization precision taken out of it in
quadrature, and the two-Gaussian start comes from **expectation-maximization**
(`em_two_gaussians`), which moves two overlapping components apart where a
gradient step cannot -- started three ways and the likeliest kept, so two
peaks far apart and of very different height are found as well as two that
overlap (a small peak far from a tall one used to be left to the
background, both components sitting on the tall one).  On a pair 25 nm apart with 8 nm structures and 8 nm
precision, EM lands on the maximum the likelihood then confirms in one
iteration; started instead from a distance of 2 nm, the same fit settles at
4.5 nm and stays there.

What is reported is the fit's log-likelihood with AIC and BIC, so "one
Gaussian or two?" is answered by a number rather than by eye; `model="all"`
fits all three and prints the comparison.  Parameter errors come from the
curvature of the likelihood at its maximum (the observed Fisher information),
which is the usual asymptotic approximation and is optimistic for very small
samples.  Below ~50 localizations, set `bootstrap` to a few hundred and read
the confidence intervals instead: they resample the localizations and assume
nothing about the shape of the likelihood, which is where the small-sample
error bar goes wrong.

`method="binned"` keeps the old way for comparison, as Poisson-weighted least
squares on the histogram rather than SMAP's plain least squares.  Even then
the likelihood that is reported is the unbinned one at the fitted parameters,
so the two methods -- and every model -- are compared on the same scale.

One fit per layer
-----------------

A line is nearly always drawn over *two* channels, and what is wanted is how
they differ -- where each one sits, how far apart they are.  So the unit of
work is a layer: every visible localization layer is fitted separately and
drawn in its own colour, as SMAP's `lineprofile` does.  ``source="selection"``
falls back to one fit over whatever is selected, for a single-channel picture
or a script.

The default profile is the one **along** the line, which is what a line is
usually drawn for; ``axis="across"`` is the one that measures the width of
something the line crosses.

The plugin declares `Plugin.live`, so the GUI offers a *live* tick that
refits while the ROI is dragged: the fit is tens of milliseconds and a
measurement one can aim with is a different tool from one that is asked for
and read afterwards.

Everything here is a module-level function over arrays: `project` and
`fit_profile` need no session, no ROI and no window.  `fit_layers` is the one
that takes a context, and only because the loop over layers needs one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from ..regions import Region
from ..render import positions
from . import Context, Plot, Plugin, Result, param, register

PRECISION_FIELDS = ("xy_err_nm", "xy_err_pix")
PRECISION_Z_FIELDS = ("z_err_nm",)
MIN_FOR_FIT = 8             # below this the maximum is not where the data is
THIN = 30                   # below this the error bars are worth a warning
TINY = 1e-300
SQRT_2PI = np.sqrt(2 * np.pi)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
# how many distinct precisions the drawn curve averages over: the density is
# a mixture over the localizations' own widths, and 200 quantiles of them draw
# the same line as ten thousand
CURVE_SAMPLES = 200
# quadrature nodes for the round shapes; see `_arc_nodes`
ARC_NODES = 32
EM_ROUNDS = 200             # for the two-Gaussian starting values


# ------------------------------------------------------------- the geometry

def line_ends(region: Region) -> Tuple[np.ndarray, np.ndarray, float]:
    """The segment and the width of a line ROI, from its four corners.

    `Region.line` stores ``[p0+n, p1+n, p1-n, p0-n]``, so the ends are the
    midpoints of the two short sides and nothing has to be remembered.
    """
    if region.kind != "line" or len(region.points) != 4:
        raise ValueError(f"a line ROI is wanted, this is a {region.kind} ROI")
    points = np.asarray(region.points, float)
    p0 = 0.5 * (points[0] + points[3])
    p1 = 0.5 * (points[1] + points[2])
    return p0, p1, float(region.width)


def project(x, y, p0, p1) -> Tuple[np.ndarray, np.ndarray]:
    """Positions in the line's own coordinates: along it, and across it.

    ``along`` is measured from the first end, ``across`` from the line itself,
    positive to its left.  Both in the units the table came in.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length <= 0:
        raise ValueError("the line ROI has no length")
    unit = direction / length
    dx, dy = x - p0[0], y - p0[1]
    along = dx * unit[0] + dy * unit[1]
    across = -dx * unit[1] + dy * unit[0]
    return along, across


def column(locs: Localizations, names: Sequence[str]) -> Optional[str]:
    """The first of ``names`` the table has, or None."""
    return next((n for n in names if n in locs), None)


# --------------------------------------------------------------- the models
#
# Each model is a probability *density* over the window, so a fit is a
# likelihood and two models are comparable.  A density takes the structural
# width `s` and the per-localization precisions separately, because what is
# fitted is the structure and what is known is the precision: the two add in
# quadrature and only their sum is ever seen in the picture.

def _widths(s: float, precision: Optional[np.ndarray]) -> np.ndarray:
    """The Gaussian each localization actually contributes."""
    s = max(float(s), 0.0)
    if precision is None:
        return np.atleast_1d(np.float64(max(s, 1e-6)))
    return np.maximum(np.hypot(s, np.asarray(precision, float)), 1e-6)


def _phi(z):
    return np.exp(-0.5 * z * z) / SQRT_2PI


def _Phi(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def gauss_density(t, mu, w, window) -> np.ndarray:
    """A Gaussian of width ``w``, normalized over the window.

    ``w`` is per localization when the precisions are folded in, and one
    number when they are not; numpy broadcasts either.
    """
    lo, hi = window
    z = (t - mu) / w
    norm = _Phi((hi - mu) / w) - _Phi((lo - mu) / w)
    return _phi(z) / w / np.maximum(norm, 1e-12)


def step_density(t, mu, w, window, side: float = 1.0) -> np.ndarray:
    """A half-plane of uniform density with an edge at ``mu``, blurred by ``w``.

    ``side = +1`` rises with t, ``-1`` falls.  The normalization is the
    integral of the error function over the window, which is closed form --
    ``(t-mu) Phi + w phi`` -- so no quadrature is needed and the density stays
    exact for a window that cuts the plateau short.
    """
    lo, hi = window
    shape = _Phi(side * (t - mu) / w)

    def integral(edge):
        z = (edge - mu) / w
        return (edge - mu) * _Phi(z) + w * _phi(z)

    rising = integral(hi) - integral(lo)
    norm = rising if side > 0 else (hi - lo) - rising
    return shape / np.maximum(norm, 1e-12)


def _arc_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature over a circle's angle: where the mass sits, and how much.

    Both round shapes are a circle seen edge-on, so both are an integral over
    the angle, and in that variable neither has a singularity to integrate
    through.  With ``u = R sin(theta)``:

        ring   p(u) du = dtheta / pi                 (uniform in theta)
        disk   p(u) du = (2/pi) cos^2(theta) dtheta

    -- the ring's ``1/sqrt(R^2-u^2)`` edge spike and the disk's square root
    both cancel against ``du``.  So each shape is a *sum of Gaussians* at
    ``R sin(theta_k)``, which costs nothing new: the truncation over the
    window, the per-localization precision and the derivative in ``R`` all
    come from `gauss_density` as they do everywhere else.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    theta = 0.5 * np.pi * nodes                     # [-1, 1] -> [-pi/2, pi/2]
    return np.sin(theta), weights


@lru_cache(maxsize=8)
def _ring_nodes(n: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    offsets, weights = _arc_nodes(n)
    weights = weights / weights.sum()
    return tuple(offsets), tuple(weights)


@lru_cache(maxsize=8)
def _disk_nodes(n: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    offsets, weights = _arc_nodes(n)
    weights = weights * (1.0 - offsets ** 2)        # cos^2(theta)
    weights = weights / weights.sum()
    return tuple(offsets), tuple(weights)


def arc_density(t, mu, radius, w, window, kind: str = "ring") -> np.ndarray:
    """A ring or a filled disk seen edge-on, blurred by ``w``.

    The quadrature is refined when the blur is small next to the radius,
    because that is when the projection's edges are sharp and a coarse sum of
    Gaussians would show as ripples in the curve rather than as a shape.
    """
    typical = float(np.median(np.atleast_1d(w)))
    n = int(np.clip(8 * radius / max(typical, 1e-9), ARC_NODES, 4 * ARC_NODES))
    offsets, weights = (_ring_nodes if kind == "ring" else _disk_nodes)(n)
    axis = (len(offsets),) + (1,) * np.broadcast(np.asarray(t), np.asarray(w)).ndim
    positions = mu + radius * np.asarray(offsets).reshape(axis)
    return np.sum(np.asarray(weights).reshape(axis)
                  * gauss_density(t, positions, w, window), axis=0)


@dataclass
class Model:
    """One shape: its parameters, where to start, and what it means."""
    key: str
    label: str
    names: Tuple[str, ...]
    start: Callable                 # start(t, precision, window, **variant)
    bounds: Callable                # bounds(t, window) -> [(lo, hi), ...]
    shape: Callable                 # shape(params, t, precision, window) -> density
    describe: Callable              # describe(params, errors) -> str
    variants: Tuple[Dict, ...] = ({},)   # fits tried, best likelihood wins
    # Which parameters are widths that add to the localization precision in
    # quadrature, and so are fitted as *variances*.  A width enters the model
    # as sqrt(s^2 + sigma_i^2), which near s = 0 is sigma_i + s^2/2 sigma_i --
    # flat in s.  So s = 0 is a stationary point of the likelihood whatever
    # the data says, and a gradient method started above it walks down to the
    # bound and stops there, reporting a structure of zero width.  In s^2 the
    # same point has a slope, and the fit finds the maximum that is there.
    squares: Tuple[int, ...] = ()


def _spread(t) -> float:
    """A robust width to start from: the MAD, which a tail cannot inflate."""
    return float(max(1.4826 * np.median(np.abs(t - np.median(t))), 1e-3))


def _fwhm(s: float) -> float:
    return FWHM_PER_SIGMA * s


def _pm(value: float, error: float, digits: int = 1) -> str:
    if not np.isfinite(error):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} +- {error:.{digits}f}"


# ------------------------------------------------------- starting values
#
# A likelihood with a distance, a radius or an edge in it is not convex, and
# the fit finds the maximum nearest where it was started: a two-Gaussian fit
# started on one peak stays on one peak and reports a distance of zero.  So
# the starting values are read off the data rather than guessed, and they are
# read off a *smoothed histogram* -- the profile as the eye sees it -- which
# is the one place in this module where a bin width appears in a number.  It
# only has to land in the right valley; the unbinned likelihood does the rest,
# and the fitted numbers do not depend on it.

def smoothed_profile(t, window, bins: Optional[int] = None):
    """A coarse histogram, smoothed: what someone reads off the picture.

    Few bins on purpose.  These are starting values, and a finely binned
    profile of two hundred localizations has a maximum wherever the noise put
    one -- which is exactly the failure this is here to avoid.
    """
    t = np.asarray(t, float)
    if bins is None:
        bins = int(np.clip(len(t) // 10, 12, 60))
    counts, edges = np.histogram(t, bins=bins, range=window)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.pad(counts.astype(float), 2, mode="edge")
    return centres, np.convolve(padded, kernel, mode="valid")


def _top_run(counts: np.ndarray) -> Tuple[int, int]:
    """The run of bins above half the maximum that contains the maximum.

    The *contiguous* run, so a second structure further along the profile
    does not drag the centre towards itself.
    """
    peak = int(np.argmax(counts))
    half = 0.5 * counts[peak]
    low = peak
    while low > 0 and counts[low - 1] >= half:
        low -= 1
    high = peak
    while high < len(counts) - 1 and counts[high + 1] >= half:
        high += 1
    return low, high


def peak_position(t, window) -> float:
    """Where the profile is highest, as the centroid of its top.

    The centroid of the half-maximum run rather than the highest bin: the bin
    is quantized to the bin width and jumps between neighbours with the noise,
    the centroid does neither, and a background that is flat under the peak
    adds to both sides of it and cancels.
    """
    t = np.asarray(t, float)
    centres, counts = smoothed_profile(t, window)
    if not counts.size or counts.max() <= 0:
        return float(np.median(t))
    low, high = _top_run(counts)
    piece = counts[low:high + 1] - 0.5 * counts[int(np.argmax(counts))]
    if piece.sum() <= 0:
        return float(centres[int(np.argmax(counts))])
    return float(np.average(centres[low:high + 1], weights=piece))


def half_max_span(t, window) -> Tuple[float, float]:
    """Where the profile passes half its maximum, on the way up and down.

    Interpolated between bins, so the width it implies is not quantized, and
    clamped to the window when the structure runs out of it.
    """
    t = np.asarray(t, float)
    centres, counts = smoothed_profile(t, window)
    if not counts.size or counts.max() <= 0:
        return (float(np.min(t)), float(np.max(t)))
    low, high = _top_run(counts)
    half = 0.5 * counts.max()

    def crossing(inner: int, outer: int) -> float:
        """Where the line between two bins passes half the maximum."""
        if outer < 0 or outer >= len(counts):
            return float(centres[inner])
        gap = counts[inner] - counts[outer]
        if gap <= 0:
            return float(centres[inner])
        fraction = (counts[inner] - half) / gap
        return float(centres[inner] + fraction * (centres[outer] - centres[inner]))

    return crossing(low, low - 1), crossing(high, high + 1)


def edge_position(t, window, side: float = 1.0) -> float:
    """Where a step profile passes half of its plateau.

    Read from the side the localizations are on: the plateau is the median of
    the smoothed counts over the third of the window where the density is
    high -- a median, so a bright spot sitting on the plateau does not raise
    it -- and the edge is where the profile last falls below half of it.
    Robust in the way that matters here: it does not care what the profile
    does far from the edge, which is where a second structure would be.
    """
    t = np.asarray(t, float)
    centres, counts = smoothed_profile(t, window)
    if not counts.size or counts.max() <= 0:
        return float(np.median(t))
    third = max(len(counts) // 3, 1)
    high = counts[-third:] if side > 0 else counts[:third]
    # the upper quartile of that third, not its median: the ROI usually has
    # empty margin beyond the structure, and a median that counts the empty
    # bins halves the plateau and puts the edge in the middle of nowhere
    plateau = float(np.percentile(high, 75))
    if plateau <= 0:
        return float(np.median(t))
    half = 0.5 * plateau
    # walk in from the fullest bin of the plateau, not from the window's rim,
    # for the same reason
    anchor = int(np.argmax(high)) + (len(counts) - third if side > 0 else 0)
    order = range(anchor, -1, -1) if side > 0 else range(anchor, len(counts))
    previous = None
    for i in order:
        if counts[i] < half:
            if previous is None:
                break
            gap = counts[previous] - counts[i]
            fraction = (counts[previous] - half) / gap if gap > 0 else 0.5
            return float(centres[previous]
                         + fraction * (centres[i] - centres[previous]))
        previous = i
    return float(np.median(t))


def _structure_sigma(observed: float, precision) -> float:
    """The width left over once the localization error is taken out.

    What is measured is ``sqrt(s^2 + sigma^2)``; what is wanted is ``s``.  The
    subtraction can go negative -- the picture can be narrower than the
    precisions say it should be, by noise -- and a start of zero is the one
    place the fit cannot leave, so it keeps a fraction of the width instead.
    """
    observed = float(max(observed, 1e-3))
    if precision is None:
        return observed
    typical = float(np.median(np.asarray(precision, float)))
    return float(max(np.sqrt(max(observed ** 2 - typical ** 2, 0.0)),
                     0.2 * observed))


def em_two_gaussians(t, precision=None, window=None, background: bool = True,
                     rounds: int = EM_ROUNDS, tolerance: float = 1e-6
                     ) -> Dict[str, float]:
    """Starting values for the two-Gaussian fit, by expectation-maximization.

    The distance between two structures is the number people come here for,
    and it is the one a gradient fit loses most easily: started between two
    peaks that overlap, the likelihood's nearest maximum is often the single
    broad Gaussian with the distance at zero.  EM does not have that failure,
    because it never moves the parameters directly -- it assigns each
    localization to a component in proportion to how well the component
    explains it, then re-fits each component to what it was assigned, which
    moves two peaks apart whenever that describes the data better.

    Three things make it the mixture that is actually meant here:

    * each localization is weighted by its **own precision** -- the component
      means are inverse-variance weighted, as they should be when the points
      have known and different errors;
    * the width is **structural**: the update subtracts each localization's
      precision, so what comes out is the width of the structure;
    * the flat **background** is a third component of density ``1/W``, which
      keeps unspecific localizations from pulling a component out to the edge
      of the window.

    EM finds the mixture nearest where it was started, like any local
    method, so it is started three ways and the likeliest answer kept:

    * at the two ends of the profile's top -- two peaks that overlap;
    * on the profile's two highest separate maxima -- two peaks far apart.
      The first start alone fails here whenever one peak is well below half
      the height of the other: the top is then the tall peak only, both
      components sit on it, and the small one is left to the background;
    * at the 15th and 85th percentiles -- as far apart as the data goes,
      which EM can only pull together, for a second peak too faint to be a
      maximum of the smoothed profile.

    This is a start, not the answer: it ignores the truncation at the window,
    where the likelihood that follows does not.  Returns the parameters by
    name, ready for `Model.start`.
    """
    t = np.asarray(t, float)
    window = window or (float(t.min()), float(t.max()))
    sigma = (np.asarray(precision, float) if precision is not None
             else np.zeros_like(t))

    low, high = half_max_span(t, window)
    centre = peak_position(t, window)
    top = np.array([min(low, centre), max(high, centre)], float)
    if top[1] - top[0] < 1e-6:
        top = centre + np.array([-1.0, 1.0]) * _spread(t)
    # the width at what is left of the top; for the separated starts, the top
    # is one peak and its half-width is one peak's width
    wide = _structure_sigma(0.5 * max(high - low, _spread(t)), precision)
    narrow = _structure_sigma(max(high - low, 1e-3) / FWHM_PER_SIGMA, precision)
    starts = [(top, wide)]
    maxima = _two_maxima(*smoothed_profile(t, window))
    if maxima is not None:
        starts.append((np.asarray(maxima, float), narrow))
    starts.append((np.percentile(t, [15.0, 85.0]), narrow))

    best = None
    for mu, s in starts:
        found = _em(t, sigma, window, mu, s, background, rounds, tolerance)
        if best is None or found[-1] > best[-1]:
            best = found
    mu, s, weights, share, _ = best

    order = np.argsort(mu)
    mu, weights = mu[order], weights[order]
    total_weight = float(weights.sum()) or 1.0
    return {"centre": float(mu.mean()), "distance": float(mu[1] - mu[0]),
            "sigma": s, "fraction": float(weights[0] / total_weight),
            "background": float(np.clip(share, 0.0, 0.95))}


def _two_maxima(centres, counts) -> Optional[Tuple[float, float]]:
    """The two highest maxima of a smoothed profile with a dip between them.

    A dip, so the shoulder of one peak is not taken for a second one: the
    profile has to fall below four fifths of the lower maximum on the way.
    None if there is only one.
    """
    counts = np.asarray(counts, float)
    if counts.size < 3:
        return None
    padded = np.r_[-np.inf, counts, -np.inf]
    peaks = [i for i in range(counts.size)
             if counts[i] > 0 and counts[i] >= padded[i] and counts[i] > padded[i + 2]]
    peaks.sort(key=lambda i: counts[i], reverse=True)
    if len(peaks) < 2:
        return None
    first = peaks[0]
    for other in peaks[1:]:
        a, b = sorted((first, other))
        if counts[a:b + 1].min() < 0.8 * counts[other]:
            return float(centres[a]), float(centres[b])
    return None


def _em(t, sigma, window, mu, s, background: bool, rounds: int,
        tolerance: float):
    """EM for two Gaussians of one structural width, from ``mu`` and ``s``.

    Returns ``(mu, s, weights, background share, log-likelihood)``; the
    likelihood is the mixture's own, so starts can be compared on it.
    """
    flat = 1.0 / max(window[1] - window[0], 1e-9)
    mu = np.asarray(mu, float).copy()
    s = float(s)
    weights = np.array([0.5, 0.5]) * (0.95 if background else 1.0)
    share = 0.05 if background else 0.0
    floor = (0.05 * _spread(t)) ** 2
    for _ in range(rounds):
        w2 = s ** 2 + sigma ** 2
        gauss = (np.exp(-0.5 * (t[None, :] - mu[:, None]) ** 2 / w2[None, :])
                 / np.sqrt(2 * np.pi * w2)[None, :])
        parts = weights[:, None] * gauss
        total = parts.sum(axis=0) + share * flat
        total = np.maximum(total, TINY)
        r = parts / total                            # responsibilities
        r_background = (share * flat) / total

        new_weights = r.mean(axis=1)
        new_share = float(r_background.mean()) if background else 0.0
        inverse = r / w2[None, :]
        mass = inverse.sum(axis=1)
        new_mu = np.where(mass > 0, (inverse * t[None, :]).sum(axis=1)
                          / np.maximum(mass, TINY), mu)
        # the structural variance: the scatter that the precisions do not
        # already account for, shared by both components
        residual = (t[None, :] - new_mu[:, None]) ** 2 - sigma[None, :] ** 2
        assigned = r.sum()
        variance = float((r * residual).sum() / assigned) if assigned > 0 else s ** 2
        new_s = float(np.sqrt(max(variance, floor)))

        moved = (np.max(np.abs(new_mu - mu)) + abs(new_s - s)
                 + np.max(np.abs(new_weights - weights)))
        mu, s, weights, share = new_mu, new_s, new_weights, new_share
        if moved < tolerance:
            break
    w2 = s ** 2 + sigma ** 2
    density = ((weights[:, None] * np.exp(-0.5 * (t[None, :] - mu[:, None]) ** 2
                                          / w2[None, :])
                / np.sqrt(2 * np.pi * w2)[None, :]).sum(axis=0) + share * flat)
    return mu, s, weights, share, float(np.sum(np.log(np.maximum(density, TINY))))


# -------------------------------------------------------------- the models

def _start_gauss(t, precision, window, **_) -> Dict[str, float]:
    low, high = half_max_span(t, window)
    observed = max(high - low, 1e-3) / FWHM_PER_SIGMA
    return {"centre": peak_position(t, window),
            "sigma": _structure_sigma(observed, precision)}


def _start_two_gauss(t, precision, window, **_) -> Dict[str, float]:
    return em_two_gaussians(t, precision, window)


def _start_step(t, precision, window, side: float = 1.0, **_) -> Dict[str, float]:
    return {"edge": edge_position(t, window, side),
            "sigma": _structure_sigma(0.5 * _spread(t), precision)}


def _start_round(t, precision, window, **_) -> Dict[str, float]:
    """Centre and radius of a round shape, from the width of its profile.

    Both projections are as wide as the shape: the profile of a ring runs
    from ``-R`` to ``R`` with its peaks at the ends, a disk's from ``-R`` to
    ``R`` with one peak in the middle.  So the half-maximum span is about
    ``2R`` for the disk and rather less for the ring, and starting both from
    it puts the radius in the right valley either way.
    """
    low, high = half_max_span(t, window)
    span = max(high - low, 1e-3)
    return {"centre": peak_position(t, window), "radius": 0.5 * span,
            "sigma": _structure_sigma(0.25 * span, precision)}


def _positive(window) -> Tuple[float, float]:
    return (0.0, window[1] - window[0])


GAUSS = Model(
    key="gauss", label="Gaussian",
    names=("centre", "sigma"),
    start=_start_gauss,
    bounds=lambda t, window: [window, _positive(window)],
    shape=lambda p, t, prec, window: gauss_density(t, p[0],
                                                   _widths(p[1], prec), window),
    describe=lambda p, e: (f"centre {_pm(p[0], e[0])}, sigma {_pm(p[1], e[1])} "
                           f"(FWHM {_fwhm(p[1]):.1f})"),
    squares=(1,),
)

TWO_GAUSS = Model(
    key="two_gauss", label="two Gaussians",
    names=("centre", "distance", "sigma", "fraction"),
    start=_start_two_gauss,
    bounds=lambda t, window: [window, _positive(window), _positive(window),
                              (0.0, 1.0)],
    shape=lambda p, t, prec, window: (
        p[3] * gauss_density(t, p[0] - 0.5 * p[1], _widths(p[2], prec), window)
        + (1 - p[3]) * gauss_density(t, p[0] + 0.5 * p[1],
                                     _widths(p[2], prec), window)),
    describe=lambda p, e: (f"distance {_pm(p[1], e[1])}, sigma {_pm(p[2], e[2])}, "
                           f"centre {_pm(p[0], e[0])}, "
                           f"weights {p[3]:.2f}/{1 - p[3]:.2f}"),
    squares=(2,),
)

STEP = Model(
    key="step", label="step (error function)",
    names=("edge", "sigma"),
    start=_start_step,
    bounds=lambda t, window: [window, _positive(window)],
    shape=lambda p, t, prec, window, side=1.0: step_density(
        t, p[0], _widths(p[1], prec), window, side),
    describe=lambda p, e: f"edge at {_pm(p[0], e[0])}, blur sigma {_pm(p[1], e[1])}",
    # which way the edge faces is not a parameter to be walked to: it is one
    # bit, and both are cheap to try
    variants=({"side": 1.0}, {"side": -1.0}),
    squares=(1,),
)

DISK = Model(
    key="disk", label="disk",
    names=("centre", "radius", "sigma"),
    start=_start_round,
    bounds=lambda t, window: [window, _positive(window), _positive(window)],
    shape=lambda p, t, prec, window: arc_density(t, p[0], p[1],
                                                 _widths(p[2], prec), window,
                                                 kind="disk"),
    describe=lambda p, e: (f"radius {_pm(p[1], e[1])}, centre {_pm(p[0], e[0])}, "
                           f"blur sigma {_pm(p[2], e[2])}"),
    squares=(2,),
)

RING = Model(
    key="ring", label="ring",
    names=("centre", "radius", "sigma"),
    start=_start_round,
    bounds=lambda t, window: [window, _positive(window), _positive(window)],
    shape=lambda p, t, prec, window: arc_density(t, p[0], p[1],
                                                 _widths(p[2], prec), window,
                                                 kind="ring"),
    describe=lambda p, e: (f"radius {_pm(p[1], e[1])}, centre {_pm(p[0], e[0])}, "
                           f"blur sigma {_pm(p[2], e[2])}"),
    squares=(2,),
)

MODELS: Dict[str, Model] = {m.key: m for m in (GAUSS, TWO_GAUSS, STEP,
                                               DISK, RING)}




# ------------------------------------------------------------------ fitting

@dataclass
class Fit:
    """A fitted model: its numbers, its quality, and the curve to draw."""
    model: str
    label: str
    names: Tuple[str, ...]
    params: np.ndarray
    errors: np.ndarray
    background: float               # the uniform fraction, 0 when not fitted
    background_error: float
    n: int
    log_likelihood: float           # always the unbinned one, whatever fitted
    method: str
    with_precision: bool
    window: Tuple[float, float]
    extra: Dict = field(default_factory=dict)
    curve: Optional[Callable] = None     # curve(t) -> density per unit length
    # what `bootstrap` adds: one row of parameters per resample (the
    # background last), and the interval each of them gives
    replicates: Optional[np.ndarray] = None
    intervals: Optional[Dict[str, Tuple[float, float]]] = None
    level: float = 0.95

    @property
    def k(self) -> int:
        """Free parameters, for the information criteria."""
        return len(self.params) + (1 if self.extra.get("has_background") else 0)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.k * np.log(max(self.n, 1)) - 2 * self.log_likelihood

    def values(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.params)}

    def uncertainties(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.errors)}

    def interval_lines(self) -> List[str]:
        """The bootstrap interval of each parameter, one line each."""
        if not self.intervals:
            return []
        percent = int(round(100 * self.level))
        rounds = 0 if self.replicates is None else len(self.replicates)
        lines = [f"{percent}% confidence intervals, {rounds} resamples:"]
        for name in tuple(self.names) + (("background",)
                                         if self.extra.get("has_background")
                                         else ()):
            low, high = self.intervals.get(name, (np.nan, np.nan))
            value = (self.background if name == "background"
                     else self.values()[name])
            lines.append(f"    {name} {value:.2f}  [{low:.2f}, {high:.2f}]")
        return lines

    @property
    def summary(self) -> str:
        text = MODELS[self.model].describe(self.params, self.errors)
        if self.extra.get("has_background"):
            text += f", background {100 * self.background:.0f}%"
        if self.extra.get("side", 1.0) < 0:
            text += ", falling"
        return f"{self.label}: {text}"


def _density_of(model: Model, params, window, background: float,
                has_background: bool, variant: Dict) -> Callable:
    """The density of one model at given parameters.

    Takes the precisions as an argument rather than closing over them, so the
    same function serves the likelihood -- one precision per localization --
    and the drawn curve, which averages over them by broadcasting.
    """
    lo, hi = window
    flat = 1.0 / (hi - lo)

    def density(t, precision):
        shape = model.shape(params, t, precision, window, **variant)
        if not has_background:
            return shape
        return background * flat + (1 - background) * shape

    return density


def _curve_of(density: Callable, precision) -> Callable:
    """The density as a histogram of these localizations would follow it.

    With per-localization precisions the model is a *mixture*: each
    localization is drawn from the structure smoothed by its own error, so
    what a histogram of all of them follows is the average over the sample.
    The precisions are sampled by quantile, which draws the same line as all
    of them and costs the same whatever the table's size.
    """
    if precision is None:
        return lambda t: np.asarray(density(np.asarray(t, float), None), float)
    sampled = np.asarray(precision, float)
    if len(sampled) > CURVE_SAMPLES:
        sampled = np.percentile(sampled, np.linspace(0, 100, CURVE_SAMPLES))

    def curve(t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.asarray(density(t[None, :], sampled[:, None]), float).mean(axis=0)

    return curve


def _hessian(f: Callable, x: np.ndarray) -> np.ndarray:
    """Central differences, with a step scaled to each parameter."""
    n = len(x)
    h = np.maximum(np.abs(x), 1.0) * 1e-4
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            up_up, up_down, down_up, down_down = (x.copy() for _ in range(4))
            up_up[i] += h[i]; up_up[j] += h[j]
            up_down[i] += h[i]; up_down[j] -= h[j]
            down_up[i] -= h[i]; down_up[j] += h[j]
            down_down[i] -= h[i]; down_down[j] -= h[j]
            H[i, j] = H[j, i] = ((f(up_up) - f(up_down) - f(down_up)
                                  + f(down_down)) / (4 * h[i] * h[j]))
    return H


def _errors_from(hessian: np.ndarray,
                 free: Optional[np.ndarray] = None) -> np.ndarray:
    """Standard errors from the curvature, NaN where it says nothing.

    ``free`` says which parameters the fit actually moved.  One that ended on
    its bound -- a width the data puts at zero, a background fraction that
    went to nothing -- has no error bar to quote and would spoil the others'
    by making the matrix singular, so it is left out of the inversion and the
    rest are the errors *given* that it is where it is.

    A missing error bar is never a failed fit: the value is still the maximum
    of the likelihood, and printing it alone is more use than raising.
    """
    n = len(hessian)
    out = np.full(n, np.nan)
    if free is None:
        free = np.ones(n, bool)
    keep = np.flatnonzero(np.asarray(free, bool))
    block = hessian[np.ix_(keep, keep)]
    if not keep.size or not np.all(np.isfinite(block)):
        return out
    try:
        covariance = np.linalg.inv(block)
    except np.linalg.LinAlgError:
        return out
    diagonal = np.diag(covariance)
    out[keep] = np.where(diagonal > 0, np.sqrt(np.abs(diagonal)), np.nan)
    return out


def histogram(t, window, bin_size: float):
    """Counts and edges over the window: for the picture, and for `binned`."""
    lo, hi = window
    n_bins = max(int(round((hi - lo) / max(bin_size, 1e-9))), 1)
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(np.asarray(t, float), bins=edges)
    return counts.astype(float), edges


def auto_bin(t, window) -> float:
    """A bin width for the picture: Freedman-Diaconis, bounded by the window.

    Drawing only -- no fitted number depends on it -- but a histogram of two
    bins or of four hundred says nothing either.
    """
    t = np.asarray(t, float)
    lo, hi = window
    span = hi - lo
    if len(t) < 4:
        return span / 10
    q1, q3 = np.percentile(t, [25, 75])
    width = 2 * (q3 - q1) / len(t) ** (1 / 3)
    if not np.isfinite(width) or width <= 0:
        width = span / 20
    return float(np.clip(width, span / 200, span / 5))


def _clean_precision(precision, n: int):
    """The precisions, or None if there are none worth using."""
    if precision is None:
        return None
    values = np.asarray(precision, float).reshape(-1)
    if values.shape != (n,):
        raise ValueError(f"{values.shape[0]} precisions for {n} localizations")
    good = np.isfinite(values) & (values > 0)
    if not good.any():
        return None
    # a handful of missing precisions become the median rather than dropping
    # their localizations: the position is still a measurement
    return np.where(good, values, float(np.median(values[good])))


def _square(value: float, yes: bool) -> float:
    return float(value) ** 2 if yes else float(value)


def _to_fit(vector: np.ndarray, squares: Sequence[int]) -> np.ndarray:
    """Model parameters as the optimizer sees them: widths become variances."""
    out = np.asarray(vector, float).copy()
    out[squares] = out[squares] ** 2
    return out


def _from_fit(vector: np.ndarray, squares: Sequence[int]) -> np.ndarray:
    """The optimizer's parameters as the model means them."""
    out = np.asarray(vector, float).copy()
    out[squares] = np.sqrt(np.maximum(out[squares], 0.0))
    return out


def _in_widths(errors: np.ndarray, params: np.ndarray,
               squares: Sequence[int]) -> np.ndarray:
    """Errors carried from the variances back to the widths.

    ``s = sqrt(v)``, so ``ds = dv / 2s`` -- and at ``s = 0`` there is no
    linear error to quote, which is the honest answer for a width the data
    puts at the bound.
    """
    out = np.asarray(errors, float).copy()
    for i in squares:
        if i < len(out):
            s = params[i]
            out[i] = out[i] / (2 * s) if s > 0 else np.nan
    return out


def fit_profile(t, precision=None, model: str = "gauss",
                window: Optional[Tuple[float, float]] = None,
                method: str = "mle", background: bool = True,
                bin_size: float = 0.0,
                start: Optional[Dict[str, float]] = None,
                variant: Optional[Dict] = None) -> Fit:
    """Fit one coordinate of the localizations with ``model``.

    ``t`` are the positions themselves, never a histogram, and ``precision``
    is each localization's own error -- or None to fit one width for all of
    them.  ``window`` is where a localization could have been (the ROI), and
    is part of the likelihood, so the truncation costs no bias.

    ``method``: ``"mle"`` maximizes the unbinned likelihood; ``"binned"`` is
    Poisson-weighted least squares on a histogram of ``bin_size``, kept for
    comparison with the old way.  Either way the likelihood reported is the
    unbinned one at the fitted parameters, so every fit is on one scale.

    ``start`` and ``variant`` override the starting values and the direction
    a step faces.  They exist for `bootstrap`, which refits the same data a
    few hundred times and should neither re-derive the starting values on
    each resample nor be free to answer with a different shape each time.
    """
    from scipy.optimize import least_squares, minimize

    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if len(t) < MIN_FOR_FIT:
        raise ValueError(f"{len(t)} localizations: too few for a profile fit "
                         f"(at least {MIN_FOR_FIT})")
    if model not in MODELS:
        raise ValueError(f"no model {model!r}: {', '.join(MODELS)}")
    spec = MODELS[model]
    if window is None:
        pad = 0.05 * (float(t.max() - t.min()) + 1e-9)
        window = (float(t.min()) - pad, float(t.max()) + pad)
    window = (float(window[0]), float(window[1]))
    precision = _clean_precision(precision, len(t))

    best: Optional[Fit] = None
    for shape in (spec.variants if variant is None else (variant,)):
        # the starting values are read off the data (see "starting values"):
        # a model may also say where to start the background, which the
        # two-Gaussian one does, since its EM has just estimated it
        guess = dict(start) if start else dict(spec.start(t, precision, window,
                                                          **shape))
        first = [float(guess[name]) for name in spec.names]
        bounds = list(spec.bounds(t, window))
        if background:
            first.append(float(guess.get("background", 0.05)))
            bounds.append((0.0, 0.95))
        squares = list(spec.squares)
        first = _to_fit(np.asarray(first, float), squares)
        bounds = [(_square(lo, i in squares), _square(hi, i in squares))
                  for i, (lo, hi) in enumerate(bounds)]
        lower = np.array([b[0] for b in bounds], float)
        upper = np.array([b[1] for b in bounds], float)
        first = np.clip(first, lower, upper)

        def split(vector):
            vector = _from_fit(np.asarray(vector, float), squares)
            return ((vector[:-1], float(vector[-1])) if background
                    else (vector, 0.0))

        def nll(vector) -> float:
            params, fraction = split(np.asarray(vector, float))
            density = _density_of(spec, params, window, fraction, background,
                                  shape)
            return float(-np.sum(np.log(np.maximum(density(t, precision), TINY))))

        if method == "binned":
            counts, edges = histogram(t, window, bin_size or auto_bin(t, window))
            centres = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            total = float(counts.sum())

            def residuals(vector):
                params, fraction = split(np.asarray(vector, float))
                density = _density_of(spec, params, window, fraction,
                                      background, shape)
                expected = total * widths * _curve_of(density, precision)(centres)
                # Poisson weights rather than SMAP's plain least squares: at
                # ten counts a bin the two differ by more than the bin width
                return (counts - expected) / np.sqrt(np.maximum(expected, 1.0))

            found = least_squares(residuals, first, bounds=(lower, upper))
            vector = np.asarray(found.x, float)
        else:
            found = minimize(nll, first, method="L-BFGS-B",
                             bounds=list(zip(lower, upper)))
            vector = np.asarray(found.x, float)
            if not np.all(np.isfinite(vector)):
                vector = first

        params, fraction = split(vector)
        span = np.maximum(upper - lower, 1e-12)
        free = (vector > lower + 1e-6 * span) & (vector < upper - 1e-6 * span)
        errors = _in_widths(_errors_from(_hessian(nll, vector), free),
                            _from_fit(vector, squares), squares)
        density = _density_of(spec, params, window, fraction, background, shape)
        fit = Fit(model=spec.key, label=spec.label, names=spec.names,
                  params=np.asarray(params, float),
                  errors=np.asarray(errors[:len(params)], float),
                  background=fraction,
                  background_error=float(errors[-1]) if background else 0.0,
                  n=len(t), log_likelihood=-nll(vector), method=method,
                  with_precision=precision is not None, window=window,
                  extra={"has_background": background, "bin_size": bin_size,
                         **shape},
                  curve=_curve_of(density, precision))
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    return best


def bootstrap(fit: Fit, t, precision=None, rounds: int = 200,
              level: float = 0.95, seed: Optional[int] = None,
              report: Optional[Callable[[str], None]] = None) -> Fit:
    """Confidence intervals by resampling the localizations.

    The errors that come with a fit are the curvature of the likelihood at
    its maximum -- the asymptotic approximation, which assumes the likelihood
    is a parabola and the sample is large.  Neither holds for the profile of
    forty localizations, which is exactly where this plugin is used, and the
    error bar is then optimistic in a way that no amount of care in the fit
    can repair.

    The bootstrap asks the question differently: draw ``len(t)`` localizations
    from the ones there are, with replacement, fit again, and do it a few
    hundred times.  The spread of the answers *is* the uncertainty -- it
    makes no assumption about the shape of the likelihood, and it reports
    asymmetry where there is asymmetry (a width whose lower end runs into
    zero, a distance that a quarter of the resamples cannot resolve at all)
    rather than averaging it into one number.

    Each localization keeps **its own precision** through the resample, so
    the bootstrap carries the heteroscedasticity of the data with it.  Each
    refit starts from the full fit's parameters and keeps its shape (which
    way a step faces): the question is how much the data moves the answer,
    not whether a fresh search finds a different maximum.  That is also why
    the interval is a statement about *this* model -- it says nothing about
    whether the model is the right one, which is what the AIC is for.

    The interval is the percentile one -- the 2.5th and 97.5th of the
    resampled answers for a 95% interval.  It is the simplest of the
    bootstrap intervals and it undercovers a little for a *scale* parameter
    on a small sample: measured here on twenty-five simulated profiles of
    forty localizations, a nominal 95% interval on the width covered the true
    value 22 times.  Close enough to read, not close enough to quote as
    exact; a BCa interval would correct the bias and the skew, at the cost of
    a jackknife on top of the resampling.

    Returns the same fit with `replicates` and `intervals` filled in.  A
    resample that will not fit -- too few distinct positions, an optimizer
    that walks off -- is dropped rather than raised; with fewer than half of
    them left the intervals are not worth quoting and none are returned.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    precision = _clean_precision(precision, len(t)) if precision is not None else None
    has_background = bool(fit.extra.get("has_background"))
    variant = {"side": fit.extra["side"]} if "side" in fit.extra else {}
    start = dict(zip(fit.names, (float(v) for v in fit.params)))
    if has_background:
        start["background"] = fit.background

    kept: List[List[float]] = []
    step = max(rounds // 10, 1)
    for i in range(int(rounds)):
        pick = rng.integers(0, len(t), len(t))
        try:
            again = fit_profile(t[pick],
                                None if precision is None else precision[pick],
                                model=fit.model, window=fit.window,
                                method=fit.method, background=has_background,
                                bin_size=float(fit.extra.get("bin_size", 0.0)),
                                start=start, variant=variant)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        kept.append([float(v) for v in again.params] + [again.background])
        if report and (i + 1) % step == 0:
            report(f"bootstrap {i + 1}/{rounds}")

    if len(kept) < max(rounds // 2, 1):
        if report:
            report(f"bootstrap: only {len(kept)} of {rounds} resamples fitted; "
                   "no intervals")
        return fit
    replicates = np.asarray(kept, float)
    edge = 100 * (1 - level) / 2
    low, high = np.percentile(replicates, [edge, 100 - edge], axis=0)
    names = tuple(fit.names) + ("background",)
    fit.replicates = replicates
    fit.level = float(level)
    fit.intervals = {name: (float(low[i]), float(high[i]))
                     for i, name in enumerate(names)}
    return fit


def fit_models(t, precision=None, models: Sequence[str] = ("gauss",),
               **kwargs) -> List[Fit]:
    """Several models on the same data, best first by AIC.

    Comparing them is the point: "one structure or two?" is a question the
    likelihood answers, and by AIC rather than by likelihood alone, because
    the second Gaussian can only ever fit better.
    """
    found = [fit_profile(t, precision, model=m, **kwargs) for m in models]
    return sorted(found, key=lambda f: f.aic)


# ------------------------------------------------------------ the profiles

AXES = {"across": "position across the line",
        "along": "position along the line",
        "z": "z"}


@dataclass
class Profile:
    """One coordinate of the localizations in a line ROI, and its window."""
    key: str
    label: str
    unit: str
    values: np.ndarray
    window: Tuple[float, float]
    precision: Optional[np.ndarray] = None
    note: str = ""              # why there is no precision, when there is none


def profiles(locs: Localizations, region: Region,
             length: float = 0.0,
             z_window: Optional[Tuple[float, float]] = None) -> Dict[str, Profile]:
    """The line's coordinates for every localization inside it.

    ``length`` overrides the ROI's own length with a window of that size
    centred on it, which is what makes two ROIs on two images comparable; the
    localizations outside it are dropped from *all* of the profiles, so the
    across profile is of the same localizations as the along profile.
    """
    p0, p1, width = line_ends(region)
    x, y = positions(locs)
    along, across = project(x, y, p0, p1)
    full = float(np.hypot(*(p1 - p0)))
    span = full if length <= 0 else float(length)
    window_along = (0.5 * (full - span), 0.5 * (full + span))
    window_across = (-0.5 * width, 0.5 * width)

    inside = ((along >= window_along[0]) & (along <= window_along[1])
              & (across >= window_across[0]) & (across <= window_across[1]))
    along, across = along[inside], across[inside]
    kept = locs[inside]

    lateral = column(kept, PRECISION_FIELDS)
    precision = np.asarray(kept[lateral], float) if lateral else None
    unit = "nm" if "x_nm" in kept else "pix"
    missing = "" if lateral else ("no localization precision in the table: one "
                                  "width is fitted for every localization")

    found = {
        "across": Profile("across", AXES["across"], unit, across, window_across,
                          precision, missing),
        "along": Profile("along", AXES["along"], unit, along, window_along,
                         precision, missing),
    }
    if "z_nm" in kept:
        z = np.asarray(kept["z_nm"], float)
        finite = z[np.isfinite(z)]
        if z_window is None and finite.size:
            pad = 0.05 * (float(finite.max() - finite.min()) + 1e-9)
            z_window = (float(finite.min()) - pad, float(finite.max()) + pad)
        z_precision_name = column(kept, PRECISION_Z_FIELDS)
        found["z"] = Profile(
            "z", AXES["z"], "nm", z, z_window or (-1.0, 1.0),
            np.asarray(kept[z_precision_name], float) if z_precision_name else None,
            "" if z_precision_name else ("no axial precision in the table: one "
                                         "width is fitted for every localization"))
    return found


@dataclass
class LayerProfile:
    """One layer\'s localizations in the ROI: its profiles and its fits.

    A line is nearly always drawn over *two* channels -- the distance between
    them is the measurement -- so the unit of work here is a layer, not the
    picture.  SMAP\'s `lineprofile` loops over the visible layers for the same
    reason; this keeps the fits beside the profiles so one figure can draw
    them together.
    """
    name: str
    colour: str
    found: Dict[str, Profile]           # by axis
    fits: List[Fit] = field(default_factory=list)
    bin_size: float = 1.0
    note: str = ""
    axis: str = "along"                 # which of `found` was fitted

    @property
    def profile(self) -> Profile:
        return self.found[self.axis]


# When a layer\'s own colour cannot be read off its LUT -- a ramp that ends
# white, like `hot` -- a curve still has to be told from the next one.
FALLBACK_COLOURS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22")


def layer_colour(display, index: int = 0, taken: Sequence[str] = ()) -> str:
    """A line colour for a layer, from the LUT it is drawn with.

    Three quarters of the way up the ramp rather than at the top: `hot` ends
    white and so does `gray`, and a white curve on a white figure is no curve.
    A colour too pale, too dark or too grey to read falls back to the palette,
    and so does one another layer already has -- telling two curves apart
    matters more than matching the picture, and two layers on one LUT (which
    is what copying a layer gives) would otherwise come out the same.
    """
    from .. import lut as luts

    try:
        table = luts.get(getattr(display, "lut", "hot"),
                         getattr(display, "invert", False))
        rgb = np.asarray(table[int(0.75 * (len(table) - 1))], float)
    except Exception:
        rgb = None
    if rgb is not None:
        level = float(rgb.max() + rgb.min()) / 2
        chroma = float(rgb.max() - rgb.min())
        colour = "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)
        if 0.12 < level < 0.88 and chroma > 0.15 and colour not in taken:
            return colour
    free = [c for c in FALLBACK_COLOURS if c not in taken]
    return free[index % len(free)] if free else FALLBACK_COLOURS[
        index % len(FALLBACK_COLOURS)]


# ---------------------------------------------------------------- drawing

def draw_profile(ax, layers: Sequence["LayerProfile"]) -> None:
    """The histogram, with what was fitted to it over it.

    The histogram is the picture and the fits are of the localizations, so the
    curve is not a fit *to these bars*: it is the density scaled by how many
    localizations and how wide a bin, which is why it can sit above an empty
    bin without anything being wrong.

    One layer is drawn as it always was -- grey bars, a curve per model, the
    best one red -- because there the question is which model.  Several layers
    are drawn one colour each, with only the model that won, because there the
    question is how the layers differ and five curves per layer answers
    nothing.
    """
    alone = len(layers) == 1
    model_colours = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b")
    for layer in layers:
        profile = layer.profile
        counts, edges = histogram(profile.values, profile.window, layer.bin_size)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color="0.75" if alone else layer.colour,
               edgecolor="0.45" if alone else "none",
               alpha=1.0 if alone else 0.45, linewidth=0.4,
               label=None if alone else layer.name)
        grid = np.linspace(profile.window[0], profile.window[1], 400)
        scale = len(profile.values) * layer.bin_size
        shown = layer.fits if alone else layer.fits[:1]
        for fit, colour in zip(shown, model_colours):
            ax.plot(grid, scale * fit.curve(grid),
                    color=colour if alone else layer.colour, linewidth=1.5,
                    label=fit.label if alone else None)
    if not alone or len(layers[0].fits) > 1:
        ax.legend(fontsize=6.5, frameon=False)
    first = layers[0].profile
    ax.set_xlabel(f"{first.label} ({first.unit})")
    ax.set_ylabel("localizations")
    if alone and layers[0].fits:
        ax.set_title(layers[0].fits[0].summary, fontsize=7.5, color="0.25")
    elif not alone:
        ax.set_title("  |  ".join(f"{l.name}: {l.fits[0].summary}"
                                  for l in layers if l.fits),
                     fontsize=7.0, color="0.25")


def draw_bootstrap(figure, fit: Fit) -> None:
    """What the resamples said, one parameter per panel.

    The interval is two numbers and the distribution behind it is not always
    a bell: a distance that a quarter of the resamples put at zero, or a
    width that piles up against it, is a fit that has not measured what it
    was asked for, and that is visible here and nowhere else.
    """
    names = tuple(fit.names) + (("background",)
                                if fit.extra.get("has_background") else ())
    axes = figure.subplots(len(names), 1, squeeze=False).ravel()
    for i, (ax, name) in enumerate(zip(axes, names)):
        values = fit.replicates[:, i]
        ax.hist(values, bins=30, color="0.75", edgecolor="0.45", linewidth=0.4)
        fitted = (fit.background if name == "background"
                  else fit.values()[name])
        ax.axvline(fitted, color="#d62728", linewidth=1.5, label="fit")
        for edge in fit.intervals.get(name, ()):
            ax.axvline(edge, color="#1f77b4", linestyle="--", linewidth=1.0)
        ax.set_xlabel(name)
        ax.set_ylabel("resamples")
        if i == 0:
            ax.set_title(f"{int(round(100 * fit.level))}% intervals from "
                         f"{len(fit.replicates)} resamples",
                         fontsize=7.5, color="0.25")


def draw_scatter(figure, layers: Sequence["LayerProfile"]) -> None:
    """Where the localizations are, in the line's coordinates.

    The profile is a projection and a projection hides things -- a filament
    that leaves the ROI half way along, two structures that cross -- so the
    scatter is next to it rather than behind a menu.  One colour per layer,
    the same as the curves, so a feature in one channel can be found in the
    other.
    """
    found = layers[0].found
    keys = ["across"] + (["z"] if any("z" in l.found for l in layers) else [])
    axes = figure.subplots(len(keys), 1, squeeze=False).ravel()
    for ax, key in zip(axes, keys):
        for layer in layers:
            if key not in layer.found:
                continue
            along, other = layer.found["along"], layer.found[key]
            ax.plot(along.values, other.values, ".", markersize=2,
                    color=layer.colour if len(layers) > 1 else "#1f77b4",
                    label=layer.name if len(layers) > 1 else None)
        along = found["along"]
        other = next(l.found[key] for l in layers if key in l.found)
        ax.set_xlabel(f"{along.label} ({along.unit})")
        ax.set_ylabel(f"{other.label} ({other.unit})")
        ax.set_xlim(*along.window)
        if key == "across":
            ax.set_ylim(*other.window)
            ax.set_aspect("equal", adjustable="box")
        if len(layers) > 1:
            ax.legend(fontsize=6.5, frameon=False, markerscale=4)


# --------------------------------------------------------------- the plugin

@dataclass
class LineProfileSettings:
    axis: str = param("along", label="profile",
                      choices=(("along", "along the line"),
                               ("across", "across the line"),
                               ("z", "z")),
                      help="which coordinate is histogrammed and fitted.  "
                           "Along is what a line is usually drawn for -- the "
                           "structure laid out under it; across measures the "
                           "width of something the line crosses")
    source: str = param("layers", label="localizations",
                        choices=(("layers", "each visible layer, in its colour"),
                                 ("selection", "the selection, as one")),
                        help="a line is nearly always drawn over two channels "
                             "and the measurement is how they differ, so a "
                             "profile per layer is the usual thing; SMAP's "
                             "lineprofile does the same")
    model: str = param("gauss", label="model",
                       choices=(("gauss", "Gaussian"),
                                ("two_gauss", "two Gaussians (a distance)"),
                                ("step", "step (error function)"),
                                ("disk", "disk, seen edge-on"),
                                ("ring", "ring, seen edge-on"),
                                ("all", "all five, compared")),
                       help="what the profile is expected to be")
    method: str = param("mle", label="fit",
                        choices=(("mle", "unbinned, maximum likelihood"),
                                 ("binned", "binned, least squares")),
                        help="the unbinned fit uses the localizations "
                             "themselves, so no fitted number depends on the "
                             "bin width; with few localizations it is the "
                             "only one worth trusting")
    use_precision: bool = param(True, label="use localization precision",
                                help="each localization is a Gaussian of its "
                                     "own precision, so the fitted width is "
                                     "the structure's rather than the "
                                     "picture's")
    background: bool = param(True, label="uniform background",
                             help="a flat fraction of unspecific "
                                  "localizations; without it a Gaussian asked "
                                  "to explain them comes back too wide")
    bootstrap: int = param(0, label="bootstrap", min=0, max=5000,
                          help="0: off.  Otherwise this many resamples of the "
                               "localizations, for confidence intervals that "
                               "assume nothing about the shape of the "
                               "likelihood -- what to use below ~50 "
                               "localizations, where the fit's own error bars "
                               "are optimistic")
    confidence: float = param(95.0, label="confidence", unit="%", min=50.0,
                              max=99.9, advanced=True,
                              help="the interval the resamples are asked for")
    bin_nm: float = param(0.0, label="bin", unit="nm", min=0.0,
                          help="0: chosen from the data.  Drawing only, "
                               "unless the binned fit is chosen")
    length_nm: float = param(0.0, label="length", unit="nm", min=0.0,
                             advanced=True,
                             help="0: the ROI's own length.  Otherwise a "
                                  "window of this length centred on it, so "
                                  "two ROIs can be compared directly")


@register("Analysis/Measure/Line Profile")
class LineProfile(Plugin):
    """Profiles across a line ROI, fitted without binning them."""

    Settings = LineProfileSettings
    version = "3"        # 2: the two-Gaussian start tries peaks far apart; 3: z_err_nm

    def run(self, ctx: Context, settings: LineProfileSettings) -> Result:
        region = _line_roi(ctx)
        layers = fit_layers(ctx, settings, region)

        lines = []
        for layer in layers:
            profile = layer.profile
            head = f"{len(profile.values)} localizations in the {region}, {profile.label}"
            lines.append(head if len(layers) == 1 else f"{layer.name}: {head}")
            if layer.note:
                lines.append(f"    {layer.note}")
            for fit in layer.fits:
                lines.append(f"{fit.summary}")
                lines.append(f"    log L {fit.log_likelihood:.1f}, "
                             f"AIC {fit.aic:.1f}, BIC {fit.bic:.1f}")
                lines.extend(fit.interval_lines())
            if len(layer.fits) > 1:
                lines.append(f"best by AIC: {layer.fits[0].label} "
                             f"(by {layer.fits[1].aic - layer.fits[0].aic:.1f})")

        def profile_plot(ax) -> None:
            draw_profile(ax, layers)

        def scatter_plot(figure) -> None:
            draw_scatter(figure, layers)

        panels = 2 if any("z" in l.found for l in layers) else 1
        plots = {"scatter": Plot(draw=scatter_plot, panels=panels,
                                 size=(5.0, 2.6 * panels))}
        best = layers[0].fits[0]
        if best.intervals:
            spread = len(best.names) + (1 if settings.background else 0)
            plots["bootstrap"] = Plot(
                draw=lambda figure: draw_bootstrap(figure, best),
                panels=spread, size=(5.0, 1.8 * spread))

        def per_layer(what):
            found = {l.name: what(l) for l in layers}
            return found[layers[0].name] if len(layers) == 1 else found

        return Result(
            text="\n".join(lines), settings=settings,
            # "layer_profiles", not "layers": `Session.apply` reads
            # data["layers"] as layer set-ups to apply (`Chain/Layers`), and
            # every Run of this plugin crashed there on a list of profiles
            data={"layer_profiles": layers,
                  "fits": per_layer(lambda l: {f.model: f for f in l.fits}),
                  "values": per_layer(lambda l: {f.model: f.values() for f in l.fits}),
                  "errors": per_layer(lambda l: {f.model: f.uncertainties()
                                                 for f in l.fits}),
                  "intervals": per_layer(lambda l: {f.model: f.intervals
                                                    for f in l.fits if f.intervals}),
                  "profiles": per_layer(lambda l: l.found),
                  "bin": layers[0].bin_size,
                  "n": sum(len(l.profile.values) for l in layers)},
            plot=profile_plot, plots=plots)

    # cheap enough to redo while the line is dragged; see `Plugin.live`
    live = True

    def preview(self, ctx: Context, settings: LineProfileSettings) -> Result:
        """The same measurement, not applied to anything.

        `run` changes no localizations either, so this is `run` -- what the
        preview is for here is the *live* tick, which wants a result without a
        line in the history for every position the line was dragged through.
        """
        return self.run(ctx, settings)


def fit_layers(ctx: Context, settings: LineProfileSettings,
               region: Region) -> List[LayerProfile]:
    """One `LayerProfile` per layer asked for, fitted.

    Module level and taking a context so that the loop over layers, which is
    the only thing the session is needed for, is in one place and the fitting
    below it needs no session at all.
    """
    z_window = _z_window(ctx)
    groups = _groups(ctx, settings)
    layers: List[LayerProfile] = []
    for name, colour, locs in groups:
        found = profiles(locs, region, length=settings.length_nm, z_window=z_window)
        if settings.axis not in found:
            raise ValueError(f"no {settings.axis} profile: the table has "
                             f"{', '.join(sorted(locs.keys()))}")
        profile = found[settings.axis]
        if len(profile.values) < MIN_FOR_FIT:
            if len(groups) == 1:
                raise ValueError(f"{len(profile.values)} localizations in the ROI: "
                                 f"too few for a profile fit (at least {MIN_FOR_FIT})")
            ctx.report(f"{name}: {len(profile.values)} localizations in the ROI, "
                       f"too few to fit -- skipped")
            continue
        note = ""
        if len(profile.values) < THIN:
            note = (f"only {len(profile.values)} localizations: the fitted "
                    "values stand, their error bars are optimistic")
            ctx.report(f"{name}: {note}" if len(groups) > 1 else note)
        if profile.note and settings.use_precision:
            note = f"{note}; {profile.note}" if note else profile.note

        precision = profile.precision if settings.use_precision else None
        bin_size = settings.bin_nm or auto_bin(profile.values, profile.window)
        models = (tuple(MODELS) if settings.model == "all" else (settings.model,))
        ctx.report(f"fitting {len(profile.values)} localizations "
                   f"({'unbinned' if settings.method == 'mle' else 'binned'})"
                   + (f" -- {name}" if len(groups) > 1 else ""))
        fits = fit_models(profile.values, precision, models=models,
                          window=profile.window, method=settings.method,
                          background=settings.background, bin_size=bin_size)

        # only the best model is resampled: a few hundred refits are worth
        # spending on the answer, not on the models that lost
        if settings.bootstrap > 0:
            bootstrap(fits[0], profile.values, precision,
                      rounds=int(settings.bootstrap),
                      level=float(np.clip(settings.confidence, 50.0, 99.9)) / 100,
                      report=ctx.report)
        elif len(profile.values) < THIN:
            ctx.report("switch the bootstrap on for intervals that do not "
                       "assume a large sample")
        layers.append(LayerProfile(name=name, colour=colour, found=found, fits=fits,
                                   bin_size=bin_size, note=note, axis=settings.axis))
    if not layers:
        raise ValueError("no layer has enough localizations in the ROI for a "
                         f"profile fit (at least {MIN_FOR_FIT} each)")
    return layers


def _groups(ctx: Context, settings: LineProfileSettings):
    """``(name, colour, locs)`` per layer to fit, or one for the selection."""
    session = ctx.session
    if settings.source == "layers" and session is not None:
        found = []
        for i, layer in enumerate(getattr(session, "layers", [])):
            if layer.is_image or not layer.visible:
                continue
            locs = session.selection(i).apply(ctx.locs)
            taken = [colour for _, colour, _ in found]
            found.append((layer.name,
                          layer_colour(layer.get_display(), len(found), taken),
                          locs))
        if found:
            return found
        if getattr(session, "layers", None):
            ctx.report("no visible localization layer: the selection is used instead")
    ctx.selection.require(MIN_FOR_FIT, ctx.report, "a profile fit")
    return [(ctx.selection.name or "selection", FALLBACK_COLOURS[0],
             ctx.selection.apply(ctx.locs))]


def _line_roi(ctx: Context) -> Region:
    """The line ROI this plugin measures, refused clearly when there is none."""
    roi = getattr(ctx.session, "roi", None)
    if not isinstance(roi, Region):
        found = getattr(ctx.selection, "roi", None)
        roi = found if isinstance(found, Region) else None
    if roi is None:
        raise ValueError("no ROI: draw a line ROI in the render window")
    if roi.kind != "line":
        raise ValueError(f"the ROI is a {roi.kind}: a line ROI is what has a "
                         "direction to take a profile along")
    if roi.width <= 0:
        raise ValueError("the line ROI has no width: widen it to take in "
                         "localizations")
    return roi


def _z_window(ctx: Context) -> Optional[Tuple[float, float]]:
    """Where z could have been: the layer's own z range, when there is one.

    Taking it from the data instead would put the window's edges *at* the
    outermost localizations, and the likelihood would read that as a structure
    that stops exactly there.
    """
    session = ctx.session
    if session is None or not hasattr(session, "z_range"):
        return None
    try:
        low, high = session.z_range()
    except Exception:
        return None
    return (float(low), float(high)) if high > low else None
