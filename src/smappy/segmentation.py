"""Finding structures and measuring each candidate: the building blocks of
*Segment regions*, on arrays, so a script, a notebook or a user's own
segmenter gets them without a session or a window.

`docs/smap_porting/wave6_design.md` has the decision.  A method is a model of
the structure, a preset is a set of values for it, and which candidates are
kept is one condition, *keep*, on what was measured on each of them:

* **the circle template** -- a ring or a disc, at one radius or over a range
  of radii.  The density image is filtered with a zero-sum template at each
  radius (`template`), the best radius wins at each position
  (`filter_radii`), the maxima are the candidates (`peaks`) and each is
  measured on its own localizations (`measure_ring`, `measure_disc`);
* **beads** -- the bright, long-lived tracks of `fiducials.find_beads`,
  measured by `measure_beads`;
* **like examples** -- Picasso's *pick similar* (jungmannlab/picasso,
  MIT, `postprocess.pick_similar`): the examples are what ROIs hold
  (`measure_rows`), a circle is moved onto the localizations from every
  point of a lattice over the field (`similar_candidates`), each measured
  the same way (`measure_at`), and kept where its values
  lie within the examples' mean plus or minus a few standard deviations
  (`similarity_window`, written as a *keep* condition by
  `window_condition`).

Every candidate gets the same shared values -- ``n_locs``, ``radius_nm``,
``roundness``, ``psf_nm``, ``touching`` -- and its model's own (`VALUES`).
*keep* is parsed by `roi_manager.order.condition`, *Filter ROIs*' parser,
so the condition that keeps a candidate is written as the one that filters
the ROIs afterwards, and nothing in it is run (`judge`).

Decisions the design left open, and why:

* **The responses are counts of localizations.**  Each template is a
  shape of weight up to one per pixel minus a surround of the same total
  weight, so a uniform background gives zero.  A disc gives the
  localizations inside it less those in the annulus around it; a ring --
  whose surround is a disc that takes in the ring too, as the ring filter's
  did -- the localizations on it less the disc's mean times the ring's
  weight, about 0.4 of those on the ring at the pore's size.  Either way a
  structure's response grows with its localizations, not with the pixel
  size, so responses at different radii compare.  It runs on the square
  root of the density, as the ring filter did (SMAP's: a molecule that
  blinks fifty times in one place otherwise draws a ring of response around
  itself); where a pixel holds at most one localization, which at 10 nm is
  nearly everywhere, that changes nothing.
* **The best radius is the one with the best signal to noise**, the
  response over the square root of the template's weight, not the largest
  response.  A disc's response stops growing once the template holds the
  whole disc -- every larger radius answers the same count -- so the largest
  would be any of them; divided by the square root of the area it peaks at
  the disc's own radius, as a matched filter does.  For a ring the response
  itself peaks at its radius, and the division only nudges it.
* **Two candidates overlap too much** when their circles overlap by more
  than half the smaller radius: ``d < r1 + r2 - min(r1, r2) / 2``, with the
  template's radius at each (`too_close`).  For two pores of 55 nm that is
  closer than 82.5 nm.  The stronger stays; an ROI already in the file
  counts as a circle of the candidate's own radius.
* **A disc's centre is the mean of its localizations** in the window, twice,
  the second time around the first answer -- not a circle fit with the
  radius fixed, which on a filled disc has nothing to hold on to but the
  edge.  A ring's is the circle fit, as it was.
* **A disc's radius** is ``sqrt(2)`` times the RMS distance from the centre,
  less the localizations' own spread: ``sqrt(2 (<d^2> - <sigma^2>))``.  The
  precision adds ``sigma^2`` to the mean square distance, which at 30 nm and
  a precision of 10 nm would be a radius 10 % too large.  It is taken three
  times, the last two over the localizations within the radius plus twice
  their precision: one background localization at the window's edge, which
  counts with its distance squared, made a 30 nm disc 16 % too large.  (The
  median of ``d^2``, also ``R^2 / 2`` for a uniform disc, ignores it but
  scattered by 20 % on a hundred localizations.)
* **A localization belongs to one candidate.**  Candidates are measured
  strongest first, and each claims its localizations -- a ring those out to
  ``R + 2 w``, as far as the filter's disc reaches, a disc those inside its
  radius; a weaker candidate's fits and counts are on what is left (a
  ring's far test still sees everything in its window).  The window of a
  candidate beside a pore takes in an arc of the pore, which otherwise
  passed as a ring of its own: next to 6 of 141 simulated pores at about
  100 nm, and at 500 photons, where a pore's localizations spread well past
  its band, next to a third of them.  The ring template's *separation* of
  120 nm kept them out, and two touching pores with them.  A candidate
  claims whether *keep* passes it or not: junk that claimed nothing left
  its localizations to the candidates beside it, which then passed (it
  nearly doubled the junk on `studies/npc_le`'s simulations).  An ROI
  already in the file claims as a candidate of the smallest radius would
  there, so a second run that does not replace adds no twins either.
* **A ring's window reaches as far as its template**, ``max(1.5 R, R + 3 w)``
  -- 100 nm for a pore, as the ring template's was -- rather than 1.5 R.
  A pore found 50 nm off its centre, as sparse pores at 500 photons
  sometimes are, fails the far test only if the window takes in the far
  side of the pore; at 82.5 nm it did not, and the junk among the sites
  kept doubled.  With both, the *NPC* preset finds on `studies/npc_le`'s
  simulations with junk what the ring template found: over 18 conditions
  the sites on a pore and the junk within 1 % and 2 % of its, and the
  fitted labelling efficiency within 0.017.
* **A candidate needs 3 localizations in its window** (`MIN_LOCS`), the
  fewest a circle is fitted to, and a positive response.  Everything else is
  *keep*'s.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SHAPES = ("ring", "disc")
MAX_PIXELS = 16_000_000
#: a candidate is measured on its localizations within this many radii
WINDOW = 1.5
#: over a range, the radii are spaced by this fraction of the radius or the
#: ring width, whichever is larger
RADIUS_STEP = 0.15
#: the fewest localizations a candidate has in its window
MIN_LOCS = 3
#: fixed-radius fits of a ring's centre: the second is on the window around
#: the first answer, which a filter peak a pixel or two off can need
FIT_PASSES = 2
#: a disc's radius from the localizations in the window, then twice more from
#: those within its edge (`measure_disc`)
DISC_PASSES = 3
# the far test, as the ring template had it: a localization further than 3
# of its precisions (widened by 5 nm of tilt and label) from the fitted ring
# is far, which on a pore happens 0.3 % of the time; far_p is the chance of
# as many far ones if 1 % may be
FAR_SIGMAS = 3.0
MAX_FAR = 0.01
RING_SPREAD_NM = 5.0

SHARED = ("n_locs", "radius_nm", "roundness", "psf_nm", "touching")
VALUES = {"ring": SHARED + ("n_ring", "n_inside", "n_outside", "far_p"),
          "disc": SHARED + ("centre_ratio",),
          "beads": SHARED + ("frames_fraction", "photons", "spread_nm"),
          "similar": SHARED + ("rmsd_nm",)}
#: what *like examples* compares a candidate with the examples on, unless told
#: otherwise: Picasso's two numbers
SIMILAR_COMPARED = ("n_locs", "rmsd_nm")
#: the mean shift stops when a circle moves less than this, or after so many
#: steps (Picasso: 0.001 camera pixels, 500 steps)
SHIFT_TOL_NM = 0.1
SHIFT_STEPS = 100


# ------------------------------------------------------------------ geometry

def fit_circle(x, y, center, radius: Optional[float] = None,
               scale: float = 15.0, iterations: int = 30) -> Tuple[float, float, float]:
    """A circle through the points: ``(x0, y0, r)``.

    With ``radius`` given only the centre is fitted, which is robust even on
    a pore with two corners; without it the radius is fitted too, which is
    what says whether the structure is a ring of the expected size.  The loss
    is soft-L1 with ``scale``, so points further than about ``scale`` from the
    circle count with their distance rather than its square.

    Gauss-Newton with reweighting (the soft-L1 weight is
    ``1 / sqrt(1 + (d / scale)^2)``) rather than scipy's ``least_squares``:
    the same minimum, twenty times faster, and the segmenter fits thousands.
    Each step solves the 2 x 2 or 3 x 3 normal equations, three times
    faster than a least-squares solve of the whole Jacobian.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(center, dtype=float).copy()
    free = radius is None
    if len(x) < (3 if free else 2):
        return float(p[0]), float(p[1]), float(np.nan if free else radius)
    r = float(np.median(np.hypot(x - p[0], y - p[1]))) if free else float(radius)
    for _ in range(iterations):
        dx, dy = x - p[0], y - p[1]
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        residual = d - r
        w = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
        J = np.empty((3 if free else 2, len(d)))
        J[0], J[1] = -dx / d, -dy / d
        if free:
            J[2] = -1.0
        Jw = J * w
        try:
            step = np.linalg.solve(Jw @ J.T, -(Jw @ residual))
        except np.linalg.LinAlgError:          # all the points in one place
            break
        p += step[:2]
        if free:
            r = abs(r + step[2])
        if np.max(np.abs(step)) < 1e-3:
            break
    return float(p[0]), float(p[1]), float(r)


def roundness(x, y) -> float:
    """The SD along the short axis over the SD along the long axis, from the
    second moments (Cluster shape's `principal_axes`): 1 for a round
    structure, b/a for an ellipse of half-axes a and b, 0 for a line.  NaN
    below 3 points."""
    from .plugins.cluster_shape import principal_axes
    if len(x) < 3:
        return float("nan")
    long, short, _ = principal_axes(np.asarray(x, float), np.asarray(y, float))
    return float(short / long) if long > 0 else float("nan")


# ------------------------------------------------------------------ filtering

def ring_kernel(radius_nm: float, width_nm: float, pixel_nm: float) -> np.ndarray:
    """A Gaussian ring minus a disc of the same weight: zero sum.

    The ring has radius ``radius_nm``, a Gaussian profile of width
    ``width_nm`` and a peak of 1; the disc reaches to ``radius + 2 width``.
    Convolved with a density it gives the localizations on the ring less the
    disc's mean density times the ring's weight: a uniform background gives
    zero, a filled blob of the ring's size about zero or less, and a ring its
    localizations at its centre.
    """
    extent = radius_nm + 3 * width_nm
    n = int(np.ceil(extent / pixel_nm))
    grid = np.arange(-n, n + 1) * pixel_nm
    r = np.hypot(*np.meshgrid(grid, grid))
    ring = np.exp(-(r - radius_nm) ** 2 / (2 * width_nm ** 2))
    disc = (r <= radius_nm + 2 * width_nm).astype(float)
    return ring - disc * ring.sum() / disc.sum()


def disc_kernel(radius_nm: float, pixel_nm: float) -> np.ndarray:
    """A filled disc minus the annulus around it out to ``sqrt(2)`` times
    the radius (the same area), at the same weight: zero sum.  Convolved with
    a density it gives the localizations in the disc less those in the
    annulus."""
    outer = radius_nm * np.sqrt(2.0)
    n = int(np.ceil(outer / pixel_nm)) + 1
    grid = np.arange(-n, n + 1) * pixel_nm
    r = np.hypot(*np.meshgrid(grid, grid))
    disc = (r <= radius_nm).astype(float)
    if not disc.any():                   # a radius below half a pixel
        disc[n, n] = 1.0
    annulus = ((r > radius_nm) & (r <= outer)).astype(float)
    if not annulus.any():
        annulus = ((r > radius_nm) & (r <= radius_nm + 1.5 * pixel_nm)).astype(float)
    return disc - annulus * disc.sum() / annulus.sum()


def template(shape: str, radius_nm: float, width_nm: float, pixel_nm: float
             ) -> Tuple[np.ndarray, float]:
    """The zero-sum kernel of a shape and its weight -- the sum of its
    positive part, the pixels a structure's localizations are counted on."""
    if shape == "ring":
        kernel = ring_kernel(radius_nm, width_nm, pixel_nm)
    elif shape == "disc":
        kernel = disc_kernel(radius_nm, pixel_nm)
    else:
        raise ValueError(f"no shape {shape!r}: one of {', '.join(SHAPES)}")
    return kernel, float(kernel[kernel > 0].sum())


def template_extent(shape: str, radius_nm: float, width_nm: float) -> float:
    """How far a template reaches from its centre, nm."""
    return radius_nm + 3 * width_nm if shape == "ring" else radius_nm * np.sqrt(2.0)


def radii(r_min: float, r_max: float, step_nm: float = 0.0) -> List[float]:
    """The radii a range is filtered at: from ``r_min`` to ``r_max``, each the
    last plus 15 % of it or ``step_nm`` (the ring width), whichever is
    larger; one radius when the two are equal."""
    if not 0 < r_min <= r_max:
        raise ValueError(f"the radii must be positive and the min not above the "
                         f"max, not {r_min:g} and {r_max:g}")
    out = [float(r_min)]
    while out[-1] < r_max - 1e-9:
        out.append(min(float(r_max), out[-1] + max(step_nm, RADIUS_STEP * out[-1])))
    return out


def density_image(xy: np.ndarray, pixel_nm: float, pad_nm: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The square root of the localizations per pixel, and the image's origin
    in nm.  The image covers the positions with ``pad_nm`` on every side."""
    pixel = float(pixel_nm)
    origin = np.floor((xy.min(axis=0) - pad_nm - pixel) / pixel) * pixel
    shape = np.ceil((xy.max(axis=0) + pad_nm + pixel - origin) / pixel).astype(int) + 1
    if int(shape[0]) * int(shape[1]) > MAX_PIXELS:
        raise ValueError("the density image exceeds 16 million pixels; "
                         "increase the bin size")
    bins = np.floor((xy - origin) / pixel).astype(int)
    density = np.zeros(tuple(shape[::-1]), dtype=float)
    np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
    return np.sqrt(density), origin


def filter_radii(image: np.ndarray, pixel_nm: float, shape: str,
                 radii_nm: Sequence[float], width_nm: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(score, response, radius)`` per pixel: the template at each radius,
    the best radius winning -- the one whose response over the square root
    of its weight (the signal to noise of a count) is largest.  ``response``
    is that radius's, a number of localizations above the background."""
    from scipy.signal import fftconvolve
    best_score = np.full(image.shape, -np.inf)
    response = np.zeros(image.shape)
    radius = np.zeros(image.shape)
    for r in radii_nm:
        kernel, weight = template(shape, r, width_nm, pixel_nm)
        found = fftconvolve(image, kernel, mode="same")
        score = found / np.sqrt(weight)
        better = score > best_score
        best_score[better] = score[better]
        response[better] = found[better]
        radius[better] = r
    return best_score, response, radius


def peaks(score: np.ndarray, response: np.ndarray, radius: np.ndarray, origin,
          pixel_nm: float, size_nm: float) -> List[Tuple[float, float, float, np.ndarray]]:
    """Local maxima of the score with a positive response, strongest first:
    ``(score, response, radius, centre in nm)``.  A maximum is the largest
    in a square of about ``size_nm`` (the smallest radius) on a side; the
    overlap rule (`too_close`) does the rest."""
    from scipy.ndimage import maximum_filter
    size = max(3, 2 * int(round(size_nm / pixel_nm / 2)) + 1)
    maxima = ((score == maximum_filter(score, size=size, mode="constant"))
              & (response > 0) & np.isfinite(score))
    yy, xx = np.nonzero(maxima)
    order = np.lexsort((xx, yy, -score[yy, xx]))
    return [(float(score[yy[i], xx[i]]), float(response[yy[i], xx[i]]),
             float(radius[yy[i], xx[i]]),
             np.asarray(origin) + (np.array([xx[i], yy[i]]) + 0.5) * pixel_nm)
            for i in order]


def too_close(d, r1, r2):
    """Whether two circles at distance ``d`` overlap by more than half the
    smaller radius."""
    return d < r1 + r2 - 0.5 * np.minimum(r1, r2)


class Accepted:
    """Circles accepted so far, asked "does this one overlap any too much?";
    an existing ROI is a circle of the asker's radius (NaN here)."""

    def __init__(self, centres: Sequence = ()):
        self.xy = np.zeros((max(len(centres), 64), 2))
        self.r = np.full(len(self.xy), np.nan)
        self.n = 0
        for c in centres:
            self.add(c, np.nan)

    def add(self, c, r: float) -> None:
        if self.n == len(self.xy):
            self.xy = np.concatenate([self.xy, np.zeros_like(self.xy)])
            self.r = np.concatenate([self.r, np.full(len(self.r), np.nan)])
        self.xy[self.n] = c
        self.r[self.n] = r
        self.n += 1

    def blocks(self, c, r: float) -> bool:
        if not self.n:
            return False
        d = np.hypot(*(self.xy[:self.n] - np.asarray(c, dtype=float)).T)
        other = np.where(np.isfinite(self.r[:self.n]), self.r[:self.n], r)
        return bool(np.any(too_close(d, other, r)))


# ------------------------------------------------------------------ measuring

def _shared(x, y, psf, near, precise, radius: float) -> Dict[str, float]:
    use = near & precise if np.sum(near & precise) >= 3 else near
    return {"n_locs": int(near.sum()), "radius_nm": float(radius),
            "roundness": roundness(x[use], y[use]),
            "psf_nm": float(np.mean(psf[near])) if psf is not None and near.any()
            else float("nan"),
            "touching": 0}


def measure_ring(x, y, center, radius_nm: float, width_nm: float,
                 sigma=None, psf=None, precision_nm: float = 20.0,
                 claimed=None) -> Dict[str, Any]:
    """One ring candidate, at the template's ``radius_nm``: its centre and
    values, on the localizations within ``max(1.5 R, R + 3 w)``, as far as
    the template reaches.

    ``claimed`` (a mask) are the localizations a stronger candidate claimed
    (`find_circles`): the fits and the counts leave them out, the far test
    does not.  The centre is fitted with the radius fixed, twice, the second
    time on the window around the first answer, using the localizations
    better than ``precision_nm`` (all of them when the table has no
    precision, or fewer than 3 are).  On the same localizations the radius
    is fitted free (``radius_nm``).  Each precise localization in the
    window, claimed or not, then has its distance from that circle taken in
    units of its own precision, widened by 5 nm:
    ``z = (r - radius) / sqrt(sigma^2 + 5^2)``; ``far_p`` is the binomial
    chance of as many with ``|z| > 3`` if 1 % may be -- low for a filled
    structure, whether the centre is on it or beside it, and for an arc
    beside a pore, whose localizations lie far from it.  ``n_ring``,
    ``n_inside`` and ``n_outside`` count the localizations in the window in
    the band ``R +- w``, inside it and outside it.  ``claim`` is what the
    candidate claims if it is kept: its localizations out to ``R + 2 w``, as
    far as the filter's disc reached.
    """
    from scipy.stats import binom
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R, w = float(radius_nm), float(width_nm)
    window = max(WINDOW * R, R + 3 * w)
    own = np.ones(len(x), bool) if claimed is None else ~np.asarray(claimed, bool)
    precise = (np.ones(len(x), bool) if sigma is None
               else np.asarray(sigma, float) < precision_nm)
    c = np.asarray(center, dtype=float)
    for _ in range(FIT_PASSES):
        near = (np.hypot(x - c[0], y - c[1]) <= window) & own
        on = near & precise if np.sum(near & precise) >= 3 else near
        c = np.array(fit_circle(x[on], y[on], c, radius=R, scale=w)[:2])
    r = np.hypot(x - c[0], y - c[1])
    near = (r <= window) & own
    use = near & precise
    fitted = fit_circle(x[use], y[use], c, scale=w)[2] if use.sum() >= 3 else np.nan
    p_far = 1.0
    judged = (r <= window) & precise
    n = int(judged.sum())
    if sigma is not None and use.sum() >= 3 and np.isfinite(fitted):
        z = (r[judged] - fitted) / np.sqrt(np.asarray(sigma, float)[judged] ** 2
                                           + RING_SPREAD_NM ** 2)
        far = int(np.sum(np.abs(z) > FAR_SIGMAS))
        p_far = float(binom.sf(far - 1, n, MAX_FAR)) if far else 1.0
    values = _shared(x, y, psf, near, precise, fitted)
    values.update({"n_ring": int(np.sum(near & (r >= R - w) & (r <= R + w))),
                   "n_inside": int(np.sum(near & (r < R - w))),
                   "n_outside": int(np.sum(near & (r > R + w))),
                   "far_p": p_far})
    return {"center": c.tolist(), "values": values, "claim": near & (r <= R + 2 * w)}


def measure_disc(x, y, center, radius_nm: float, sigma=None, psf=None,
                 precision_nm: float = 20.0) -> Dict[str, Any]:
    """One disc candidate, at the template's ``radius_nm``: its centre (the
    mean position in the window of 1.5 R, twice) and values.

    ``radius_nm`` is ``sqrt(2 (<d^2> - <sigma^2>))`` over the precise
    localizations in the window, the radius of a uniform disc with that
    spread, taken again twice over those within the radius plus twice their
    spread (a tenth of the radius without a precision); ``centre_ratio`` is the density inside half that radius over the
    density in its outer half -- 1 for a uniform disc, near 0 for a ring.
    ``claim``: a disc claims what lies inside its radius.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    window = WINDOW * float(radius_nm)
    precise = (np.ones(len(x), bool) if sigma is None
               else np.asarray(sigma, float) < precision_nm)
    c = np.asarray(center, dtype=float)
    for _ in range(FIT_PASSES):
        near = np.hypot(x - c[0], y - c[1]) <= window
        if near.any():
            c = np.array([x[near].mean(), y[near].mean()])
    r = np.hypot(x - c[0], y - c[1])
    near = r <= window
    use = near & precise if np.sum(near & precise) >= 3 else near
    radius = float("nan")
    if use.sum() >= 3:
        blur = np.mean(np.asarray(sigma, float)[use] ** 2) if sigma is not None else 0.0
        inside = use
        for _ in range(DISC_PASSES):
            if inside.sum() < 3:
                break
            radius = float(np.sqrt(2 * max(np.mean(r[inside] ** 2) - blur, 0.0)))
            # a background localization at the window's edge weighs its
            # distance squared: the next pass leaves out what lies beyond
            # the edge by more than twice the spread
            inside = use & (r <= radius + 2 * (np.sqrt(blur) if blur else 0.1 * radius))
    values = _shared(x, y, psf, near, precise, radius)
    edge = radius if np.isfinite(radius) and radius > 0 else float(radius_nm)
    inner = int(np.sum(r < edge / 2))
    outer = int(np.sum((r >= edge / 2) & (r <= edge)))
    # the inner circle is a quarter of the area, the outer half three
    values["centre_ratio"] = 3.0 * inner / outer if outer else float("nan")
    return {"center": c.tolist(), "values": values, "claim": near & (r <= edge)}


def positions(locs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
                             Optional[np.ndarray]]:
    """``(xy, finite, sigma, psf)`` of a table: the finite positions, which
    rows they are, and their precision and PSF width where the table has
    them (``xy_err_nm`` or ``x_err_nm``; ``sigma_nm``)."""
    xy = np.column_stack((np.asarray(locs["x_nm"], dtype=float),
                          np.asarray(locs["y_nm"], dtype=float)))
    finite = np.isfinite(xy).all(axis=1)
    column = next((n for n in ("xy_err_nm", "x_err_nm") if n in locs), None)
    sigma = None if column is None else np.asarray(locs[column], float)[finite]
    psf = np.asarray(locs["sigma_nm"], float)[finite] if "sigma_nm" in locs else None
    return xy[finite], finite, sigma, psf


def find_circles(locs, shape: str, min_radius_nm: float, max_radius_nm: float,
                 ring_width_nm: float = 15.0, bin_nm: float = 10.0,
                 precision_nm: float = 20.0, existing: Sequence = ()
                 ) -> List[Dict[str, Any]]:
    """Every candidate of a ring or a disc between two radii, measured.

    One dict per candidate, strongest first: ``center``, ``values``
    (`VALUES[shape]`), the filter's ``response`` and ``score`` and the
    template's radius, ``template_nm``.  A candidate that overlaps a
    stronger one, or an ``existing`` centre, too much (`too_close`) is left
    out -- checked at the filter's maximum and again after the fit has moved
    it -- and so is one with fewer than 3 localizations in its window.  Each
    one claims its localizations (``claim`` of `measure_ring`,
    `measure_disc`), and a weaker one is measured on what is left -- whether
    *keep* passes it or not: junk that claims nothing leaves its
    localizations to the candidates beside it, which then pass (letting only
    those that pass claim nearly doubled the junk on `studies/npc_le`'s
    simulations).  ``touching`` is 0 and ``use`` unset until `judge`.
    """
    for name, value in (("bin", bin_nm), ("ring width", ring_width_nm)):
        if not value > 0:
            raise ValueError(f"the {name} must be positive")
    from scipy.spatial import cKDTree
    xy, _, sigma, psf = positions(locs)
    if len(xy) < MIN_LOCS:
        return []
    width = ring_width_nm if shape == "ring" else 0.0
    rr = radii(min_radius_nm, max_radius_nm, width)
    image, origin = density_image(xy, bin_nm, template_extent(shape, rr[-1], width))
    score, response, radius = filter_radii(image, bin_nm, shape, rr, width)
    found_peaks = peaks(score, response, radius, origin, bin_nm, rr[0])
    if not found_peaks:
        return []
    tree = cKDTree(xy)
    accepted = Accepted(existing)
    claimed = np.zeros(len(xy), dtype=bool)
    # an ROI already in the file claims what a candidate of the smallest
    # radius would there
    for c in existing:
        r = np.hypot(*(xy - np.asarray(c, dtype=float)).T)
        claimed |= r <= (rr[0] + 2 * width if shape == "ring" else rr[0])
    found = []
    for s, resp, r0, centre in found_peaks:
        if accepted.blocks(centre, r0):
            continue
        # the window, and room for the centre to move as it is fitted
        window_nm = max(WINDOW * r0, r0 + 3 * width) if shape == "ring" else WINDOW * r0
        reach = window_nm + (2 * width if shape == "ring" else 0.5 * r0)
        window = np.asarray(tree.query_ball_point(centre, reach), dtype=int)
        # what a stronger candidate claimed is not this one's, and with
        # fewer than MIN_LOCS left it is not measured at all -- most maxima
        # beside a structure, which would otherwise cost a fit each
        if np.sum(~claimed[window]) < MIN_LOCS:
            continue
        if shape == "disc":
            window = window[~claimed[window]]
        sub = (lambda a: None if a is None else a[window])
        if shape == "ring":
            # the far test sees the claimed ones too (`measure_ring`)
            site = measure_ring(xy[window, 0], xy[window, 1], centre, r0, width,
                                sub(sigma), sub(psf), precision_nm, claimed[window])
        else:
            site = measure_disc(xy[window, 0], xy[window, 1], centre, r0,
                                sub(sigma), sub(psf), precision_nm)
        if site["values"]["n_locs"] < MIN_LOCS or accepted.blocks(site["center"], r0):
            continue
        accepted.add(site["center"], r0)
        claimed[window[site.pop("claim")]] = True
        site.update({"response": resp, "score": s, "template_nm": r0})
        found.append(site)
    return found


def measure_beads(locs, min_photons: float = 5000.0, min_fraction: float = 0.25,
                  max_jump_nm: float = 150.0, memory: int = 50,
                  n_frames: Optional[int] = None, select=None) -> List[Dict[str, Any]]:
    """Every bead of `fiducials.find_beads`, as a candidate: its centre the
    median position of its localizations, ``rows`` which rows of ``locs``
    they are, and its values -- ``frames_fraction`` (the frames it was seen
    in over all frames), ``photons`` (median), ``spread_nm`` (the SD of its
    positions, the mean of x's and y's variances under the root, drift
    included) and the shared ones, its ``radius_nm`` as a disc's.  The most
    frames first."""
    from .fiducials import find_beads
    for name in ("x_nm", "y_nm", "frame", "photons"):
        if name not in locs:
            raise ValueError(f"beads need a {name} attribute; the table has "
                             + ", ".join(locs.columns))
    x = np.asarray(locs["x_nm"], float)
    y = np.asarray(locs["y_nm"], float)
    frame = np.asarray(locs["frame"])
    photons = np.asarray(locs["photons"], float)
    z = np.asarray(locs["z_nm"], float) if "z_nm" in locs else None
    if n_frames is None:
        n_frames = int(frame.max()) + 1 if len(frame) else 0
    beads = find_beads(x, y, frame, photons, z, n_frames=n_frames,
                       min_photons=min_photons, min_fraction=min_fraction,
                       max_jump_nm=max_jump_nm, memory=memory, select=select)
    psf = np.asarray(locs["sigma_nm"], float) if "sigma_nm" in locs else None
    found = []
    for k in range(len(beads.count)):
        rows = np.flatnonzero(beads.bead == k)
        bx, by = x[rows], y[rows]
        c = np.array([np.median(bx), np.median(by)])
        d2 = (bx - c[0]) ** 2 + (by - c[1]) ** 2
        values = {"n_locs": int(len(rows)),
                  "radius_nm": float(np.sqrt(2 * np.mean(d2))),
                  "roundness": roundness(bx, by),
                  "psf_nm": float(np.mean(psf[rows])) if psf is not None
                  else float("nan"),
                  "touching": 0,
                  "frames_fraction": float(len(np.unique(frame[rows])) / max(n_frames, 1)),
                  "photons": float(np.median(photons[rows])),
                  "spread_nm": float(np.sqrt((np.var(bx) + np.var(by)) / 2))}
        found.append({"center": c.tolist(), "values": values, "rows": rows})
    found.sort(key=lambda s: -s["values"]["n_locs"])
    return found


# ------------------------------------------------------------------ like examples

#: the shapes a candidate of *like examples* can have, the ROI manager's two,
#: as the Minkowski norm of a circle (p=2) or a square (p=inf) of half-width R
SIMILAR_SHAPES = {"circle": 2.0, "square": np.inf}


def _norm(shape: str) -> float:
    if shape not in SIMILAR_SHAPES:
        raise ValueError(f"no shape {shape!r}: one of {', '.join(SIMILAR_SHAPES)}")
    return SIMILAR_SHAPES[shape]


def mean_shift(xy: np.ndarray, centres, radius_nm: float, tree=None,
               steps: int = SHIFT_STEPS, tol_nm: float = SHIFT_TOL_NM,
               shape: str = "circle") -> np.ndarray:
    """Each circle (or square) of half-width ``radius_nm`` moved to the
    centre of mass of the localizations inside it, again and again until it
    moves less than ``tol_nm``: it climbs onto the nearest structure.  All
    at once; one that holds nothing stays where it is."""
    from itertools import chain
    from scipy.spatial import cKDTree
    p = _norm(shape)
    tree = cKDTree(xy) if tree is None else tree
    c = np.array(centres, dtype=float).reshape(-1, 2)
    moving = np.arange(len(c))
    for _ in range(steps):
        if not len(moving):
            break
        lists = tree.query_ball_point(c[moving], radius_nm, p=p)
        n = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(n.sum()))
        owner = np.repeat(np.arange(len(moving)), n)
        held = n > 0
        new = c[moving].copy()
        for k in (0, 1):
            total = np.bincount(owner, xy[flat, k], minlength=len(moving))
            new[held, k] = total[held] / n[held]
        moved = np.hypot(*(new - c[moving]).T)
        c[moving] = new
        moving = moving[held & (moved > tol_nm)]
    return c


def _values_of(xy, sigma, psf, precise, rows) -> Dict[str, float]:
    """The values of one set of localizations (``rows`` of ``xy``), about
    their own centre of mass: `VALUES["similar"]`."""
    near = np.zeros(len(xy), bool)
    near[rows] = True
    values = _shared(xy[:, 0], xy[:, 1], psf, near, precise, float("nan"))
    values["rmsd_nm"] = float("nan")
    if len(rows):
        d2 = np.sum((xy[rows] - xy[rows].mean(axis=0)) ** 2, axis=1)
        values["rmsd_nm"] = float(np.sqrt(d2.mean()))
        blur = float(np.mean(sigma[rows] ** 2)) if sigma is not None else 0.0
        values["radius_nm"] = float(np.sqrt(2 * max(d2.mean() - blur, 0.0)))
    return values


def measure_rows(locs, rows, precision_nm: float = 20.0) -> Dict[str, Any]:
    """An example: the localizations ``rows`` of ``locs`` -- everything an
    ROI holds, whatever its shape -- measured as a candidate is
    (`measure_at`).  ``center`` is their centre of mass."""
    xy, finite, sigma, psf = positions(locs)
    # rows of the table, as `positions` drops the non-finite ones
    where = np.cumsum(finite) - 1
    rows = np.asarray(rows, dtype=np.int64)
    rows = where[rows[finite[rows]]] if len(rows) else rows
    precise = (np.ones(len(xy), bool) if sigma is None
               else np.asarray(sigma, float) < precision_nm)
    centre = xy[rows].mean(axis=0).tolist() if len(rows) else [float("nan")] * 2
    return {"center": centre, "values": _values_of(xy, sigma, psf, precise, rows)}


def measure_at(locs, centres, radius_nm: float, shift: bool = True,
               precision_nm: float = 20.0, shape: str = "circle"
               ) -> List[Dict[str, Any]]:
    """The values of a circle (or square) of half-width ``radius_nm`` at
    each centre, after the mean shift onto the localizations (`mean_shift`)
    unless ``shift`` is off: one dict per centre, ``center``, ``values``
    (`VALUES["similar"]`) and ``template_nm``, the half-width.

    ``rmsd_nm`` is Picasso's: the RMS distance of the localizations inside
    from their centre of mass -- a structure's size, with the precision in
    it.  ``radius_nm`` is a disc's radius from the same spread, the
    precision taken out, as `measure_disc` takes it:
    ``sqrt(2 (<d^2> - <sigma^2>))``."""
    from scipy.spatial import cKDTree
    p = _norm(shape)
    xy, _, sigma, psf = positions(locs)
    centres = np.array(centres, dtype=float).reshape(-1, 2)
    if not len(xy):
        return [{"center": c.tolist(), "template_nm": float(radius_nm),
                 "values": {**{k: float("nan") for k in VALUES["similar"]},
                            "n_locs": 0, "touching": 0}} for c in centres]
    tree = cKDTree(xy)
    if shift:
        centres = mean_shift(xy, centres, radius_nm, tree, shape=shape)
    precise = (np.ones(len(xy), bool) if sigma is None
               else np.asarray(sigma, float) < precision_nm)
    return [{"center": c.tolist(), "template_nm": float(radius_nm),
             "values": _values_of(xy, sigma, psf, precise,
                                  np.asarray(rows, dtype=np.int64))}
            for c, rows in zip(centres, tree.query_ball_point(centres, radius_nm, p=p))]


def similarity_window(examples: Sequence[Dict[str, Any]], names: Sequence[str],
                      spread: float = 2.0) -> Dict[str, Tuple[float, float]]:
    """``{name: (low, high)}``: the examples' mean of each value plus or
    minus ``spread`` standard deviations, as Picasso's *std range*; at least
    3 localizations (`MIN_LOCS`), the fewest a candidate has.  Needs two
    examples with the value, for a spread to be taken at all.

    Two departures from Picasso, both because a handful of examples
    underestimates the spread of the structures they stand for: the standard
    deviation is the sample one (``ddof=1``), and that of ``n_locs`` is at
    least ``sqrt(mean)``, a count's Poisson spread, which labelling and
    blinking only widen.  On 150 simulated discs of 40 localizations, five
    examples gave a spread of 3.6 against the true 9.6, and Picasso's window
    found 60 % of the others; with both, 80 %, and still none of the
    clutter beside them (`tests/test_segment_similar.py`)."""
    out: Dict[str, Tuple[float, float]] = {}
    for name in names:
        values = np.array([e["values"].get(name, np.nan) for e in examples], float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError(f"{len(values)} example{'s' * (len(values) != 1)} "
                             f"with {name}: at least 2 are needed to learn its range")
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        if name == "n_locs":
            sd = max(sd, float(np.sqrt(mean)))
        low, high = mean - spread * sd, mean + spread * sd
        if name == "n_locs":
            low = max(low, float(MIN_LOCS))
        out[name] = (low, high)
    return out


def window_condition(window: Dict[str, Tuple[float, float]]) -> str:
    """The window as a *keep* condition, ``12.5 <= n_locs <= 30.1 and ...``,
    which a person can read, `judge` applies and `bounds` draws."""
    return " and ".join(f"{low:.4g} <= {name} <= {high:.4g}"
                        for name, (low, high) in window.items())


def similar_candidates(locs, radius_nm: float,
                       window: Optional[Dict[str, Tuple[float, float]]] = None,
                       existing: Sequence = (), precision_nm: float = 20.0,
                       shape: str = "circle") -> List[Dict[str, Any]]:
    """Every structure a circle (or square) of half-width ``radius_nm``
    settles on, measured (`measure_at`), the ones most like the examples
    first.

    The circles start on a triangular lattice of spacing ``radius_nm`` over
    the localizations -- no point of the field further than ``R/sqrt(3)``
    from one, so a structure smaller than the circle is never missed between
    them; Picasso's spacing of one diameter leaves gaps of ``0.58 d`` --
    those with fewer than 3 localizations dropped, and each is moved onto
    the localizations by the mean shift.  Many settle on the same
    structure.  They are ranked by how far their values lie from the middle
    of ``window`` (the sum of each value's distance in half-widths, squared;
    unranked without one), and a candidate overlapping a better one or an
    ``existing`` centre too much (`too_close`, the circle template's rule)
    is left out -- so a candidate on junk never pushes out a good one beside
    it, as Picasso's first-come order on its grid can.  ``use`` is unset
    until `judge`."""
    from scipy.spatial import cKDTree
    p = _norm(shape)
    R = float(radius_nm)
    if not R > 0:
        raise ValueError("the radius must be positive")
    xy, _, _, _ = positions(locs)
    if len(xy) < MIN_LOCS:
        return []
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    rows = np.arange(y0, y1 + R, R * np.sqrt(3) / 2)
    columns = np.arange(x0, x1 + R, R)
    if len(rows) * len(columns) > MAX_PIXELS:
        raise ValueError(f"an ROI of {2 * R:g} nm puts {len(rows) * len(columns):,} "
                         "circles on this field: choose a larger ROI size")
    gx = (columns[None, :] + (np.arange(len(rows)) % 2)[:, None] * R / 2).ravel()
    gy = np.repeat(rows, len(columns))
    grid = np.column_stack((gx, gy))
    tree = cKDTree(xy)
    held = tree.query_ball_point(grid, R, p=p, return_length=True)
    starts = grid[held >= MIN_LOCS]
    if not len(starts):
        return []
    settled = mean_shift(xy, starts, R, tree, shape=shape)
    # the circles that settled on one structure agree to the tolerance
    settled = np.unique(np.round(settled / SHIFT_TOL_NM), axis=0) * SHIFT_TOL_NM
    sites = [s for s in measure_at(locs, settled, R, shift=False,
                                   precision_nm=precision_nm, shape=shape)
             if s["values"]["n_locs"] >= MIN_LOCS]
    if window:
        def distance(site) -> float:
            total = 0.0
            for name, (low, high) in window.items():
                v = site["values"].get(name, np.nan)
                half = (high - low) / 2 or 1.0
                total += ((v - (low + high) / 2) / half) ** 2
            return total if np.isfinite(total) else np.inf
        sites.sort(key=distance)
    accepted = Accepted(existing)
    found = []
    for site in sites:
        if accepted.blocks(site["center"], R):
            continue
        accepted.add(site["center"], R)
        found.append(site)
    return found


def read_positions(path) -> Dict[str, Any]:
    """Positions from a delimited text file, by the csv reader's conventions
    (`io.formats`): the separator found from the header, a header of
    ``x``, ``xnm`` or ``x_nm`` (any case) for x and the same for y, and a
    unit in brackets -- ``x [um]`` -- in nm, um or mm, nm when none is given.
    Optionally the file each position belongs to: ``file_id`` (as *Export
    table* writes it), ``file`` (its name) or ``filenumber``.

    ``{"x_nm", "y_nm"}`` arrays, and each file column found as a list;
    rows without a finite x and y are left out."""
    import csv
    from pathlib import Path
    from .io.formats import CSV_NAMES, LENGTH_UNITS, _clean, csv_delimiter
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        f.seek(0)
        rows = list(csv.reader(f, delimiter=csv_delimiter(first.rstrip("\r\n"))))
    if not rows:
        raise ValueError(f"{path.name} is empty")
    header, body = rows[0], rows[1:]
    found: Dict[str, Tuple[int, float]] = {}
    files: Dict[str, int] = {}
    for i, cell in enumerate(header):
        name, unit = _clean(cell)
        target = CSV_NAMES.get(name)
        if target in ("x_nm", "y_nm") and target not in found:
            scale = LENGTH_UNITS.get(unit or "nm")
            if scale is None:
                raise ValueError(f"{path.name}: {cell!r} is in {unit}; positions "
                                 "are read in nm, um or mm")
            found[target] = (i, scale)
        elif name in ("file_id", "file", "filenumber") and name not in files:
            files[name] = i
    missing = [c for c in ("x_nm", "y_nm") if c not in found]
    if missing:
        raise ValueError(f"{path.name}: no column for {', '.join(missing)}; the "
                         f"headers are {', '.join(header)}")

    def number(row, i) -> float:
        try:
            return float(row[i])
        except (IndexError, ValueError):
            return float("nan")

    xs = np.array([number(r, found["x_nm"][0]) for r in body]) * found["x_nm"][1]
    ys = np.array([number(r, found["y_nm"][0]) for r in body]) * found["y_nm"][1]
    ok = np.isfinite(xs) & np.isfinite(ys)
    out: Dict[str, Any] = {"x_nm": xs[ok], "y_nm": ys[ok]}
    for name, i in files.items():
        cells = [(r[i].strip() if i < len(r) else "") for r, good in zip(body, ok) if good]
        if name == "filenumber":
            cells = [int(float(c)) if c else "" for c in cells]
        out[name] = cells
    return out


def mark_touching(sites: Sequence[Dict[str, Any]], among=None) -> None:
    """``touching`` 1 on each candidate whose circle -- its measured
    ``radius_nm``, else the template's -- overlaps at all the circle of
    another candidate of ``among`` (a mask; default all), else 0."""
    for site in sites:
        site["values"]["touching"] = 0
    if len(sites) < 2:
        return
    from scipy.spatial import cKDTree
    among = np.ones(len(sites), bool) if among is None else np.asarray(among, bool)
    centres = np.array([s["center"] for s in sites], dtype=float)
    r = np.array([s["values"].get("radius_nm", np.nan) for s in sites], dtype=float)
    fallback = np.array([s.get("template_nm", 0.0) for s in sites], dtype=float)
    r = np.where(np.isfinite(r), r, fallback)
    r = np.where(np.isfinite(r), r, 0.0)
    if not r.max() > 0:
        return
    tree = cKDTree(centres)
    for i, j in tree.query_pairs(2 * float(r.max())):
        if np.hypot(*(centres[i] - centres[j])) < r[i] + r[j]:
            if among[j]:
                sites[i]["values"]["touching"] = 1
            if among[i]:
                sites[j]["values"]["touching"] = 1


# ------------------------------------------------------------------ keep

def _quoted(text: str) -> Tuple[str, Dict[str, str]]:
    """Backquoted names replaced by stand-ins, as `order.condition` does."""
    quoted: Dict[str, str] = {}

    def stand_in(match) -> str:
        quoted[f"_q{len(quoted)}"] = match.group(0)
        return f"_q{len(quoted) - 1}"

    return re.sub(r"`([^`]+)`", stand_in, text), quoted


def clauses(text: str) -> List[str]:
    """The parts of a condition joined by its top-level ``and``, as written:
    what a rejected candidate says it failed."""
    plain, quoted = _quoted(text.strip())
    if not plain:
        return []
    try:
        tree = ast.parse(plain, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot read keep: {error.msg}") from None
    parts = (tree.body.values if isinstance(tree.body, ast.BoolOp)
             and isinstance(tree.body.op, ast.And) else [tree.body])
    out = []
    for part in parts:
        segment = ast.get_source_segment(plain, part) or ""
        for stand, original in quoted.items():
            segment = re.sub(rf"\b{stand}\b", original, segment)
        out.append(segment.strip())
    return out


def bounds(text: str) -> Dict[str, List[float]]:
    """``{name: [numbers]}``: each number a condition compares a value with
    directly -- ``40 <= radius_nm <= 70`` gives ``radius_nm: [40, 70]`` --
    which a histogram of that value marks."""
    plain, quoted = _quoted(text.strip())
    out: Dict[str, List[float]] = {}
    if not plain:
        return out
    try:
        tree = ast.parse(plain, mode="eval")
    except SyntaxError:
        return out

    def name(node) -> Optional[str]:
        if isinstance(node, ast.Name):
            found = quoted.get(node.id, node.id)
            return found.strip("`")
        return None

    def number(node) -> Optional[float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = number(node.operand)
            return None if inner is None else -inner
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            terms = [node.left] + list(node.comparators)
            for a, b in zip(terms, terms[1:]):
                for n, v in ((name(a), number(b)), (name(b), number(a))):
                    if n is not None and v is not None:
                        out.setdefault(n, []).append(v)
    return out


def unavailable(locs) -> Dict[str, str]:
    """Values a method names but cannot measure on this table, and why."""
    if locs is not None and "sigma_nm" not in locs:
        return {"psf_nm": "the table has no sigma_nm attribute, whose mean psf_nm is"}
    return {}


def keep_test(text: str, provided: Sequence[str], missing: Dict[str, str] = None
              ) -> Tuple[Callable[[Dict[str, Any]], bool], List[str]]:
    """*keep* as a test of a candidate's values, and the names it reads.

    A name the method does not measure is an error naming the ones it does,
    and so is one it cannot measure here (``missing``: name -> why, such as
    ``psf_nm`` on a table without ``sigma_nm``): a condition on a value that
    is not there would otherwise fail, or pass, every candidate silently.
    An empty condition keeps everything."""
    from .roi_manager.order import condition
    if not text.strip():
        return (lambda values: True), []
    test, names = condition(text)
    unknown = [n for n in dict.fromkeys(names) if n not in provided]
    if unknown:
        raise ValueError(f"keep names {', '.join(unknown)}, which this method does "
                         f"not measure; it measures {', '.join(provided)}")
    for n in dict.fromkeys(names):
        if missing and n in missing:
            raise ValueError(f"keep names {n}, but {missing[n]}")
    return test, list(dict.fromkeys(names))


def judge(sites: Sequence[Dict[str, Any]], text: str, provided: Sequence[str],
          missing: Dict[str, str] = None) -> List[str]:
    """Each candidate's ``touching``, ``use`` and ``failed`` (the clauses of
    *keep* it fails) set in place; the names *keep* reads.

    ``touching`` is whether a candidate overlaps one that passes *keep* --
    *keep* first taken with every ``touching`` 0 -- not any candidate at
    all: the rejected ones are mostly background, beside every structure,
    and would flag them all."""
    from .roi_manager.order import condition
    test, names = keep_test(text, provided, missing)
    for site in sites:
        site["values"]["touching"] = 0
    first = [bool(test(site["values"])) for site in sites]
    mark_touching(sites, first)
    parts = clauses(text)
    tests = [(part, condition(part)[0]) for part in parts] if len(parts) > 1 else []
    for site in sites:
        values = site["values"]
        ok = bool(test(values))
        site["use"] = ok
        if ok:
            site["failed"] = []
        else:
            site["failed"] = [part for part, t in tests if not t(values)] or [text.strip()]
    return names
