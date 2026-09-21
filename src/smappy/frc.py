"""Fourier ring correlation: what the picture resolves, not what a fit claims.

Split the localizations into two halves, render both, and ask at which spatial
frequency the two images stop agreeing.  Above that frequency each half is
showing its own noise, so the structure there is not in the data -- whatever a
precision says.  The recipe is Nieuwenhuizen et al., *Nat Methods* 10, 557
(2013), and SMAP's ``Analyze/measure/FRCresolution`` is a port of the MATLAB
that came with it; this is the same recipe in numpy, with the departures noted
at the end.

FRC is not a precision, and the difference is the useful part.  It folds in
everything the picture depends on: the localization error, but also the
labeling density (an unlabeled structure is not resolved however well its
neighbours are placed), the drift left after correction, and the noise of a
thin dataset.  A precision says how well one molecule was placed; the FRC says
what can be seen.

**The split is by time, never by localization**, and the reason is narrower
than it is usually put.  Dealing individual localizations into the two halves
costs nothing as long as their errors are independent -- simulate that and the
two splits agree to a percent.  What it costs is everything a blink shares
with itself: the same background estimate, the same overlapping neighbour, the
same residual of a PSF that does not quite match, the same moment of the
drift.  Those are one error on one molecule, and if the blink is dealt into
both halves they both inherit it, agree about it, and the FRC counts a
systematic error as resolved structure.  Give each blink a common 15 nm error
in simulation and the honest split reports 57 nm where dealing localizations
reports 38 nm -- the error is real, and only the time split sees it.

Splitting into *blocks of frames* keeps a blink on one side, at the cost of the
few blinks that straddle a block boundary, which is why the number of blocks is
a setting and not a constant: more blocks mix the halves more evenly over the
acquisition, fewer blocks keep more blinks whole.

## The curve

With the two half-images ``F1`` and ``F2``, the correlation of a Fourier ring
of radius ``q`` is::

    FRC(q) = sum_ring Re(F1 conj(F2)) / sqrt(sum_ring |F1|^2 * sum_ring |F2|^2)

which is 1 where the halves agree and 0 where they are unrelated.  The images
are tapered first (a Tukey window over the outer eighth): a hard edge is a step
in the image, a step is a ridge of power along the axes of the transform, and
that ridge is correlated between the halves -- it would hold the curve up at
every frequency.

The resolution is where the smoothed curve first falls through 1/7, the
threshold of the paper: the frequency at which the two halves still share more
signal than noise by a factor that a rendered image needs to show a feature.
Its uncertainty is the standard deviation of the curve at the crossing divided
by the slope there, which is where a reading of the crossing can wander to.

## How to read it against a precision

The correlated part of the two halves carries the blur of the localization
error, ``exp(-4 pi^2 sigma^2 q^2)`` -- twice ``2 pi^2 sigma^2 q^2`` because two
independent images each carry it once -- while the noise carries none of it.
So the curve is::

    FRC(q) = S(q) phi(q) / (S(q) phi(q) + N),      phi = exp(-4 pi^2 sigma^2 q^2)

with ``S/N`` the signal-to-noise of the sampling, which is roughly how many
localizations there are per labelled molecule.  The threshold is crossed where
``phi = N / (6 S)``, and both terms matter: the FRC resolution is *not* a fixed
multiple of the precision, and there is no multiple that it cannot beat.  A
structure localized fifty times per molecule resolves to about ``2.6 sigma``,
because averaging that many repeats recovers the molecule's position far
better than one localization places it; at ten times per molecule it is
``3.1 sigma``; sparsely labelled, it is worse than either by any amount.
(Those two numbers are what this module returns on simulated data, to within a
tenth of a sigma, which is the check that the curve means what it says.)

That is why the blur envelope ``phi(q)`` is drawn beside the measured curve
rather than divided out of it.  It says where the localization error alone
starts to cost correlation, and the comparison is the question worth asking:
if the measured curve falls away well to the left of the envelope, the picture
is limited by labeling density, drift or counting noise, and better
localizations will not move it; if the two fall together, it is the precision
that is in the way and only more photons will help.  The envelope's own
crossing of 1/7, at ``2 pi sigma / sqrt(ln 7) = 4.50 sigma``, is reported as
one number on that scale -- the resolution this precision would give if signal
and noise were equal -- and not as a floor, which it is not.

(SMAP has a version of this comparison in `FRCresolution.m`, commented out, as
the *deblurred numerator* ``frc_nom ./ mean(exp(-4 pi^2 locprec^2 q^2))``.  It
is the same idea, but dividing amplifies the noise exponentially at exactly
the frequencies one wants to read, the result is no longer on the 0-1 scale
that 1/7 belongs to, and it deblurs with the CRLB -- the quantity being
checked.  Drawing the envelope instead keeps both curves on one axis and needs
nothing from the camera calibration.)

## Departures from the reference implementation

* The smoothing is a Savitzky-Golay filter of the same span rather than
  MATLAB's ``loess``; the uncertainty then propagates through the filter's own
  coefficients, which is exact for a linear filter and is what the reference
  approximates with a local least-squares covariance.
* The split is made several times over, with the blocks dealt at random, and
  the curves averaged -- the reference implementation does the same.  The
  spread of the resolutions across those repeats is then the error bar, which
  is the uncertainty that actually shows when the same data is split again;
  the analytic one from the ring statistics is reported when there is only one
  split to work with.
* The reference clips the brightest pixels (a quantile at 0.9999) before the
  transform.  That is a rendering habit -- it keeps a fiducial bead from
  dominating a picture -- and it changes the spectrum, so it is not done here.
* 3D data is projected along z.  A slab deeper than the resolution blurs its
  own picture, so cut a slab before asking, and read the answer as the
  resolution of that projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .render import FieldOfView, render_histogram

THRESHOLD = 1 / 7                  # the paper's, and what every implementation uses
TAPER = 0.25                       # Tukey window: the outer eighth of each side
DEFAULT_BLOCKS = 20
# the canvas, when the pixel size is left to the run.  Big enough that the
# Nyquist frequency is far above any resolution worth reporting, small enough
# that two transforms of it cost a second.
DEFAULT_PIXELS = 1024
MAX_PIXELS = 4096
MIN_LOCS = 1000                    # below this the curve is noise


@dataclass
class FRC:
    """One FRC curve and what it says."""
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))       # 1/nm
    curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    smoothed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    resolution: float = float("nan")          # nm
    error: float = float("nan")               # nm
    pixelsize: float = float("nan")
    size: int = 0                             # the square canvas, in pixels
    n_blocks: int = 0
    n_locs: Tuple[int, int] = (0, 0)          # in the two halves
    repeats: int = 1
    per_repeat: Tuple[float, ...] = ()        # the resolution of each split
    analytic_error: float = float("nan")      # from the ring statistics alone
    message: str = ""

    @property
    def ok(self) -> bool:
        return np.isfinite(self.resolution) and self.resolution > 0


def time_blocks(frame, n_blocks: int = DEFAULT_BLOCKS,
                assignment: str = "alternating", seed: int = 0) -> np.ndarray:
    """Which half of the split each localization belongs to.

    Blocks are equal stretches of *frames*, so the localizations of one blink
    stay together -- see the module docstring for why that decides the answer.
    ``alternating`` deals the blocks A, B, A, B, which spreads each half evenly
    over the acquisition and so over whatever drifted during it; ``random``
    deals them by coin toss, which is what the reference offers and is worth a
    look when the sample changes over time in a way the alternation could
    resonate with.
    """
    frame = np.asarray(frame)
    low, high = float(np.min(frame)), float(np.max(frame))
    n_blocks = max(int(n_blocks), 2)
    span = max(high - low, 1e-9)
    index = np.minimum(((np.asarray(frame, dtype=float) - low) / span
                        * n_blocks).astype(np.int64), n_blocks - 1)
    if assignment == "random":
        side = np.random.default_rng(seed).integers(0, 2, n_blocks).astype(bool)
    else:
        side = (np.arange(n_blocks) % 2).astype(bool)
    return side[index]


def taper(image: np.ndarray, alpha: float = TAPER) -> np.ndarray:
    """A Tukey window on both axes: the edge of the field is not a structure."""
    from scipy.signal.windows import tukey

    ny, nx = image.shape
    return image * np.outer(tukey(ny, alpha), tukey(nx, alpha))


def _ring_index(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Which ring each pixel of a centred transform falls in, and how many."""
    axis = np.arange(size) - size // 2
    radius = np.rint(np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
                     ).astype(np.int64)
    length = size // 2 + 1
    radius = np.minimum(radius, length)        # the corners go in a bin nobody reads
    counts = np.bincount(radius.ravel(), minlength=length + 1)[:length]
    return radius, counts


def frc_curve(first: np.ndarray, second: np.ndarray, rings=None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """The correlation of the two images, ring by ring, and the ring sizes.

    ``rings`` is what `_ring_index` returned for this size, for a caller that
    splits the same data several times: building it is a hypotenuse over every
    pixel of the canvas, which is not worth paying five times.
    """
    from scipy.fft import fft2, fftshift

    size = first.shape[0]
    radius, counts = _ring_index(size) if rings is None else rings
    length = len(counts)
    # scipy's, not numpy's: it threads, and the canvas was sized for it
    a = fftshift(fft2(taper(first), workers=-1))
    b = fftshift(fft2(taper(second), workers=-1))
    flat = radius.ravel()

    def ring_sum(values) -> np.ndarray:
        return np.bincount(flat, weights=np.asarray(values).ravel(),
                           minlength=length + 1)[:length]

    numerator = ring_sum(np.real(a * np.conj(b)))
    power = np.sqrt(ring_sum(np.abs(a) ** 2) * ring_sum(np.abs(b) ** 2))
    curve = np.divide(numerator, power, out=np.zeros(length),
                      where=power > 0)
    return np.clip(curve, -1.0, 1.0), counts


def _smoother(length: int) -> Tuple[np.ndarray, int]:
    """Savitzky-Golay coefficients of the reference's span, and that span."""
    from scipy.signal import savgol_coeffs

    window = max(int(np.ceil(length / 10)), 5)
    window += 1 - window % 2                   # odd, as a centred window must be
    window = min(window, length - (1 - length % 2))
    return savgol_coeffs(window, 2), window


def resolution_from_curve(curve: np.ndarray, counts: np.ndarray, size: int,
                          pixelsize: float, threshold: float = THRESHOLD) -> FRC:
    """Where the curve falls through the threshold, and how well that is known.

    The uncertainty follows the reference: the variance of an FRC value follows
    from the value itself and the number of pixels in its ring, corrected by
    ``ne_inv`` for the fact that neighbouring rings are not independent (it is
    read off the high-frequency end, where the true correlation is zero and
    whatever is left is the noise floor).  Here it then goes through the
    smoothing filter's own coefficients and the slope at the crossing, which is
    where a reading of it can wander to.
    """
    from scipy.signal import savgol_filter

    out = FRC(curve=np.asarray(curve, dtype=float), pixelsize=float(pixelsize),
              size=int(size))
    length = len(out.curve)
    # the frequency of ring k, in 1/nm: k cycles over the whole canvas
    out.q = np.arange(length) / (size * pixelsize)
    if length < 8:
        out.message = "too few rings to read a resolution"
        return out
    coefficients, window = _smoother(length)
    out.smoothed = savgol_filter(out.curve, window, 2, mode="nearest")

    crossing = _first_crossing(out.smoothed, threshold)
    if crossing is None:
        out.message = ("the curve never falls through 1/7: too few "
                       "localizations, or a pixel too big to reach the "
                       "frequencies that matter")
        return out
    index, fraction = crossing
    q = out.q[index] + fraction * (out.q[index + 1] - out.q[index])
    if q <= 0:
        out.message = "the crossing is at zero frequency"
        return out
    out.resolution = float(1.0 / q)

    # the noise floor, from the end of the curve where nothing is correlated
    negative = np.flatnonzero(out.curve < 0)
    start = int(negative[0]) if len(negative) else int(0.75 * length)
    tail = slice(start, length)
    safe_counts = np.maximum(counts, 1)
    ne_inv = float(np.mean(out.curve[tail] ** 2 * safe_counts[tail])) if \
        start < length - 1 else 1.0
    variance = ((1 + 2 * out.curve - out.curve ** 2) * (1 - out.curve) ** 2
                * max(ne_inv, 1e-12) / safe_counts)
    variance = np.clip(variance, 0, None)
    half = len(coefficients) // 2
    window_slice = slice(max(index - half, 0), min(index + half + 1, length))
    weights = coefficients[:window_slice.stop - window_slice.start] ** 2
    spread = float(np.sqrt(np.sum(weights * variance[window_slice])))
    slope = _slope(out.q, out.smoothed, index)
    if slope < 0 and np.isfinite(spread):
        # dq from the value's spread, and the resolution is 1/q
        out.analytic_error = out.error = float(
            out.resolution ** 2 * spread / abs(slope))
    return out


def _first_crossing(curve: np.ndarray, threshold: float
                    ) -> Optional[Tuple[int, float]]:
    """The first place a falling curve passes below the threshold."""
    for index in range(len(curve) - 1):
        high, low = curve[index], curve[index + 1]
        if high >= threshold > low:
            step = high - low
            return index, float((high - threshold) / step) if step > 0 else 0.0
    return None


def _slope(q: np.ndarray, curve: np.ndarray, index: int) -> float:
    lo, hi = max(index - 1, 0), min(index + 2, len(curve))
    if hi - lo < 2:
        return 0.0
    return float(np.polyfit(q[lo:hi], curve[lo:hi], 1)[0])


def blur_envelope(q, sigma: float) -> np.ndarray:
    """``exp(-4 pi^2 sigma^2 q^2)``: what the localization error leaves at q."""
    return np.exp(-4 * np.pi ** 2 * float(sigma) ** 2 * np.asarray(q, float) ** 2)


def envelope_resolution(sigma: float, threshold: float = THRESHOLD) -> float:
    """Where the blur envelope alone crosses the threshold: ``4.50 sigma``.

    Read it as the resolution this precision gives when the signal and the
    noise of the sampling are equal, and never as a limit: a densely sampled
    structure resolves better than this, because the repeats average down the
    localization error (see the module docstring).
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(2 * np.pi * sigma / np.sqrt(-np.log(threshold)))


def frc_resolution(x, y, frame, pixelsize: float = 0.0,
                   n_blocks: int = DEFAULT_BLOCKS, assignment: str = "alternating",
                   repeats: int = 5, seed: int = 0, max_pixels: int = MAX_PIXELS,
                   threshold: float = THRESHOLD, report=None) -> FRC:
    """The FRC resolution of these localizations, in nanometres.

    Takes arrays and gives numbers: no session, no plugin, no window.  The
    canvas is square (a ring is only a ring on a square grid) and covers
    everything handed in, which is what makes a ROI the field of view -- cut
    the table first and the answer is about what is left.

    With ``repeats`` above one the blocks are dealt at random that many times
    and the curves averaged.  One split is one draw, and two draws of the same
    data do not give the same number; the spread across the draws is the error
    bar that means something, and it is usually larger than what the ring
    statistics alone suggest.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = FRC(n_blocks=int(n_blocks))
    if len(x) < MIN_LOCS:
        out.message = f"{len(x)} localizations: too few for an FRC"
        return out

    span = max(float(np.max(x) - np.min(x)), float(np.max(y) - np.min(y)))
    if span <= 0:
        out.message = "the localizations have no extent"
        return out
    if pixelsize <= 0:
        # a pixel coarser than about a fifth of the resolution reads it too
        # large, and the resolution is what the run is for -- so find it
        # roughly on a cheap canvas first and then pick the pixel from it.
        # One extra transform pair, against a percent or two of bias.
        rough = frc_resolution(x, y, frame, pixelsize=span / DEFAULT_PIXELS,
                               n_blocks=n_blocks, assignment=assignment,
                               repeats=1, seed=seed, max_pixels=max_pixels,
                               threshold=threshold)
        pixelsize = (rough.resolution / 5 if rough.ok
                     else span / DEFAULT_PIXELS)
        if report:
            report(f"FRC: about {rough.resolution:.0f} nm on a coarse grid, "
                   f"measuring again at {pixelsize:.1f} nm")
    from scipy.fft import next_fast_len

    # a transform of 3346 pixels costs ten times one of 3360, because 3346 is
    # 2 x 7 x 239 and the algorithm wants small factors.  Rounding the canvas
    # up to the next good length and letting the pixel size follow costs a
    # fraction of a nanometre per pixel and buys that factor back.
    size = int(min(max(int(np.ceil(span / pixelsize)), 16), max_pixels))
    size = int(min(next_fast_len(size), max_pixels))
    pixelsize = span / size                # the canvas decides, so it stays square
    fov = FieldOfView(float(np.min(x)), float(np.min(y)), pixelsize, size, size)

    repeats = max(int(repeats), 1)
    if repeats > 1:
        assignment = "random"      # an alternating split has only one draw in it
    if report:
        report(f"FRC: {size} x {size} pixels of {pixelsize:.1f} nm, "
               f"{n_blocks} blocks, {repeats} split(s)")

    rings = _ring_index(size)
    curves, counts, singles, halves = [], rings[1], [], (0, 0)
    for repeat in range(repeats):
        side = time_blocks(frame, n_blocks, assignment, seed + repeat)
        if not side.any() or side.all():
            continue
        first = render_histogram(x[~side], y[~side], fov).weight.astype(np.float64)
        second = render_histogram(x[side], y[side], fov).weight.astype(np.float64)
        curve, counts = frc_curve(first, second, rings)
        curves.append(curve)
        halves = (int((~side).sum()), int(side.sum()))
        if repeats > 1:
            one = resolution_from_curve(curve, counts, size, pixelsize, threshold)
            if one.ok:
                singles.append(one.resolution)
    if not curves:
        out.message = "every localization landed in one half: too few frames"
        return out

    out = resolution_from_curve(np.mean(curves, axis=0), counts, size, pixelsize,
                                threshold)
    out.n_blocks, out.n_locs = int(n_blocks), halves
    out.repeats, out.per_repeat = len(curves), tuple(singles)
    if len(singles) >= 3:
        # what splitting the same data again would have given
        out.error = float(np.std(singles, ddof=1))
    return out
