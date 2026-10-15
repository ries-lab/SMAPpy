"""Operations on the pixels of an image layer: what *Process image* runs.

Each operation takes a plane (2D, any float units) and its sizes in pixels and
hands back a plane of the same shape; the plugin converts the settings' nm to
pixels with the image's pixel size.  A script calls these directly.

**Background subtraction** estimates a smooth background and subtracts it, in
three ways:

* *Gaussian*: the plane smoothed with a Gaussian of ``sigma``.  On its own
  that is a high-pass filter, and it takes a share of every bright structure
  with the background: a spot sits on a dip after it, and a dense patch of
  them loses its floor.  So the estimate is refined, ``rounds`` times: the
  pixels above the background by more than three robust standard
  deviations of the residual (1.4826 x its median absolute deviation) are
  taken for structure and replaced by the background under them, and the
  plane is smoothed again.  Clipping them at the threshold instead left a
  dense structure three SDs of lift on each of its pixels: on a simulated
  ring and lines at 15 % of the field, the background came out twice as far
  off.  The noise, cut only in its upper 0.1 %, barely biases it.
* *wavelet*: SMAP's (Tiff_remove_BG, `mywaveletfilter`), the coarse
  approximation of a wavelet transform at ``level``.  SMAP keeps the low-pass
  of a dual-tree complex wavelet transform; here it is the a trous transform
  with the B3-spline kernel (Starck), the isotropic one of the SMLM
  literature, whose approximation at level J is a smoothing over about 2^J
  pixels.  SMAP's refinement -- a coarser estimate at twice the level, the
  residual clipped at three standard deviations once, smoothed again -- is
  replaced by the Gaussian's rounds, so the two methods differ only in
  their smoothing kernel.
* *rolling ball* (Sternberg 1983, as ImageJ has it): the grey opening of the
  plane with a ball of ``radius`` -- the surface the ball's top reaches when
  it rolls under the plane.  It never rises above the plane, so no structure
  narrower than the ball is taken into the background at all, and no
  rounds are needed.  The ball's height is in the plane's units, as in
  ImageJ: for photons, hundreds against a radius of tens of pixels, the ball
  is nearly flat, and the opening is that with a disc.  As ImageJ does, the
  plane is first smoothed over 3 x 3 pixels (else the ball sits in the
  noise's minima, and the result is offset by about two noise SDs), and for
  a large ball shrunk by the minimum over blocks (2 above 10 pixels, 4 above
  30, 8 above 100) and the background interpolated back.

**The edges.**  Every method needs the plane beyond its edges; it is
extended by point reflection (``2 f(edge) - f(edge - d)``), which continues a
slope as a slope, so a gradient across the field is removed to the edge.
The mirror reflection usual in filters bends a slope back at the edge and
leaves a rim of the width of the kernel.  The price: a bright spot right at
the edge is reflected as a dip outside it, which pulls the background down
next to it.

**Local contrast normalisation** divides out how much the plane varies
locally, so that dim and bright parts of a cell can be looked at with one
contrast: ``(f - m) / sqrt(v + c^2 median(v))``, with ``m`` and ``v`` the
local mean and variance over a Gaussian window.  The floor -- ``c`` times
the median local SD -- keeps empty areas, where ``v`` is noise only, from
being blown up into texture.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

# the B3-spline kernel of the a trous transform: variance 1 pixel^2
B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
CLIP_SD = 3.0
MAD_TO_SD = 1.4826


def robust_sd(values: np.ndarray) -> float:
    """The SD of the bulk of ``values`` from their median absolute
    deviation: bright outliers do not count."""
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0
    return float(MAD_TO_SD * np.median(np.abs(v - np.median(v))))


def extend(plane: np.ndarray, width: int) -> np.ndarray:
    """``plane`` extended by ``width`` pixels by point reflection, which
    continues a slope (see the module docstring)."""
    if width <= 0:
        return np.asarray(plane, np.float64)
    return np.pad(np.asarray(plane, np.float64), int(width), mode="reflect",
                  reflect_type="odd")


def _crop(plane: np.ndarray, width: int) -> np.ndarray:
    return plane[width:-width, width:-width] if width > 0 else plane


def _gaussian(sigma_px: float) -> Tuple[Callable, int]:
    from scipy.ndimage import gaussian_filter
    width = int(math.ceil(4.0 * sigma_px)) + 1
    return (lambda p: gaussian_filter(p, sigma_px, mode="nearest", truncate=4.0)), width


def _a_trous(level: int) -> Tuple[Callable, int]:
    from scipy.ndimage import convolve1d

    def smooth(p):
        for j in range(int(level)):
            kernel = np.zeros(4 * 2 ** j + 1)
            kernel[:: 2 ** j] = B3
            p = convolve1d(convolve1d(p, kernel, axis=0, mode="nearest"),
                           kernel, axis=1, mode="nearest")
        return p
    return smooth, 2 * (2 ** int(level) - 1) + 1


def _smoothed(plane: np.ndarray, smooth: Callable, width: int) -> np.ndarray:
    """``smooth`` of ``plane`` extended by ``width`` pixels, cropped back."""
    return _crop(smooth(extend(plane, width)), width)


def _refined(plane: np.ndarray, smooth: Callable, width: int, rounds: int) -> np.ndarray:
    """``smooth`` of the plane, refined ``rounds`` times by replacing the
    pixels more than `CLIP_SD` robust SDs above the background by the
    background.  Each round extends the plane afresh: the background is
    only right where the kernel saw extended pixels, not out at their edge."""
    plane = np.asarray(plane, np.float64)
    background = _smoothed(plane, smooth, width)
    for _ in range(int(rounds)):
        sd = robust_sd(plane - background)
        bright = plane > background + CLIP_SD * sd
        background = _smoothed(np.where(bright, background, plane), smooth, width)
    return background


def gaussian_background(plane: np.ndarray, sigma_px: float, rounds: int = 5) -> np.ndarray:
    """The background under ``plane``: a Gaussian smoothing of ``sigma_px``,
    with bright structures taken out ``rounds`` times."""
    smooth, width = _gaussian(max(float(sigma_px), 1e-3))
    return _refined(plane, smooth, width, rounds)


def wavelet_background(plane: np.ndarray, level: int, rounds: int = 5) -> np.ndarray:
    """The background under ``plane``: the a trous approximation at
    ``level`` (about 2^level pixels), bright structures taken out
    ``rounds`` times."""
    smooth, width = _a_trous(max(int(level), 1))
    return _refined(plane, smooth, width, rounds)


def _shrink_factor(radius_px: float) -> int:
    # ImageJ's (`RollingBall`): larger balls on a coarser grid
    return 1 if radius_px <= 10 else 2 if radius_px <= 30 else 4 if radius_px <= 100 else 8


def _ball(radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """The footprint and the heights of a ball of ``radius`` pixels, its top
    at 0."""
    r = max(int(math.floor(radius)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (xx * xx + yy * yy).astype(np.float64)
    footprint = d2 <= radius * radius + 1e-9
    heights = np.where(footprint, np.sqrt(np.maximum(radius * radius - d2, 0.0)) - radius, 0.0)
    return footprint, heights


def rolling_ball_background(plane: np.ndarray, radius_px: float) -> np.ndarray:
    """The background under ``plane``: where the top of a ball of
    ``radius_px`` reaches as it rolls under it (a grey opening)."""
    from scipy.ndimage import grey_dilation, grey_erosion, map_coordinates, uniform_filter
    radius = max(float(radius_px), 1.0)
    shrink = _shrink_factor(radius)
    width = int(math.ceil(radius)) + 2 * shrink
    big = extend(plane, width)
    smoothed = uniform_filter(big, 3, mode="nearest")
    ny, nx = smoothed.shape
    if shrink > 1:
        my, mx = -(-ny // shrink) * shrink, -(-nx // shrink) * shrink
        padded = np.pad(smoothed, ((0, my - ny), (0, mx - nx)), mode="edge")
        small = padded.reshape(my // shrink, shrink, mx // shrink, shrink).min(axis=(1, 3))
    else:
        small = smoothed
    footprint, heights = _ball(radius / shrink)
    eroded = grey_erosion(small, footprint=footprint, structure=heights, mode="nearest")
    opened = grey_dilation(eroded, footprint=footprint, structure=heights, mode="nearest")
    if shrink > 1:
        # each small pixel stands for its block's centre
        rows = (np.arange(ny) - (shrink - 1) / 2.0) / shrink
        cols = (np.arange(nx) - (shrink - 1) / 2.0) / shrink
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        opened = map_coordinates(opened, [rr, cc], order=1, mode="nearest")
    return _crop(opened, width)


def local_contrast(plane: np.ndarray, sigma_px: float, floor: float = 0.5) -> np.ndarray:
    """``plane`` minus its local mean over a Gaussian window of
    ``sigma_px``, over its local SD; the SD is floored at ``floor`` times
    its median over the plane, so that flat areas stay flat."""
    smooth, width = _gaussian(max(float(sigma_px), 1e-3))
    plane = np.asarray(plane, np.float64)
    mean = _smoothed(plane, smooth, width)
    # the squared deviations have no slope to continue: mirrored
    deviation = np.pad((plane - mean) ** 2, width, mode="reflect")
    variance = np.maximum(_crop(smooth(deviation), width), 0.0)
    level = float(floor) ** 2 * float(np.median(variance))
    # a plane with no texture at all is flat, not rounding error blown up
    level = max(level, (1e-9 * float(np.abs(plane).max(initial=0.0))) ** 2, 1e-30)
    return (plane - mean) / np.sqrt(variance + level)


def subtract_background(plane: np.ndarray, method: str, *, sigma_px: float = 10.0,
                        radius_px: float = 10.0, level: int = 3,
                        rounds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """``(plane - background, background)`` by ``method``: "gaussian",
    "ball" or "wavelet"."""
    plane = np.asarray(plane, np.float64)
    if method == "gaussian":
        background = gaussian_background(plane, sigma_px, rounds)
    elif method == "ball":
        background = rolling_ball_background(plane, radius_px)
    elif method == "wavelet":
        background = wavelet_background(plane, level, rounds)
    else:
        raise ValueError(f"no background method {method!r}: gaussian, ball or wavelet")
    return plane - background, background

