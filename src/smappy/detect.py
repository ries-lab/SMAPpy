"""Candidate detection: image filtering and local-maximum finding.

The image is filtered to suppress noise and (for the difference-of-Gaussians)
the smooth background, then every strict local maximum above a cutoff becomes a
candidate.  Both stages work on a block of frames ``(n, y, x)`` at once.

Differences to SMAP's ``ImageFilter`` / ``PeakFinder``:

* no Anscombe transform (it belongs with a background estimate, which is not
  part of this pipeline yet) and no sCMOS variance weighting
* no background offset subtracted before filtering.  In SMAP that value only
  compensated the zero padding of the convolution at the image borders; we pad
  by edge replication instead, which removes the artifact at its source
* only the difference-of-Gaussians and Gaussian filters, and only the dynamic
  and absolute cutoffs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy import ndimage

try:
    from . import _fit3d
except ImportError:  # pure-Python fallback if the extension is not built
    _fit3d = None


# --------------------------------------------------------------------- filters
class ImageFilter(Protocol):
    """Turns a block of images into a block of detection images."""

    def __call__(self, images: np.ndarray, n_threads: int = 0) -> np.ndarray: ...


@dataclass(frozen=True)
class GaussFilter:
    """Gaussian smoothing; use when the background is flat or already removed."""

    sigma: float = 1.2
    use_extension: bool = True

    #: what a constant image becomes: itself (`Counts`)
    constant_response = 1.0

    def __call__(self, images: np.ndarray, n_threads: int = 0) -> np.ndarray:
        if self.sigma <= 0:
            return np.asarray(images, dtype=np.float32)
        radius = _gauss_radius(self.sigma)
        if self.use_extension and _fit3d is not None:
            return _fit3d.filter_images(
                _as_block(images), self.sigma, 0.0, radius,
                separable=radius >= _SEPARABLE_FROM_RADIUS, n_threads=n_threads)
        return _gaussian(images, self.sigma, radius)

    def __str__(self) -> str:
        return f"Gauss(sigma={self.sigma:g})"


@dataclass(frozen=True)
class DoGFilter:
    """Difference of Gaussians: smooths the PSF and removes smooth background.

    The wide Gaussian follows SMAP: ``max(1, 2.5 * sigma)``.  The kernel sums to
    zero exactly, so a constant background cancels.
    """

    sigma: float = 1.2
    use_extension: bool = True

    #: what a constant image becomes: nothing, the kernel sums to zero (`Counts`)
    constant_response = 0.0

    @property
    def sigma_wide(self) -> float:
        return max(1.0, 2.5 * self.sigma)

    def __call__(self, images: np.ndarray, n_threads: int = 0) -> np.ndarray:
        radius = _dog_radius(self.sigma)
        if self.use_extension and _fit3d is not None:
            # small kernel: one 2-D convolution with the subtracted kernels
            # (the 2-D DoG has rank 2, so it is not separable, but it can be
            # applied directly).  Large kernel: two Gaussians fused into one
            # separable traversal.
            return _fit3d.filter_images(
                _as_block(images), self.sigma, self.sigma_wide, radius,
                separable=radius >= _SEPARABLE_FROM_RADIUS, n_threads=n_threads)
        narrow = _gaussian(images, self.sigma, radius)
        wide = _gaussian(images, self.sigma_wide, radius)
        return narrow - wide

    def __str__(self) -> str:
        return f"DoG(sigma={self.sigma:g}, wide={self.sigma_wide:g})"


def _as_block(images: np.ndarray) -> np.ndarray:
    """A C-contiguous block ``(n, y, x)``, as the extension expects: float32,
    or raw uint16 counts, which it reads as they are (`Counts`)."""
    images = np.asarray(images)
    out = np.ascontiguousarray(images, dtype=np.uint16 if images.dtype == np.uint16
                               else np.float32)
    return out[None] if out.ndim == 2 else out


def _gauss_radius(sigma: float) -> int:
    """Kernel radius, following SMAP's ``max(5, ceil(3.5/2*s)*2+1)`` window."""
    return max(2, int(np.ceil(1.75 * sigma)))


def _dog_radius(sigma: float) -> int:
    """Kernel radius, following SMAP's ``max(ceil(6*s-1), 3)`` window."""
    window = max(int(np.ceil(6.0 * sigma - 1.0)), 3)
    return max(1, (window - 1) // 2)


# Below this radius a single 2-D convolution beats the separable passes: it
# does more arithmetic per pixel ((2r+1)^2 vs 4(2r+1)) but in one traversal
# with no temporaries, and for small kernels the traversal is what costs.
# Measured crossover on an M1 Pro.
_SEPARABLE_FROM_RADIUS = 4


def _gaussian(images: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """Separable Gaussian over the two image axes, edge-replicate padding."""
    images = np.asarray(images, dtype=np.float32)
    axes = (images.ndim - 2, images.ndim - 1)
    out = images
    for axis in axes:
        out = ndimage.gaussian_filter1d(out, sigma, axis=axis, mode="nearest",
                                        radius=radius, output=np.float32)
    return out


# ----------------------------------------------------------------- raw counts
@dataclass(frozen=True)
class Counts:
    """How a block's raw camera counts become photons, ``((counts - offset) *
    gain) / excess`` (`camera.convert`), for peak finding on the raw frames.

    Converting every pixel of every frame to photons was a quarter of peak
    finding's time on one core, and is needed only for the boxes that are
    fitted.  A filter is linear: what it makes of the counts is ``scale``
    times what it makes of the photons, plus the offset times its response to
    a constant -- none for a difference of Gaussians, whose kernel sums to
    zero, the offset itself for a Gaussian.  So the filter reads the counts, an
    absolute cutoff in photons is moved into counts, and the values that come
    back are moved into photons; the dynamic cutoff, a median and a quantile
    range, moves with the values and needs nothing.  The candidates are the
    photons' but for float rounding: a maximum within a few ulp of the cutoff
    may fall on the other side.  Only for one offset for all pixels; a noise
    map's per-pixel offsets go through the photons as before.
    """

    offset: float
    gain: float
    excess: float = 1.0

    @property
    def scale(self) -> float:
        """Photons per count, after the excess noise."""
        return float(np.float32(self.gain) / np.float32(self.excess))

    def photons(self, frames: np.ndarray, n_threads: int = 0) -> np.ndarray:
        """``frames`` converted, as `camera.convert` does."""
        from .camera import convert
        return convert(frames, self.offset, self.gain, self.excess, n_threads=n_threads)


# --------------------------------------------------------------------- cutoffs
@dataclass(frozen=True)
class AbsoluteCutoff:
    """Fixed threshold on the filtered image."""

    value: float

    def __call__(self, maxima_values: np.ndarray) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return f"absolute({self.value:g})"


@dataclass(frozen=True)
class DynamicCutoff:
    """Threshold derived from the distribution of local-maximum intensities.

    SMAP's ``getdynamiccutoff``: the cutoff is the median of the maxima plus
    ``factor`` times the slope of their 20-80 % quantile range.  Computed per
    frame, so it adapts to changing background without needing the whole
    movie.

    A quantile is the ceil(n p)-th smallest maximum, as SMAP's
    ``myquantilefast`` takes it.  Up to 0.3.1 it was ``np.quantile``'s linear
    interpolation -- numpy's default, and nothing records choosing it.  The
    two differ in most frames, by 0.04 % of the cutoff at 1500 maxima a frame
    and about 1 % at 140, and nearest rank is the higher -- it drops 0.01-0.14 %
    of the candidates, the ones at the threshold, and adds none.  Matching SMAP
    is the point of the port, and the kernel that computes it in C++ is the
    simpler for it (no interpolation to reproduce to the last bit).
    """

    factor: float = 1.7

    def __call__(self, maxima_values: np.ndarray) -> float:
        n = maxima_values.size
        if n == 0:
            return 0.0
        if n < 10:
            return float(np.mean(maxima_values) * self.factor)
        ranks = [math.ceil(n * p) - 1 for p in (0.2, 0.5, 0.8)]
        q20, q50, q80 = (float(q) for q in np.partition(maxima_values, ranks)[ranks])
        # 0.8 - 0.2 is what SMAP divides by, and 0.6000000000000001 in double
        slope = (q80 - q20) / (0.8 - 0.2)
        return q50 + slope * self.factor

    def thresholds(self, values: np.ndarray, starts: np.ndarray,
                   n_threads: int = 0) -> np.ndarray:
        """The cutoff of every segment ``values[starts[i]:starts[i+1]]`` at once.

        The same numbers as calling the cutoff once per segment, bit for bit,
        returned as float32 because that is what ``values > cutoff`` compared
        in: a Python float against a float32 array is rounded to float32 first.

        Called per frame, the cutoff was a Python loop in detection -- three
        quantiles a frame, 80 % of it numpy's dispatch: 9.3 ms for 200
        frames of 1500 maxima.  One lexsort over the block, the obvious
        vectorisation, is slower than that (52 ms, a comparison sort over
        300 k values), and sorting a padded frames x maxima array along its
        rows is 5.2 ms, most of it the sort.  A cutoff needs three order
        statistics, not a sort: `csrc/cutoff.hpp` finds them with nth_element,
        frames split over threads, in 1.1 ms.  Without the extension each
        segment goes through `__call__`, which is the definition anyway.
        """
        values = np.ascontiguousarray(values, dtype=np.float32)
        starts = np.ascontiguousarray(starts, dtype=np.int64)
        counts = np.diff(starts)
        kernel = getattr(_fit3d, "segment_cutoffs", None)
        if kernel is not None:
            out = kernel(values, starts, float(self.factor), 10, n_threads)
            # below ten maxima the rule is a mean, which the kernel leaves NaN
            rest = np.flatnonzero(np.isnan(out) & (counts > 0))
        else:
            out = np.full(counts.size, np.nan, np.float32)
            rest = np.flatnonzero(counts > 0)
        for i in rest:
            out[i] = self(values[starts[i]:starts[i + 1]])
        return out

    def __str__(self) -> str:
        return f"dynamic(factor={self.factor:g})"


# ------------------------------------------------------------------ candidates
@dataclass
class Candidates:
    """Peak candidates found in a block of frames (pixel coordinates)."""

    frame: np.ndarray  # frame index, int64
    x: np.ndarray  # column, int32
    y: np.ndarray  # row, int32
    value: np.ndarray  # intensity in the filtered image, float32

    def __len__(self) -> int:
        return self.frame.size

    def __getitem__(self, mask) -> "Candidates":
        return Candidates(self.frame[mask], self.x[mask], self.y[mask],
                          self.value[mask])

    @classmethod
    def empty(cls) -> "Candidates":
        return cls(np.empty(0, np.int64), np.empty(0, np.int32),
                   np.empty(0, np.int32), np.empty(0, np.float32))


_HOLLOW3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)


def local_maxima(images: np.ndarray, threshold: float = -np.inf,
                 use_extension: bool = True, n_threads: int = 0) -> tuple:
    """Strict 3x3 local maxima -> ``(frame, y, x, value)``.

    "Strict" means larger than all eight neighbours, as in SMAP's
    ``maximumfind2.c``; plateaus therefore yield no maximum.  The one-pixel
    border is excluded.  ``threshold`` skips pixels at or below it, which is
    only a speed-up: pass ``-inf`` (the default) when the cutoff still has to be
    derived from the distribution of all maxima.

    Uses the C++ search when the extension is available: eight pixels at a
    time compared with all eight neighbours without a branch
    (``csrc/maxima.hpp``), since without a threshold the scalar search's
    early exits mispredicted on noise.  The numpy path is kept as a reference
    implementation and produces identical results.
    """
    images = np.ascontiguousarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[None]

    if use_extension and _fit3d is not None:
        frame, y, x, value = _fit3d.find_maxima(images, float(threshold),
                                                n_threads)
        return frame.astype(np.int64), y.astype(np.int64), x.astype(np.int64), value

    centre = images[:, 1:-1, 1:-1]
    is_max = ((centre > images[:, :-2, :-2]) & (centre > images[:, :-2, 1:-1])
              & (centre > images[:, :-2, 2:]) & (centre > images[:, 1:-1, :-2])
              & (centre > images[:, 1:-1, 2:]) & (centre > images[:, 2:, :-2])
              & (centre > images[:, 2:, 1:-1]) & (centre > images[:, 2:, 2:]))
    if np.isfinite(threshold):
        is_max &= centre > threshold
    frame, y, x = np.nonzero(is_max)
    y += 1
    x += 1
    return frame, y, x, images[frame, y, x]


def find_candidates(filtered: np.ndarray, cutoff,
                    n_threads: int = 0, split=None, affine=None) -> Candidates:
    """All local maxima of ``filtered`` above the cutoff.

    The cutoff is evaluated per frame, from that frame's maxima, which is what
    makes :class:`DynamicCutoff` adaptive.

    ``split`` is ``(axis, position)`` in frame coordinates -- axis 0 for rows,
    1 for columns -- and makes it adaptive per *half* as well.  On a split
    frame the two halves are two detection channels that happen to share a
    sensor, and a dynamic cutoff pooled over both is set by whichever is
    brighter: the dim channel is then held to a threshold its own molecules
    never reach, and the fainter partner of every pair is simply not found.
    That is the difference between a two-colour dataset and a one-colour one
    with extra steps, so the two-channel engine always passes this.

    ``affine`` is ``(shift, scale)`` when ``filtered`` is a filter of raw
    counts (`Counts`): ``(filtered - shift) * scale`` is the filter of the
    photons.  The cutoff is in photons, and so are the values returned.
    """
    filtered = np.asarray(filtered, dtype=np.float32)
    if filtered.ndim == 2:
        filtered = filtered[None]
    fused = _fused(filtered, cutoff, n_threads, split, affine)
    if fused is not None:
        return fused[0]
    if affine is not None:
        filtered = (filtered - np.float32(affine[0])) * np.float32(affine[1])

    # an absolute cutoff can be applied during the search, which is cheaper
    fixed = cutoff(np.empty(0)) if isinstance(cutoff, AbsoluteCutoff) else -np.inf
    frames, rows, cols, values = local_maxima(filtered, threshold=fixed,
                                              n_threads=n_threads)
    if frames.size == 0:
        return Candidates.empty()
    if np.isfinite(fixed):
        return Candidates(frames, cols.astype(np.int32), rows.astype(np.int32),
                          values)

    side = _halves(rows, cols, split)
    applied = _cutoffs(frames, values, side, filtered.shape[0], cutoff,
                       1 if split is None else 2, n_threads)
    keep = values > applied[frames, side]
    return Candidates(frame=frames[keep].astype(np.int64),
                      x=cols[keep].astype(np.int32),
                      y=rows[keep].astype(np.int32),
                      value=values[keep])


def _fused(filtered: np.ndarray, cutoff, n_threads: int = 0, split=None, affine=None):
    """``(candidates, cutoffs)`` of a filtered block in the extension's one
    pass (`csrc/detect.hpp`), or None where it cannot: no extension, or a
    cutoff that is neither absolute nor dynamic.

    The numpy route searched every maximum (2 500 a 256 x 256 frame), handed
    them all to Python and grouped them by frame and half to set and apply the
    cutoffs -- 0.2 ms a frame of glue around the search.  Here each frame's
    maxima stay in a buffer of its thread and only the candidates come back.
    A half with fewer than ten maxima comes back whole, its cutoff
    `DynamicCutoff.__call__`'s mean rule here, so the candidates and cutoffs
    are the numpy route's, bit for bit (tested).  With ``affine``
    (`find_candidates`) an absolute cutoff is moved into the filter's units
    and the values and cutoffs back into photons.
    """
    kernel = getattr(_fit3d, "detect_candidates", None)
    if kernel is None or not isinstance(cutoff, (AbsoluteCutoff, DynamicCutoff)):
        return None
    dynamic = isinstance(cutoff, DynamicCutoff)
    threshold = 0.0 if dynamic else float(cutoff(np.empty(0)))
    if affine is not None and not dynamic:
        threshold = threshold / affine[1] + affine[0]
    axis, position = (-1, 0) if split is None else (int(split[0]), int(split[1]))
    found, cutoffs, deferred = kernel(
        filtered, dynamic, float(np.float32(threshold)),
        float(cutoff.factor) if dynamic else 0.0, axis, position, 10, int(n_threads))
    frame, y, x, value = found
    dframe, dy, dx, dvalue = deferred
    if affine is not None:
        shift, scale = np.float32(affine[0]), np.float32(affine[1])
        value, dvalue = (value - shift) * scale, (dvalue - shift) * scale
        cutoffs = (cutoffs - shift) * scale
    cutoffs = cutoffs.astype(np.float64)
    if dframe.size:
        side = _halves(dy, dx, split)
        groups, which = np.unique(dframe.astype(np.int64) * 2 + side, return_inverse=True)
        applied = np.empty(groups.size, np.float32)
        for g, key in enumerate(groups):
            applied[g] = np.float32(cutoff(dvalue[which == g]))
            cutoffs[key // 2, key % 2] = applied[g]
        keep = dvalue > applied[which]
        if keep.any():
            frame = np.concatenate([frame, dframe[keep]])
            y = np.concatenate([y, dy[keep]])
            x = np.concatenate([x, dx[keep]])
            value = np.concatenate([value, dvalue[keep]])
            # back into the search's order: by frame, then row-major
            order = np.lexsort((x, y, frame))
            frame, y, x, value = frame[order], y[order], x[order], value[order]
    return Candidates(frame=frame.astype(np.int64), x=x, y=y, value=value), cutoffs


def frame_cutoffs(filtered: np.ndarray, cutoff, n_threads: int = 0,
                  split=None) -> np.ndarray:
    """The cutoff :func:`find_candidates` applies to each frame, ``(n_frames,
    halves)``: one column, or two on a split frame -- the half below
    ``position`` first.  NaN where a dynamic cutoff had no maxima to set it.

    The same maxima and the same code as the search itself, so what a preview
    reports is what detection did.  A percentile of all pixels, which the
    preview once reported, is a different and much larger number (322 against
    8 on a test frame): the maxima are mostly noise peaks, and those are what
    a dynamic cutoff is set from.
    """
    filtered = np.asarray(filtered, dtype=np.float32)
    if filtered.ndim == 2:
        filtered = filtered[None]
    n_halves = 1 if split is None else 2
    if isinstance(cutoff, AbsoluteCutoff):
        return np.full((filtered.shape[0], n_halves), cutoff(np.empty(0)))
    fused = _fused(filtered, cutoff, n_threads, split)
    if fused is not None:
        return fused[1]
    frames, rows, cols, values = local_maxima(filtered, n_threads=n_threads)
    return _cutoffs(frames, values, _halves(rows, cols, split),
                    filtered.shape[0], cutoff, n_halves, n_threads)


def _halves(rows, cols, split) -> np.ndarray:
    """Which half of a split frame each maximum is in: 0 below the position,
    1 at or above it; all 0 without a split."""
    if split is None:
        return np.zeros(rows.size, dtype=np.intp)
    axis, position = split
    return ((rows if axis == 0 else cols) >= position).astype(np.intp)


def _cutoffs(frames, values, side, n_frames, cutoff, n_halves,
             n_threads: int = 0) -> np.ndarray:
    """The cutoff of each frame and half, from that frame's (half's) maxima,
    NaN where there are none.

    Each is a float32 value, as the filtered image is: a Python float compared
    with a float32 array is rounded to float32 first, and so must this be for
    ``values > cutoff`` to keep what it always kept.
    """
    out = np.full((n_frames, n_halves), np.nan)
    for half in range(n_halves):
        # maxima come out sorted by frame, and so does each half's share:
        # each frame is a contiguous segment of it
        index = np.flatnonzero(side == half)
        part = values[index]
        starts = np.searchsorted(frames[index], np.arange(n_frames + 1))
        if isinstance(cutoff, DynamicCutoff):
            out[:, half] = cutoff.thresholds(part, starts, n_threads)
            continue
        for f, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
            if stop > start:
                out[f, half] = np.float32(cutoff(part[start:stop]))
    return out


@dataclass
class PeakFinder:
    """Filter a block of frames and return the candidates above the cutoff."""

    filter: ImageFilter
    cutoff: object
    n_threads: int = 0  # 0 = one per core
    # (axis, position) in frame coordinates: threshold each half of a split
    # frame on its own, so a dim channel is not judged against a bright one
    split: Optional[tuple] = None

    def __call__(self, photons: np.ndarray, first_frame: int = 0,
                 n_threads: Optional[int] = None, counts: Optional[Counts] = None):
        """Returns ``(candidates, filtered)``; frame indices are absolute.

        With ``counts`` the frames are the camera's raw counts and are filtered
        as they are (`Counts`); the candidates are in photons all the same, and
        ``filtered`` is None: the filter of the counts is not the image a
        preview shows.
        """
        threads = self.n_threads if n_threads is None else n_threads
        affine = None
        if counts is not None:
            response = getattr(self.filter, "constant_response", None)
            if response is None:
                # a filter that does not say what it makes of an offset: photons first
                photons, counts = counts.photons(photons, threads), None
            else:
                affine = (float(counts.offset) * response, counts.scale)
        filtered = self.filter(photons, n_threads=threads)
        candidates = find_candidates(filtered, self.cutoff, n_threads=threads,
                                     split=self.split, affine=affine)
        candidates.frame = candidates.frame + first_frame
        return candidates, (None if affine is not None else filtered)

    def __str__(self) -> str:
        halves = "" if self.split is None else ", each half on its own"
        return f"PeakFinder({self.filter}, {self.cutoff}{halves})"

    def cutoffs(self, filtered: np.ndarray,
                n_threads: Optional[int] = None) -> np.ndarray:
        """The cutoff applied to each frame of ``filtered`` (and each half, on
        a split frame): :func:`frame_cutoffs`."""
        threads = self.n_threads if n_threads is None else n_threads
        return frame_cutoffs(filtered, self.cutoff, n_threads=threads,
                             split=self.split)
